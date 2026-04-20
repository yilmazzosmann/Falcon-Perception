"""Falcon Perception — single-image inference.

Run detection or segmentation on one image.
If no image is provided, a sample is streamed from the PBench dataset.

Usage
-----
    python run_perception_single.py
    python run_perception_single.py --image photo.jpg --query "the red car"
    python run_perception_single.py --image https://example.com/photo.jpg --query dumplings
    python run_perception_single.py --task detection --query "all objects"
    python run_perception_single.py --engine-type batch
    python run_perception_single.py --hf-local-dir ./my_export/
"""

from pathlib import Path
from typing import Literal

import torch
import tyro

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    cuda_timed,
    load_and_prepare_model,
    setup_torch_config,
)
from falcon_perception.data import load_image, stream_samples_from_hf_dataset

setup_torch_config()


def mb(x: int) -> float:
    """Return megabytes for a byte count."""
    return float(x) / (1024 ** 2)


def print_cuda_mem(tag: str, device: torch.device | None = None) -> None:
    """Print allocated/reserved/peak GPU memory for `device`.

    Safe to call when CUDA isn't available (prints a message instead).
    """
    if not torch.cuda.is_available():
        print(f"[GPU MEM] {tag}: cuda not available")
        return
    device = device or torch.device("cuda")
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)
    print(f"[GPU MEM] {tag}: allocated={mb(allocated):.1f}MB reserved={mb(reserved):.1f}MB peak={mb(peak):.1f}MB")


def print_model_size(model: torch.nn.Module) -> None:
    """Print approximate size of model parameters in MB."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"[MODEL] params ~ {mb(param_bytes):.1f}MB")


def peak_during(fn, *args, device: torch.device | None = None, **kwargs):
    """Run `fn(*args, **kwargs)` and print peak GPU memory during the call.

    Returns the function's return value.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else None)
    if device is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
    out = fn(*args, **kwargs)
    if device is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        peak = torch.cuda.max_memory_allocated(device)
        print(f"[GPU PEAK] peak during {getattr(fn, '__name__', str(fn))}: {mb(peak):.1f}MB")
    return out


@torch.inference_mode()
def main(
    image: str | None = None,
    query: str | None = None,
    task: Literal["segmentation", "detection"] = "segmentation",
    hf_model_id: str | None = None,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
    device: str | None = None,
    dtype: Literal["bfloat16", "float32", "float"] = "bfloat16",
    engine_type: Literal["batch", "paged"] = "batch",
    flex_attn_safe: bool = True,
    out_dir: str = "./outputs/",
    compile: bool = False,
    cudagraph: bool = False,
):
    """Run Falcon Perception (detection/segmentation) on a single image.

    If --image is omitted, a sample is streamed from the PBench dataset.

    Use --flex-attn-safe on GPUs with limited per-SM shared memory
    (A40, RTX 3090/4090, L40) to avoid FlexAttention Triton OOM. See README.
    """
    kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1} if flex_attn_safe else {}
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=hf_model_id or PERCEPTION_MODEL_ID,
        hf_revision=hf_revision,
        hf_local_dir=hf_local_dir,
        device=device,
        dtype=dtype,
        compile=compile,
    )
    resolved_device = model.device
    # Report memory immediately after loading the model and show model param size
    try:
        print_cuda_mem("after load_and_prepare_model", resolved_device)
        print_model_size(model)
    except Exception:
        # Don't fail if instrumentation hits an unexpected edge
        pass

    if task == "segmentation" and not model_args.do_segmentation:
        print("Model does not support segmentation (do_segmentation=False), falling back to detection.")
        task = "detection"

    if image is not None:
        pil_image = load_image(image).convert("RGB")
    else:
        print("No --image provided, loading a demo sample ...")
        sample = stream_samples_from_hf_dataset("tiiuae/PBench", split="level_1")[0]
        pil_image = sample["image"]
        sample_query = sample.get("expression") or sample.get("expressions") or "all objects"
        if isinstance(sample_query, list):
            sample_query = ", ".join(str(q) for q in sample_query) if sample_query else "all objects"
        print(f"  Sample query: {sample_query!r}")
        if query is None:
            query = str(sample_query)

    if query is None:
        query = "all objects"

    w, h = pil_image.size
    print(f"  Task    : {task}")
    print(f"  Query   : {query!r}")
    print(f"  Image   : {w} x {h}")
    print()

    from falcon_perception.data import ImageProcessor

    image_processor = ImageProcessor(patch_size=16, merge_size=1)
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    input_image_path = out_path / "perception_input.jpg"
    pil_image.save(input_image_path)

    if engine_type == "paged":
        from falcon_perception.paged_inference import (
            PagedInferenceEngine,
            SamplingParams,
            Sequence,
        )
        from falcon_perception.visualization_utils import render_paged_inference_outputs

        engine = PagedInferenceEngine(
            model, tokenizer, image_processor,
            max_batch_size=1,
            max_seq_length=2048,
            n_pages=128,
            page_size=128,
            prefill_length_limit=2048,
            enable_hr_cache=False,
            capture_cudagraph=cudagraph,
            kernel_options=kernel_options or None,
        )

        # Report memory after creating paged engine / KV cache
        try:
            print_cuda_mem("after paged engine init", resolved_device)
        except Exception:
            pass

        prompt = build_prompt_for_task(query, task)
        sampling_params = SamplingParams(stop_token_ids=stop_token_ids)

        def _make_sequences():
            return [Sequence(
                text=prompt,
                image=pil_image,
                min_image_size=256,
                max_image_size=1024,
                task=task,
            )]

        # Warmup absorbs torch.compile cost
        print("Warmup run ...")
        warmup_seqs = _make_sequences()
        with cuda_timed(reset_peak_memory=False) as warmup_timer:
            engine.generate(
                warmup_seqs,
                sampling_params=sampling_params,
                use_tqdm=False,
                print_stats=False,
            )
        print(f"Warmup done in {warmup_timer.elapsed:.1f}s")
        try:
            print_cuda_mem("after warmup", resolved_device)
        except Exception:
            pass

        print("Running inference ...")
        sequences = _make_sequences()
        try:
            print_cuda_mem("before paged generate", resolved_device)
        except Exception:
            pass
        engine.generate(
            sequences,
            sampling_params=sampling_params,
            use_tqdm=True,
            print_stats=True,
        )
        try:
            print_cuda_mem("after paged generate", resolved_device)
        except Exception:
            pass

        seq = sequences[0]
        aux = seq.output_aux

        print(f"\n{'=' * 60}")
        print("Results")
        print("=" * 60)
        if task == "segmentation":
            print(f"  Masks : {len(aux.masks_rle)}")
        elif task == "detection":
            from falcon_perception.visualization_utils import pair_bbox_entries
            n_boxes = len(pair_bbox_entries(aux.bboxes_raw))
            print(f"  Boxes : {n_boxes}")

        render_paged_inference_outputs(sequences, image_processor, output_dir=out_dir, task=task)

        print(f"\n  Input image : {input_image_path}")
        sub = "masks" if task == "segmentation" else "boxes"
        print(f"  Output dir  : {out_path / sub}")

    elif engine_type == "batch":
        from falcon_perception.batch_inference import BatchInferenceEngine, process_batch_and_generate
        from falcon_perception.visualization_utils import render_batch_inference_outputs

        prompt = build_prompt_for_task(query, task)
        engine = BatchInferenceEngine(model, tokenizer, kernel_options=kernel_options or None)
        try:
            print_cuda_mem("after batch engine init", resolved_device)
        except Exception:
            pass
        batch_inputs = process_batch_and_generate(
            tokenizer,
            [(pil_image, prompt)],
            max_length=2048,
            min_dimension=256,
            max_dimension=512,
        )
        try:
            print_cuda_mem("after process_batch_and_generate (CPU)", None)
        except Exception:
            pass
        batch_inputs = {
            k: (v.to(resolved_device) if torch.is_tensor(v) else v)
            for k, v in batch_inputs.items()
        }
        try:
            print_cuda_mem("after moving batch_inputs to device", resolved_device)
        except Exception:
            pass
        _, aux_out = engine.generate(
            **batch_inputs,
            max_new_tokens=512,
            temperature=0.0,
            stop_token_ids=stop_token_ids,
            seed=42,
        )
        try:
            print_cuda_mem("after batch generate", resolved_device)
            print(f"  max during batch generate: {mb(torch.cuda.max_memory_allocated(resolved_device)):.1f}MB")
        except Exception:
            pass

        from falcon_perception.aux_output import AuxOutput
        from falcon_perception.visualization_utils import pair_bbox_entries
        n_dets = sum(
            len(pair_bbox_entries(a.bboxes_raw)) if isinstance(a, AuxOutput) else len(a) // (3 if task == "segmentation" else 2)
            for a in aux_out
        )

        print(f"\n{'=' * 60}")
        print("Results")
        print("=" * 60)
        print(f"  Masks : {n_dets}")

        batch_inputs["__orig_images__"] = [pil_image]
        render_batch_inference_outputs(
            "BATCH", batch_inputs, aux_out,
            [], task, out_dir=out_dir, queries=[query],
        )

        print(f"\n  Input image : {input_image_path}")
        print(f"  Output dir  : {out_path / 'masks'}")


if __name__ == "__main__":
    tyro.cli(main)
