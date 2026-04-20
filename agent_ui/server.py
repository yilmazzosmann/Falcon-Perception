import base64
import os
import json
import httpx
from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile
import google.generativeai as genai

app = FastAPI(title="Falcon Agentic UI Server")

# --- Orchestrator Configuration ---
# Change this between "ollama" and "gemini" to switch backends
ORCHESTRATOR_TYPE = "ollama" 
# Change this to whatever local model you have pulled in Ollama
OLLAMA_MODEL = "qwen3.5:0.8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Configure Google Generative AI if key is present
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    
model_name = "gemini-3.1-flash-lite-preview"

FALCON_PERCEPTION_URL = "http://localhost:7860/v1/predictions"

class TrackPayload(BaseModel):
    image: str # Base64 encoded image
    target: str

LAST_LOGGED_TARGET = None

@app.post("/api/orchestrate")
async def orchestrate(
    image: str = Form(...), # base64
    text: str = Form(None),
    audio: UploadFile = File(None)
):
    """
    Receives an image and a user prompt (text or raw audio).
    Returns the target object to track. 
    """
    if not text and not audio:
        raise HTTPException(400, "Must provide text or audio prompt.")

    # Remove the metadata from base64 if present (e.g. data:image/jpeg;base64,...)
    if image.startswith("data:image"):
        image = image.split(",")[1]
    
    img_bytes = base64.b64decode(image)

    # Prepare inputs for Gemini
    inputs = [
        {"mime_type": "image/jpeg", "data": img_bytes}
    ]

    prompt_instruction = (
        "You are a grounded visual reasoning assistant. The user has provided an image showing their current camera feed and a prompt. "
        "Your goal is to extract the exact referring expression they want to track. "
        "Extract the object class and any specific positional or state attributes the user provides (e.g. 'bottle on the right', 'broken cup').\n"
        "Falcon Perception is highly capable of identifying specific objects using these spatial relationships, so preserve them if the user specifies them.\n"
        "Keep the expression as a short, clean phrase of 1-6 words maximum.\n"
        "If the user asks a yes/no question about an object, extract the hidden object noun anyway.\n"
        "Respond ONLY with this clean referring expression. Do not output anything else. If no clear object is mentioned, respond with 'NONE'."
    )
    inputs.append(prompt_instruction)

    if text:
        inputs.append(f"User Prompt: {text}")

    if audio:
        # Save temp audio file
        audio_bytes = await audio.read()
        mime_type = audio.content_type or "audio/webm"
        inputs.append({"mime_type": mime_type, "data": audio_bytes})

    # ---------------------------------------------------------
    # Route 1: Local Ollama Model (Text Parsing Only)
    # ---------------------------------------------------------
    if ORCHESTRATOR_TYPE == "ollama":
        if audio and not text:
            print("WARNING: Ollama text-only models cannot process raw audio. You must type a text query.")
            return {"target": "NONE"}
        
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "system": prompt_instruction,
                "prompt": text,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 300
                }
            }
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                print(f"\n====================== [ ORCHESTRATOR REQUEST ] ======================")
                print(f"-> Sending payload to Ollama: {OLLAMA_MODEL} (via /api/generate)")
                print(f"-> User Input: '{text}'")
                
                res = await client.post(OLLAMA_URL, json=payload)
                res.raise_for_status()
                
                # /api/generate returns `response`
                raw_response = res.json().get("response", "")
                print(f"\n<- Raw Ollama Response:\n{raw_response}")
                
                # Qwen might wrap the word in quotes or spaces, clean it aggressively
                content = raw_response.strip().replace('\n', '').replace('"', '').replace("'", "")
                print(f"<- Extracted Target: '{content}'")
                print(f"======================================================================\n")
                
                return {"target": content}
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            raise HTTPException(500, f"Error calling Ollama: {e}")

    # ---------------------------------------------------------
    # Route 2: Google Gemini (Multimodal Audio/Vision/Text)
    # ---------------------------------------------------------
    if not GEMINI_API_KEY:
        # Fallback if no key is provided, purely for testing
        print("WARNING: No GEMINI_API_KEY provided. Returning mock response.")
        target = text if text else "bottle"
        return {"target": target}
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(inputs)
        content = response.text.strip().replace("\n", "")
        return {"target": content}
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        raise HTTPException(500, f"Error calling LLM: {str(e)}")

@app.post("/api/track")
async def track(payload: TrackPayload):
    """
    Proxies requests to the local Falcon Perception engine.
    """
    # Base64 string sent by the frontend
    b64_image = payload.image
    if b64_image.startswith("data:image"):
        b64_image = b64_image.split(",")[1]

    falcon_payload = {
        "image": {"base64": b64_image},
        "query": payload.target,
        "task": "detection",
        "max_tokens": 2048,
        "min_image_size": 256,
        "max_image_size": 512
    }
    
    global LAST_LOGGED_TARGET
    if payload.target != LAST_LOGGED_TARGET:
        print(f"\n======================= [ FALCON INFERENCE ] =======================")
        print(f"-> Sending first frame to Falcon Perception")
        print(f"-> Query Target: '{payload.target}'")
        print(f"======================================================================\n")
        LAST_LOGGED_TARGET = payload.target

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(FALCON_PERCEPTION_URL, json=falcon_payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise HTTPException(500, f"Error querying Falcon Perception: {e}")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
