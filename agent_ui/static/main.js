const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const promptInput = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const recordBtn = document.getElementById('record-btn');
const statusBadge = document.getElementById('status-badge');
const trackingTargetLabel = document.getElementById('tracking-target');
const audioPill = document.getElementById('audio-pill');
const removeAudioBtn = document.getElementById('remove-audio-btn');

let trackingInterval = null;
let currentTarget = null;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let audioBlob = null;

// Initialize Webcam
async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            // Match canvas coordinates to the natural video size for accurate bbox mapping
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };
    } catch (e) {
        console.error("Camera access denied or unavailable", e);
        alert("Please allow camera access.");
    }
}

// Extract current video frame as base64 JPEG
function captureFrame() {
    // We use a temporary, off-screen canvas if we don't want to mess with the overlay
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    return tempCanvas.toDataURL('image/jpeg', 0.8);
}

// Orchestrator call
async function startOrchestration() {
    const text = promptInput.value.trim();
    if (!text && !audioBlob) {
        alert("Please provide a text or audio prompt.");
        return;
    }

    setUIState('loading');
    const imageB64 = captureFrame();

    const formData = new FormData();
    formData.append("image", imageB64);
    if (text) formData.append("text", text);
    if (audioBlob) {
        formData.append("audio", audioBlob, "prompt.webm");
    }

    try {
        const response = await fetch('/api/orchestrate', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.target && data.target.toUpperCase() !== "NONE") {
            startTracking(data.target);
        } else {
            alert("No target object found in your prompt.");
            setUIState('default');
        }
    } catch (e) {
        console.error("Orchestration failed:", e);
        alert("Failed to contact orchestrator.");
        setUIState('default');
    }
}

let isTracking = false;

// Async Tracking Loop to avoid queue buildup and latency
async function trackLoop() {
    if (!currentTarget || !isTracking) return;
    
    const currentFrame = captureFrame();
    
    try {
        const response = await fetch('/api/track', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: currentFrame, target: currentTarget })
        });
        const data = await response.json();
        
        // Prevent drawing a box if the user clicked Stop while this network request was in flight
        if (isTracking) {
            drawBoxes(data.masks);
        }
    } catch (e) {
        console.error("Tracking API error:", e);
    }
    
    // Once fetch resolves, call the loop again immediately (or with brief 30ms sleep) 
    // to strictly pull the newest frame
    if (isTracking) {
        setTimeout(trackLoop, 30);
    }
}

function startTracking(target) {
    currentTarget = target;
    isTracking = true;
    setUIState('tracking');
    trackingTargetLabel.textContent = target;
    
    trackLoop();
}

function stopTracking() {
    isTracking = false;
    currentTarget = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setUIState('default');
    trackingTargetLabel.textContent = "None";
}

// Draw transparent boxes with cyan edges
function drawBoxes(masks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!masks || masks.length === 0) return;

    ctx.lineWidth = 4;
    ctx.strokeStyle = '#22d3ee';
    ctx.fillStyle = 'rgba(34, 211, 238, 0.2)';

    masks.forEach(m => {
        if (!m.bbox) return;
        const [x1, y1, x2, y2] = m.bbox;
        const w = x2 - x1;
        const h = y2 - y1;
        
        ctx.beginPath();
        ctx.rect(x1, y1, w, h);
        ctx.fill();
        ctx.stroke();
    });
}

// UI State Manager
function setUIState(state) {
    if (state === 'default') {
        sendBtn.classList.remove('hidden');
        recordBtn.classList.remove('hidden');
        promptInput.classList.remove('hidden');
        stopBtn.classList.add('hidden');
        statusBadge.classList.add('hidden');
        sendBtn.classList.remove('loading');
        promptInput.value = '';
        audioBlob = null;
        recordBtn.style.color = '';
        audioPill.classList.add('hidden');
    } else if (state === 'loading') {
        sendBtn.classList.add('loading');
        statusBadge.classList.remove('hidden');
        trackingTargetLabel.textContent = "Thinking...";
    } else if (state === 'tracking') {
        // Switch to Stop button
        sendBtn.classList.add('hidden');
        recordBtn.classList.add('hidden');
        promptInput.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        statusBadge.classList.remove('hidden');
        sendBtn.classList.remove('loading');
    }
}

// Setup Audio Recording
async function setupAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };
        
        mediaRecorder.onstop = () => {
            audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioChunks = [];
            recordBtn.classList.remove('recording');
            
            // Show the audio pill instead of auto-sending
            audioPill.classList.remove('hidden');
        };
    } catch (e) {
        console.warn("Audio not available", e);
    }
}

// Event Listeners
sendBtn.addEventListener('click', startOrchestration);
stopBtn.addEventListener('click', stopTracking);
promptInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') startOrchestration();
});

removeAudioBtn.addEventListener('click', () => {
    audioBlob = null;
    audioPill.classList.add('hidden');
});

recordBtn.addEventListener('mousedown', () => {
    if (mediaRecorder && mediaRecorder.state === 'inactive') {
        audioChunks = [];
        mediaRecorder.start();
        recordBtn.classList.add('recording');
    }
});

recordBtn.addEventListener('mouseup', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
});
// Touch support for mobile
recordBtn.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (mediaRecorder && mediaRecorder.state === 'inactive') {
        audioChunks = [];
        mediaRecorder.start();
        recordBtn.classList.add('recording');
    }
});
recordBtn.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
});

// Boot
setupWebcam();
setupAudio();
