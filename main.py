# V3
import os
import io
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Tatar TTS API", description="Simple text-to-speech API for Tatar language")
from fastapi.middleware.cors import CORSMiddleware

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# Setup TTS model
device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

# Download the model if not present
if not os.path.isfile(local_file):
    torch.hub.download_url_to_file(
        'https://models.silero.ai/models/tts/tt/v3_tt.pt', local_file)

# Load the model
model = torch.package.PackageImporter(
    local_file).load_pickle("tts_models", "model")
model.to(device)


class TTSRequest(BaseModel):
    text: str
    speaker: str = "dilyara"
    sample_rate: int = 48000
    put_accent: bool = True


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech and return MP3"""
    text = request.text

    # Ensure text ends with punctuation for better synthesis
    if text and not text[-1] in '.!?':
        text = text + '.'

    try:
        # Generate audio
        audio_tensor = model.apply_tts(
            text=text,
            speaker=request.speaker,
            sample_rate=request.sample_rate,
            put_accent=request.put_accent
        )

        # Create a buffer for the audio
        buffer = io.BytesIO()

        # Save audio to the buffer as MP3
        torchaudio.save(
            buffer,
            audio_tensor.unsqueeze(0),
            sample_rate=request.sample_rate,
            format="mp3"
        )

        # Reset buffer position to the beginning
        buffer.seek(0)

        # Return the audio file
        return StreamingResponse(
            buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename=tatar_tts.mp3"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def home():
    """Serve a simple HTML page with a form for TTS conversion"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tatar TTS Web Interface</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            textarea { width: 100%; height: 100px; margin-bottom: 10px; }
            button { padding: 8px 16px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #45a049; }
            #audioContainer { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Tatar Text-to-Speech</h1>
        <textarea id="textInput" placeholder="Enter Tatar text here..."></textarea>
        <div>
            <button onclick="convertText()">Convert to Speech</button>
        </div>
        <div id="audioContainer">
            <audio id="audioPlayer" controls style="display: none;"></audio>
        </div>
        
        <script>
            async function convertText() {
                const text = document.getElementById('textInput').value;
                if (!text) return;
                
                try {
                    const response = await fetch('/tts', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            speaker: "dilyara",
                            sample_rate: 48000,
                            put_accent: true
                        }),
                    });
                    
                    if (!response.ok) {
                        throw new Error('API request failed');
                    }
                    
                    const audioBlob = await response.blob();
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    const audioPlayer = document.getElementById('audioPlayer');
                    audioPlayer.src = audioUrl;
                    audioPlayer.style.display = 'block';
                } catch (error) {
                    console.error('Error:', error);
                    alert('Failed to convert text to speech');
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "model": "Silero TTS Tatar v3"}


if __name__ == '__main__':
    import uvicorn

    # Get port from environment variable for Railway deployment
    # or use default 5000 for local development
    port = int(os.environ.get("PORT", 80))

    uvicorn.run(app, host="0.0.0.0", port=port)
