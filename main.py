import os
import torch
import torchaudio
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import re
from typing import Optional

app = FastAPI(title="Tatar TTS API", description="Text-to-speech API for Tatar language")

# Setup TTS model
device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'
AUDIO_DIR = 'audio'

# Create audio directory if it doesn't exist
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Download the model if not present
if not os.path.isfile(local_file):
    torch.hub.download_url_to_file(
        'https://models.silero.ai/models/tts/tt/v3_tt.pt', local_file)

# Load the model
model = torch.package.PackageImporter(
    local_file).load_pickle("tts_models", "model")
model.to(device)


def sanitize_filename(text):
    """Convert text to a safe filename"""
    # Replace special characters and limit length
    if not text or text.isspace():
        return "audio_sample"  # Default name for empty text
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
    safe_name = re.sub(r'_+', '_', safe_name)  # Replace multiple underscores with one
    return safe_name[:50].strip('_')  # Limit filename length and trim underscores


class TTSRequest(BaseModel):
    text: str
    speaker: str = "dilyara"
    sample_rate: int = 48000
    put_accent: bool = True


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech and return audio file"""
    text = request.text

    # Ensure text ends with punctuation for better synthesis
    if text and not text[-1] in '.!?':
        text = text + '.'

    # Generate safe filename
    safe_name = sanitize_filename(text)
    timestamp = str(int(time.time()))
    filename = f"{timestamp}_{safe_name}_{request.speaker}_{request.sample_rate}.mp3"
    file_path = os.path.join(AUDIO_DIR, filename)

    try:
        # Generate audio
        audio_paths = model.apply_tts(
            text=text,
            speaker=request.speaker,
            sample_rate=request.sample_rate,
            put_accent=request.put_accent
        )

        # Save audio file
        torchaudio.save(
            file_path,
            audio_paths.unsqueeze(0),
            sample_rate=request.sample_rate
        )

        # Return the audio file
        return FileResponse(file_path, media_type='audio/mpeg')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speakers")
async def get_speakers():
    """Return available speakers"""
    # For the Tatar model, we know there's the 'dilyara' speaker
    # but there might be others - this could be expanded
    speakers = ['dilyara']
    return {"speakers": speakers}


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "model": "Silero TTS Tatar v3"}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
