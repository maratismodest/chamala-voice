import os
import torch
import torchaudio
import io
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Tatar TTS API", description="Text-to-speech API for Tatar language")

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
    """Convert text to speech and stream audio"""
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

        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name

        # Save audio to the temporary file
        torchaudio.save(
            temp_filename,
            audio_tensor.unsqueeze(0),
            sample_rate=request.sample_rate
        )

        # Function to stream the file content
        def iterfile():
            with open(temp_filename, 'rb') as f:
                yield from f
            # Delete the temporary file after streaming
            os.unlink(temp_filename)

        # Create a buffer for the audio instead of a temp file
        buffer = io.BytesIO()

        # Save audio to the buffer
        torchaudio.save(
            buffer,
            audio_tensor.unsqueeze(0),
            sample_rate=request.sample_rate,
            format="wav"
        )

        # Reset buffer position to the beginning
        buffer.seek(0)


        # Stream the audio file
        return StreamingResponse(
            iterfile(),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=tatar_tts.wav"}
        )

    except Exception as e:
        # Make sure to clean up the temp file if an error occurs
        if 'temp_filename' in locals():
            try:
                os.unlink(temp_filename)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/speakers")
async def get_speakers():
    """Return available speakers"""
    # For the Tatar model, we know there's the 'dilyara' speaker
    speakers = ['dilyara']
    return {"speakers": speakers}

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "model": "Silero TTS Tatar v3"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)