import os
import torch
import torchaudio
import io
from pydub import AudioSegment
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
    text = request.text
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

        # First save as WAV
        wav_buffer = io.BytesIO()
        torchaudio.save(
            wav_buffer,
            audio_tensor.unsqueeze(0),
            sample_rate=request.sample_rate,
            format="wav"
        )
        wav_buffer.seek(0)

        # Convert WAV to MP3 using pydub
        wav_audio = AudioSegment.from_wav(wav_buffer)
        mp3_buffer = io.BytesIO()
        wav_audio.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0)

        return StreamingResponse(
            mp3_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename=tatar_tts.mp3"}
        )

    except Exception as e:
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
