from dotenv import load_dotenv
load_dotenv()
import os
import io
import torch
import torchaudio
import uuid
# Используем библиотеку requests для отправки запроса
import requests

# FastAPI is a framework
from fastapi import FastAPI, HTTPException, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

# Constants
from templates import html_error_template, TTSRequest, local_file, title, description
from openai_chat_route import router as openai_router

app = FastAPI(title=title, description=description)

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

# Download the model if not present
if not os.path.isfile(local_file):
    torch.hub.download_url_to_file(
        'https://models.silero.ai/models/tts/tt/v3_tt.pt', local_file)

# Load the model
model = torch.package.PackageImporter(
    local_file).load_pickle("tts_models", "model")
model.to(device)

app.include_router(openai_router)


@app.post("/speech-to-text")
async def speech_to_text(file: bytes = File(...)):
    """Convert speech to text using Tatar ASR API"""
    try:
        # Подготовка формы для отправки
        form_data = {"file": ("audio.wav", file, "audio/wav")}

        # URL Tatar ASR API
        url = 'https://tat-asr.api.translate.tatar/listening/'

        response = requests.post(url, files=form_data)

        # Проверка ответа
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=f"ASR API error: {response.text}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/mp3")
async def text_to_speech_url(request: TTSRequest):
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

        myuuid = uuid.uuid4()
        url = 'static/audio/' + str(myuuid) + '.mp3'
        # Save audio to the buffer as MP3
        torchaudio.save(
            url,
            audio_tensor.unsqueeze(0),
            sample_rate=request.sample_rate,
            format="mp3"
        )
        link = 'https://chamala.tatar/' + url
        # Return the audio file
        return {"status": "ok", "url": link}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def home():
    """Serve the HTML page from an external file"""
    try:
        with open("static/dist/index.html", "r", encoding="utf-8") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        # Fallback HTML if file doesn't exist
        html_content = html_error_template
        return HTMLResponse(content=html_content, status_code=200)


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "model": "Silero TTS Tatar v3"}


# Create a static files directory to serve assets like CSS, JS, images
try:
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass

if __name__ == '__main__':
    import uvicorn

    # Get port from environment variable for Railway deployment
    # or use default 80 for production
    port = int(os.environ.get("PORT", 80))

    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=10)
    # uvicorn.run("main:app", host="0.0.0.0", port=port, timeout_keep_alive=10, reload=True)
