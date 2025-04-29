# Fallback HTML if file doesn't exist
from pydantic import BaseModel

html_error_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tatar TTS Web Interface</title>
        </head>
        <body>
            <h1>Something went wrong</h1>
            <p>Sorry, the Tatar TTS service is currently unavailable.</p>
        </body>
        </html>
        """


class TTSRequest(BaseModel):
    text: str
    speaker: str = "dilyara"
    sample_rate: int = 48000
    put_accent: bool = True


local_file = 'model.pt'

title = "Tatar TTS API"
description = "Simple text-to-speech API for Tatar language"
