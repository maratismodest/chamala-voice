import os
import torch
import torchaudio
from flask import Flask, request, send_file, jsonify
import tempfile
import re

app = Flask(__name__)

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
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
    return safe_name[:50]  # Limit filename length


@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech and return audio file"""
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Text is required'}), 400

    # Get parameters from request or use defaults
    text = data['text']
    speaker = data.get('speaker', 'dilyara')
    sample_rate = int(data.get('sample_rate', 48000))
    put_accent = data.get('put_accent', True)

    # Ensure text ends with punctuation for better synthesis
    if text and not text[-1] in '.!?':
        text = text + '.'

    # Generate safe filename
    safe_name = sanitize_filename(text)
    filename = f"{safe_name}_{speaker}_{sample_rate}.mp3"
    file_path = os.path.join(AUDIO_DIR, filename)

    try:
        # Generate audio
        audio_paths = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=sample_rate,
            put_accent=put_accent
        )

        # Save audio file
        torchaudio.save(
            file_path,
            audio_paths.unsqueeze(0),
            sample_rate=sample_rate
        )

        # Return the audio file
        return send_file(file_path, mimetype='audio/mpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/speakers', methods=['GET'])
def get_speakers():
    """Return available speakers"""
    # For the Tatar model, we know there's the 'dilyara' speaker
    # but there might be others - this could be expanded
    speakers = ['dilyara']
    return jsonify({'speakers': speakers})


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok', 'model': 'Silero TTS Tatar v3'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
