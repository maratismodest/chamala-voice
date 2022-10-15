# V3
import os
import torch
import torchaudio

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/tt/v3_tt.pt',
                                   local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

example_text = 'Алиса, мин сине бик ныкъ сагындым!'
sample_rate = 48000
speaker='dilyara'



audio_paths = model.apply_tts(text=example_text,
                        speaker=speaker,
                        sample_rate=sample_rate)

torchaudio.save('test_1.mp3',
                  audio_paths.unsqueeze(0),
                  sample_rate=sample_rate)