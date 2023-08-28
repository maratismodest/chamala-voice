# V4
import os
import torch

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/cyr/v4_cyrillic.pt',
                                   local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

example_text = """<speak><p>
                                      <s>Мин почти кеше тавышы.</s>
                                      <s>Мин сезнең белән д+ус булырга телим.</s>
                                       <s>Минем дустым Америкадан.</s>
                                    </p></speak>"""
sample_rate = 48000
speaker='marat_tt'

audio_paths = model.save_wav(text=example_text,
                             speaker=speaker,
                             sample_rate=sample_rate)
