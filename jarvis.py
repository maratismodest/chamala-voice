import torch
import sounddevice as sd
import time

language = 'tt'
model_id = 'v3_tt'
sample_rate = 48000
speaker = 'dilyara'
put_accent = True
put_yo = True
device = torch.device('cpu') # cpu или gpu
text = "Хауди Хо, друзья!!!"
example_text = "Мин яратам сине, Татарстан."
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)
model.to(device)


# воспроизводим
def va_speak(what: str):
    audio = model.apply_tts(text=what+"..",
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo)

    sd.play(audio, sample_rate * 1.05)
    time.sleep((len(audio) / sample_rate) + 0.5)
    sd.stop()
#va_speak('Мин яратам сине, Татарстан.')
# sd.play(audio, sample_rate)
# time.sleep(len(audio) / sample_rate)
# sd.stop()

audio_paths = model.save_wav(text=example_text,
                             speaker=speaker,
                             sample_rate=sample_rate)