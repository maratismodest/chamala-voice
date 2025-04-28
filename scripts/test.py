import torch
import torchaudio

# Create a dummy waveform (1 channel, 16000 samples)
waveform = torch.sin(torch.linspace(0, 1000, 16000))
waveform = waveform.unsqueeze(0)  # Add channel dimension

# Set the sample rate
sample_rate = 16000

# Save the audio file
torchaudio.save('output.wav', waveform, sample_rate)