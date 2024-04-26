from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from datasets import load_dataset
import torch
import torchaudio

# Set up Espeak properly (will be different for other OS)
EspeakWrapper.set_library("/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib")

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    
# Load your MP3 file
audio_path = 'hello_hello_hallo.mp3'
waveform, sample_rate = torchaudio.load(audio_path)

if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Tokenize
input_values = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values

# retrieve logits
with torch.no_grad():
    logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)