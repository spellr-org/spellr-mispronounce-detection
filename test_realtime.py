import sounddevice as sd
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.backend.espeak.wrapper import EspeakWrapper

# Setup the Espeak library and the model
EspeakWrapper.set_library("/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

block_size = 32000  # 0.5 seconds at 16000 Hz
overlap = 8000     # 20% of the block size

def process_chunk(data):
    waveform = torch.tensor(data).float().cpu()
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        print("Transcribed:", transcription[0])

def callback(indata, frames, time, status):
    global buffer
    # Append new data to the buffer
    buffer = np.concatenate((buffer, indata[:, 0]))
    if len(buffer) >= block_size:
        process_chunk(buffer[:block_size])
        # Move the buffer ahead by the block size minus overlap
        buffer = buffer[block_size-overlap:]

def start_transcription():
    global buffer
    buffer = np.zeros(overlap, dtype='float32')  # Pre-fill buffer with size of overlap
    try:
        with sd.InputStream(callback=callback, samplerate=16000, channels=1, blocksize=block_size, dtype='float32'):
            input("Recording... Press Enter to stop.\n")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    start_transcription()