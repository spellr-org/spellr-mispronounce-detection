import sounddevice as sd
import numpy as np
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from pydub import AudioSegment
import threading
import torch

# Constants
fs = 16000
duration = 5  # seconds, adjust as needed for longer recordings
channels = 1

# Initialize model and processor
EspeakWrapper.set_library("/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

def record_audio():
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()  # Wait until recording is finished
    print("Recording stopped.")
    return recording

def play_audio(audio):
    print("Playing...")
    sd.play(audio, fs)
    sd.wait()
    print("Playback finished.")

def process_audio(audio):
    waveform = np.squeeze(audio)  # Remove channel dimension if present
    input_values = processor(waveform, return_tensors="pt", sampling_rate=fs).input_values
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        print("Transcription:", transcription)
        return transcription

# Main function to handle threading
def main():
    recorded_audio = record_audio()  # Record audio first
    transcription = process_audio(recorded_audio)  # Process the recorded audio
    play_audio(recorded_audio)  # Play the recorded audio

if __name__ == "__main__":
    main()