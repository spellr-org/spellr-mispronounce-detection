import sounddevice as sd
import numpy as np
import threading
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import torch

# Global flag to control the recording state
is_recording = True

def process_audio(data):
    waveform = torch.tensor(data).float().cpu()
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription[0].replace("ː", "").replace("ˈ", "").replace("ˌ", "")

def rms_frame(audio_frame):
    """Calculate RMS of the audio frame."""
    rms = np.sqrt(np.mean(audio_frame**2))
    return rms

def record_and_process_audio():
    global is_recording
    silence_threshold = -45  # Set silence threshold in dB
    minimum_duration = 20000  # Minimum samples to consider for processing
    buffer = np.zeros((16000 * 60, 1), dtype='float32')  # Buffer for 1 minute of recording
    last_chunk_start = 0

    with sd.InputStream(samplerate=16000, channels=1, dtype='float32') as stream:
        idx = 0

        keeper = False

        while is_recording:
            data, overflowed = stream.read(1024)
            if idx + 1024 <= buffer.shape[0]:
                buffer[idx:idx + 1024] = data
                rms = rms_frame(data)
                current_volume = 20 * np.log10(rms + 1e-10)  # Convert RMS to decibels

                if current_volume >= silence_threshold:
                    keeper = True

                if not keeper:
                    last_chunk_start = idx
                
                if current_volume < silence_threshold and idx - last_chunk_start > minimum_duration and keeper:
                    phonemes = process_audio(buffer[last_chunk_start:idx, 0])
                    print(phonemes)
                    # Reset chunk start
                    last_chunk_start = idx
                    keeper = False
                idx += 1024
            else:
                break  # Stop recording if exceeding 1 minute
        # Process any remaining audio
        if last_chunk_start < idx and (idx - last_chunk_start) >= minimum_duration:
            phonemes = process_audio(buffer[last_chunk_start:idx, 0])
            print(phonemes)

def listen_for_stop():
    global is_recording
    input("Recording... Type enter to stop recording.\n")
    is_recording = False

# Setup the Espeak library and the model
EspeakWrapper.set_library("/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

thread = threading.Thread(target=listen_for_stop)
thread.start()
record_and_process_audio()
thread.join()  # Ensure the stop listening thread has finished
