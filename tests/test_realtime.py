import sounddevice as sd
import numpy as np
import threading
from queue import Queue
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import torch
import os
from pydub import AudioSegment

# Global flag to control the recording state
is_recording = True
audio_queue = Queue()  # Queue to hold audio data for processing

# Directory to save audio chunks
chunk_directory = "audio_chunks"
if not os.path.exists(chunk_directory):
    os.makedirs(chunk_directory)

def save_chunk(data, file_name):
    """Save the audio chunk to an MP3 file."""
    x = np.int16(data * 32767).reshape(-1, 1)
    # Convert the numpy array to audio segment
    audio_segment = AudioSegment(
        x.tobytes(),
        frame_rate=16000,
        sample_width=data.dtype.itemsize,
        channels=1
    )
    # Define the file path
    file_path = os.path.join(chunk_directory, file_name)
    # Export the audio segment to an MP3 file
    audio_segment.export(file_path, format="mp3", bitrate="256k")
    print(f"Saved recording to {file_path}")

def process_audio():
    global audio_queue
    idx = 0
    while is_recording or not audio_queue.empty():
        data = audio_queue.get()
        if data is None:
            continue  # Skip if data is None
        waveform = torch.tensor(data).float().cpu()
        input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            print("You:", transcription[0].replace("ː", "").replace("ˈ", "").replace("ˌ", ""))
        save_chunk(data, f"chunk_{int(idx)}.mp3")  # Save the audio chunk for debugging
        idx += 1
        audio_queue.task_done()

def record_audio():
    global is_recording, audio_queue
    silence_threshold = -40  # Set silence threshold in dB
    minimum_duration = int(16000 * 0.75)  # Minimum samples to consider for processing
    silence_duration = int(16000 * 0.4)  # Silence duration to consider as the end of a phrase
    buffer = np.zeros((16000 * 60, 1), dtype='float32')  # Buffer for 1 minute of recording
    last_chunk_start = 0

    with sd.InputStream(samplerate=16000, channels=1, dtype='float32') as stream:
        idx = 0
        keeper = False
        ready_to_stop = False
        last_silent_start = 0
        while is_recording:
            data, overflowed = stream.read(1024)
            if idx + 1024 <= buffer.shape[0]:
                buffer[idx:idx + 1024] = data
                rms = np.sqrt(np.mean(data**2))
                current_volume = 20 * np.log10(rms + 1e-10)  # Convert RMS to decibels

                if current_volume >= silence_threshold:
                    keeper = True
                if not keeper:
                    last_chunk_start = idx

                if current_volume < silence_threshold and not ready_to_stop:
                    last_silent_start = idx
                    ready_to_stop = True
                
                if ready_to_stop and current_volume >= silence_threshold:
                    ready_to_stop = False

                if current_volume < silence_threshold and (idx - last_chunk_start) >= minimum_duration and ready_to_stop and (idx - last_silent_start) >= silence_duration:
                    audio_queue.put(buffer[last_chunk_start:idx, 0].copy())  # Enqueue data for processing
                    last_chunk_start = idx
                    keeper = False
                idx += 1024
            else:
                break  # Stop recording if exceeding 1 minute
        # Enqueue any remaining audio
        if last_chunk_start < idx and (idx - last_chunk_start) >= minimum_duration:
            audio_queue.put(buffer[last_chunk_start:idx, 0].copy())

def listen_for_stop():
    global is_recording
    input("Recording... Type enter to stop recording.\n")
    is_recording = False
    audio_queue.put(None)  # Signal processing thread to finish

# Setup the Espeak library and the model
EspeakWrapper.set_library("/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

# Start processing thread
processing_thread = threading.Thread(target=(process_audio))
processing_thread.start()

# Start recording
record_thread = threading.Thread(target=record_audio)
record_thread.start()

# Listen for stop signal
listen_for_stop()

# Wait for threads to finish
record_thread.join()
processing_thread.join()
