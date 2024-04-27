import sounddevice as sd
import numpy as np
import queue
import torch
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import os
import threading
import time

# Setup model and processor
EspeakWrapper.set_library("/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

# Audio settings
samplerate = 16000
channels = 1
chunk_size = 16000 

# Directory to save audio chunks
chunk_directory = "audio_chunks_v2"
if not os.path.exists(chunk_directory):
    os.makedirs(chunk_directory)

# Buffer to hold incoming audio
audio_queue = queue.Queue()
audio_buffer = np.array([])

def add_to_queue(indata, frames, time, status):
    """This is called for each audio block from the sound device."""
    if status:
        print(status)
    global audio_buffer
    audio_buffer = np.concatenate((audio_buffer, indata[:, 0]))

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
    audio_segment.export(file_path, format="mp3", bitrate="512k")
    print(f"Saved recording to {file_path}")

def process_audio_from_queue():
    global audio_buffer
    while True:
        # clear buffer to queue if needed
        if audio_buffer.size >= chunk_size:
            audio_queue.put(audio_buffer[:chunk_size].copy())
            save_chunk(np.float32(audio_buffer[:chunk_size]), f"chunk_{int(time.time())}.mp3")  # Save the chunk for debugging
            audio_buffer = audio_buffer[chunk_size:]

        # process queue
        if not audio_queue.empty():
            data = audio_queue.get()
            waveform = torch.tensor(data, dtype=torch.float32).cpu()
            input_values = processor(waveform, return_tensors="pt", sampling_rate=samplerate).input_values
            print("input_values shape:", input_values.shape)
            with torch.no_grad():
                logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)
                print("Transcription:", transcription[0].replace("ː", "").replace("ˈ", "").replace("ˌ", ""))

        if not is_recording:
            break

def record():
    global is_recording
    is_recording = True
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=add_to_queue):
        print("Recording started...")
        while is_recording:
            # Process audio from the queue in real-time
            process_audio_from_queue()

def stop_recording():
    global is_recording
    input("Recording... Press Enter to stop.\n")
    is_recording = False


stop_thread = threading.Thread(target=stop_recording)
stop_thread.start()

record()
stop_thread.join()
