import sounddevice as sd
import torch
import numpy as np
import threading
import os
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import pronounce_score as ps
import phoneme_alignment as aligner

# Global flag to control the recording state
is_recording = True

# Directory to save audio chunks
chunk_directory = "audio_chunks_async"
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

def process_audio(data, model, processor):
    waveform = torch.tensor(data).float().cpu()
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    print("input_values shape:", input_values.shape)
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription[0].replace("ː", "").replace("ˈ", "").replace("ˌ", "")

def record_audio():
    global is_recording
    # Allocate space for up to 1 minute of recording
    myrecording = np.zeros((16000 * 60, 1), dtype='float32')
    with sd.InputStream(samplerate=16000, channels=1, dtype='float32') as stream:
        idx = 0
        while is_recording:
            data, overflowed = stream.read(1024)
            if idx + 1024 <= myrecording.shape[0]:
                myrecording[idx:idx + 1024] = data
                idx += 1024
            else:
                break  # Stop recording if exceeding 1 minute
        return myrecording[:idx]  # Return only the recorded part

def listen_for_stop():
    global is_recording
    input("Recording... Type enter to stop recording.\n")
    is_recording = False

# Setup the Espeak library and the model
EspeakWrapper.set_library("/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

text = "The dog jumped over the cat."
print(text)

thread = threading.Thread(target=listen_for_stop)
thread.start()
recorded_audio = record_audio()
thread.join()  # Ensure the stop listening thread has finished

print("Processing...")
phonemes = process_audio(recorded_audio[:, 0], model, processor)  # Process the recording

# Save the entire recording for playback analysis
save_chunk(recorded_audio, "complete_recording.mp3")

original_list = ps.text_to_phoneme(text).split(" ")
spoken_list = phonemes.split(" ")

print("Original: ", original_list)
print("Spoken: ", spoken_list)

distances = aligner.match(original_list, spoken_list)

# Print results
for i, (match, dist) in distances.items():
    print(f"Expected: {original_list[i]}, Best Match: {' '.join(match)}, Distance: {dist}")
