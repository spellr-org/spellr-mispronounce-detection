import sounddevice as sd
import torch
import numpy as np
import threading
import os
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import phoneme_alignment as aligner
from Levenshtein import distance as levenshtein_distance

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
        return transcription[0].replace("ː", "").replace("ˈ", "").replace("ˌ", "").replace("dʒ", "ʤ").replace("ɡ", "g").replace("ɔ", "ɑ").replace("ɚ", "ər")

def record_audio():
    global is_recording
    # Allocate space for up to 1 minute of recording
    myrecording = np.zeros((16000 * 60, 1), dtype='float32')

    num_batches = 16
    recent_volume = []
    intervening = False

    with sd.InputStream(samplerate=16000, channels=1, dtype='float32') as stream:
        idx = 0
        while is_recording:
            data, overflowed = stream.read(1024)

            rms = np.sqrt(np.mean(data**2))
            current_volume = 20 * np.log10(rms + 1e-10)
            recent_volume.append(current_volume)

            if len(recent_volume) > num_batches:
                recent_volume.pop(0)
            
            if len(recent_volume) == num_batches and all(volume < -40 for volume in recent_volume) and not intervening:
                intervening = True
                print("Do you need help?")
            
            if not all(volume < -40 for volume in recent_volume):
                intervening = False


            if idx + 1024 <= myrecording.shape[0]:
                myrecording[idx:idx + 1024] = data
                idx += 1024
            else:
                break                   # Stop recording if exceeding 1 minute
        return myrecording[:idx]        # Return only the recorded part

def listen_for_stop():
    global is_recording
    input("Recording... Type enter to stop recording.\n")
    is_recording = False

# Setup the Espeak library and the model
EspeakWrapper.set_library("/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

text = "The water in the lake was pure."
print(text)

thread = threading.Thread(target=listen_for_stop)
thread.start()
recorded_audio = record_audio()
thread.join()  # Ensure the stop listening thread has finished

print("Processing...")
spoken_list = process_audio(recorded_audio[:, 0], model, processor).split(" ")  # Process the recording

# Save the entire recording for playback analysis
save_chunk(recorded_audio, "complete_recording.mp3")

all_original_lists = aligner.text_to_phoneme_options(text)
original_list = aligner.text_to_phoneme(text)

print("Original: ", original_list)
print("Spoken: ", spoken_list)

distances = aligner.match(all_original_lists, spoken_list)

for i, (chunk, match, dist) in distances.items():
    original = chunk
    spoken = "".join(match).replace(" ", "")
    distance = levenshtein_distance(original, spoken)
    score = 1 - distance / len(original)
    if score < 0.8:
        print("mispronounced: ", text.split(" ")[i])