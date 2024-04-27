import sounddevice as sd
import torch
import numpy as np
import threading
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import pronounce_score as ps
import phoneme_alignment as aligner

# Global flag to control the recording state
is_recording = True

def process_audio(data):
    waveform = torch.tensor(data).float().cpu()
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        # want to return a list of phonemes
        return transcription[0]

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

text = "The fan blows very fast."
print(text)

thread = threading.Thread(target=listen_for_stop)
thread.start()
recorded_audio = record_audio()
thread.join()  # Ensure the stop listening thread has finished
print("Processing...")
phonemes = process_audio(recorded_audio[:, 0])  # Process the recording

original_list = ps.text_to_phoneme(text).split(" ")
spoken_list = phonemes.split(" ")

print("Original: ", original_list)
print("Spoken: ", spoken_list)

distances = aligner.match(original_list, spoken_list)

# Print results
for chunk, (match, dist) in distances.items():
    print(f"Expected: {chunk}, Best Match: {' '.join(match)}, Distance: {dist}")