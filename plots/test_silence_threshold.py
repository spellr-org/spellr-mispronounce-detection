import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

def record_audio(duration=10, samplerate=16000):
    """Record audio for a given duration and samplerate."""
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return recording

def calculate_decibels(audio, frame_length=1024):
    """Calculate the decibel level of audio frames."""
    decibel_levels = []
    for i in range(0, len(audio), frame_length):
        # Calculate RMS of the audio frame
        rms = np.sqrt(np.mean(audio[i:i+frame_length]**2))
        # Convert RMS to decibels
        decibels = 20 * np.log10(rms + 1e-10)  # Avoid log(0) by adding a small constant
        decibel_levels.append(decibels)
    return decibel_levels

def plot_decibels(decibel_levels):
    """Plot decibel levels over samples."""
    plt.figure(figsize=(10, 4))
    plt.plot(decibel_levels)
    plt.title('Decibel Levels Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Decibels (dB)')
    plt.grid(True)
    plt.show()

# Record audio for 10 seconds
audio = record_audio(duration=10)
# Calculate decibel levels
decibel_levels = calculate_decibels(audio)
# Plot decibel levels
plot_decibels(decibel_levels)