import sounddevice as sd

fs = 16000

duration = 5  # seconds
sd.default.samplerate = fs
sd.default.channels = 2

print("listening...")
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)

sd.wait()

print("speaking...")

sd.play(myrecording, fs)

sd.wait()

print("done speaking")
