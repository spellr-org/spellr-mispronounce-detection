from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from datasets import load_dataset
import torch
import torchaudio
import subprocess

def transcribe_to_phoneme(audiopath, model, processor):
    # Load your MP3 file
    audio_path = audiopath
    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Tokenize
    input_values = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values

    # retrieve logits
    with torch.no_grad():
        logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

def get_ipa(text):
    # Get the IPA transcription using espeak
    command = ['espeak', '--ipa', '-q', text]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip()

def clean_ipa(ipa_string):
    # Remove stress markers from IPA transcriptions
    return ipa_string.replace("ˈ", "").replace("ˌ", "")

def phoneme_cost(a, b):
    # Define specific phoneme transformation costs
    if a == b:
        return 0  # No cost if identical
    if (a in "ˈˌ" and b not in "ˈˌ") or (b in "ˈˌ" and a not in "ˈˌ"):
        return 0.2  # Lower cost for stress differences, recognizing them as minor
    return 1  # Standard cost for other differences

def compare_ipa(ipa1, ipa2):
    # Clean IPA strings to focus on phonemes and stress markers
    ipa1 = clean_ipa(ipa1)
    ipa2 = clean_ipa(ipa2)  # Assuming ipa2 comes as a list of phonemes

    # Create a distance matrix
    dp = [[0] * (len(ipa2) + 1) for _ in range(len(ipa1) + 1)]

    # Initialize the matrix for base case
    for i in range(len(ipa1) + 1):
        dp[i][0] = i
    for j in range(len(ipa2) + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, len(ipa1) + 1):
        for j in range(1, len(ipa2) + 1):
            cost = phoneme_cost(ipa1[i - 1], ipa2[j - 1])
            dp[i][j] = min(dp[i - 1][j] + 1,        # Deletion
                           dp[i][j - 1] + 1,        # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution

    # Calculate similarity as 1 - (edit distance / max possible edits)
    max_len = max(len(ipa1), len(ipa2))
    edit_distance = dp[len(ipa1)][len(ipa2)]
    similarity = 1 - (edit_distance / max_len)
    return similarity





EspeakWrapper.set_library("/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
        
text = "Chad the bull likes to kick."
expected_phonemes = get_ipa(text)
spoken_phonemes = transcribe_to_phoneme("chad_the_bull_likes_to_kick.mp3", model, processor)
spoken_phonemes_bad = transcribe_to_phoneme("chad_the_bull_likes_to_kick_bad.mp3", model, processor)

similarity = compare_ipa(expected_phonemes, spoken_phonemes)
similarity_bad = compare_ipa(expected_phonemes, spoken_phonemes_bad)

print(f"Expected phonemes: {expected_phonemes}")
print(f"Spoken phonemes: {spoken_phonemes}")
print(f"Spoken phonemes bad: {spoken_phonemes_bad}")
print(f"Similarity: {similarity}")
print(f"Similarity bad: {similarity_bad}")