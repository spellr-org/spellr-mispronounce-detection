import subprocess

def get_phonemes(text):
    # Call espeak with the --ipa option to get phonemes in International Phonetic Alphabet (IPA)
    result = subprocess.run(['espeak', '--ipa', '-q', text], capture_output=True, text=True)
    return result.stdout.strip()

# Example usage:
phrase = "hello hello hallo"
phonemes = get_phonemes(phrase)
print("Phonemes:", phonemes)
