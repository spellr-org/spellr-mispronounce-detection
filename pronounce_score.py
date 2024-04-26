import subprocess

def text_to_phoneme(text):
    # Call espeak with the --ipa option to get phonemes in International Phonetic Alphabet (IPA)
    result = subprocess.run(['espeak', '--ipa', '-q', text], capture_output=True, text=True)
    return result.stdout.strip()

def __phoneme_cost(a, b):
    if a == b:
        return 0
    return 1

def __edit_cost(char):
    # Define lower cost for space and stress marks
    if char in " ˈˌ":
        return 0.1
    # Higher cost for modifying actual phonetic symbols
    return 10

def compare_ipa(expected, spoken):
    dp = [[0] * (len(spoken) + 1) for _ in range(len(expected) + 1)]

    # Initialize the matrix for base case
    for i in range(1, len(expected) + 1):
        dp[i][0] = dp[i-1][0] + __edit_cost(expected[i-1])
    for j in range(1, len(spoken) + 1):
        dp[0][j] = dp[0][j-1] + __edit_cost(spoken[j-1])

    # Fill the matrix
    for i in range(1, len(expected) + 1):
        for j in range(1, len(spoken) + 1):
            cost = __phoneme_cost(expected[i - 1], spoken[j - 1])
            dp[i][j] = min(dp[i - 1][j] + __edit_cost(expected[i-1]),  # Deletion
                        dp[i][j - 1] + __edit_cost(spoken[j-1]),  # Insertion
                        dp[i - 1][j - 1] + cost * 10)  # Substitution, if different, heavily penalized

    total_cost = dp[len(expected)][len(spoken)]
    max_cost = max(len(expected), len(spoken)) * 10  # Maximum cost assumes all characters are deleted/inserted
    similarity = 1 - (total_cost / max_cost)
    return similarity

def compare_text_to_phoneme(expected_text, spoken_phonemes):
    expected_phonemes = text_to_phoneme(expected_text)
    return compare_ipa(expected_phonemes, spoken_phonemes)