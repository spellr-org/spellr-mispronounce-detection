# TODO: IMPROVE THIS ALGORITHM


def phoneme_cost(a, b):
    if a == b:
        return 0
    return 1

def edit_cost(char):
    # Define lower cost for space and stress marks
    if char in " ˈˌ":
        return 0.1
    # Higher cost for modifying actual phonetic symbols
    return 10

def compare_ipa(expected, spoken):
    dp = [[0] * (len(spoken) + 1) for _ in range(len(expected) + 1)]

    # Initialize the matrix for base case
    for i in range(1, len(expected) + 1):
        dp[i][0] = dp[i-1][0] + edit_cost(expected[i-1])
    for j in range(1, len(spoken) + 1):
        dp[0][j] = dp[0][j-1] + edit_cost(spoken[j-1])

    # Fill the matrix
    for i in range(1, len(expected) + 1):
        for j in range(1, len(spoken) + 1):
            cost = phoneme_cost(expected[i - 1], spoken[j - 1])
            dp[i][j] = min(dp[i - 1][j] + edit_cost(expected[i-1]),  # Deletion
                           dp[i][j - 1] + edit_cost(spoken[j-1]),  # Insertion
                           dp[i - 1][j - 1] + cost * 10)  # Substitution, if different, heavily penalized

    total_cost = dp[len(expected)][len(spoken)]
    max_cost = max(len(expected), len(spoken)) * 10  # Maximum cost assumes all characters are deleted/inserted
    similarity = 1 - (total_cost / max_cost)
    return similarity

# Example usage
expected_phonemes = "tʃˈad ðə bˈʊl lˈaɪks tə kˈɪk"
spoken_phonemes_good = "tʃ æ d ð ə b oʊ l l aɪ k s t ə k ɪ k"
spoken_phonemes_bad = "k æ d ð ə b ʊ l l aɪ k s t ə b ɪ k"

print("Similarity for good pronunciation:", compare_ipa(expected_phonemes, spoken_phonemes_good))
print("Similarity for bad pronunciation:", compare_ipa(expected_phonemes, spoken_phonemes_bad))
