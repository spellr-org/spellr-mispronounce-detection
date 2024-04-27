import eng_to_ipa as ipa

def text_to_phoneme(text):
    result = ipa.convert(text).replace("ˈ", "").replace(".", "").replace("ˌ", "").replace(",", "").split(" ")
    return result

def text_to_phoneme_options(text):
    result = ipa.ipa_list(text)
    return [[element.replace("ˈ", "").replace(".", "").replace("ˌ", "").replace(",", "") for element in sublist] for sublist in result]

def levenshtein_distance(expected, spoken, i=0, j=0, memo={}):
    if (i, j) in memo:
        return memo[(i, j)]

    # If end of one list is reached, return the number of elements left in the other list
    if i == len(expected):
        return len(spoken) - j  # Cost of adding remaining spoken phonemes
    if j == len(spoken):
        return len(expected) - i  # Cost of deleting remaining expected phonemes

    if expected[i] == spoken[j]:
        # No operation needed if phonemes match
        cost = levenshtein_distance(expected, spoken, i + 1, j + 1, memo)
    else:
        # Compute costs of deletion, insertion, and substitution
        delete_cost = levenshtein_distance(expected, spoken, i + 1, j, memo) + 1
        insert_cost = levenshtein_distance(expected, spoken, i, j + 1, memo) + 1
        substitute_cost = levenshtein_distance(expected, spoken, i + 1, j + 1, memo) + 1
        cost = min(delete_cost, insert_cost, substitute_cost)

    memo[(i, j)] = cost
    return cost

def match(expected, spoken):
    distances = {}

    # Calculate Levenshtein distances for each expected chunk to spoken segments
    for index, chunks in enumerate(expected):
        # Starting at each possible segment in the spoken list
        min_distance = float('inf')
        for chunk in chunks:
            for start in range(len(spoken)):
                for end in range(start + 1, len(spoken) + 1):
                    spoken_segment = spoken[start:end]
                    distance = levenshtein_distance(chunk, spoken_segment, memo={})
                    if distance < min_distance:
                        min_distance = distance
                        best_chunk = chunk
                        best_match = spoken_segment
                        best_start = start
                        best_end = end
        distances[index] = (best_chunk, best_match, min_distance)

        # if best start is not at the very beginning
        # that means that the phonemes before are loose
        loose_phonemes = spoken[:best_start]

        # these need to be attatched to the previous chunk if index is not 0
        if index != 0:
            chunk, prev_match, prev_distance = distances[index - 1]
            # recompute distance with loose phonemes
            distances[index - 1] = (chunk, prev_match + loose_phonemes, prev_distance + len(loose_phonemes))

        
        # remove match from spoken list to avoid reusing
        spoken =  spoken[best_end:]
    
    return distances

# test main
if __name__ == "__main__":
    # Example usage
    expected = ['ðə', 'dɒɡ', 'dʒʌmpt', 'əʊvə', 'ðə', 'kat']
    expected_options = [['ði', 'ðə'], ['dɔg'], ['ʤəmpt'], ['oʊvər'], ['ði', 'ðə'], ['kæt']]
    spoken = ['ð', 'ə', 'd', 'ɑ', 'ɡ', 'dʒ', 'ʌ', 'm', 'p', 't', 'oʊ', 'v', 'ɚ', 'ð', 'ə', 'k', 'æ', 't']

    distances = match(expected_options, spoken)

    # Print results
    for i, (chunk, match, dist) in distances.items():
        print(f"Expected: {chunk}, Best Match: {' '.join(match)}, Distance: {dist}")
