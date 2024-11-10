from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    print("Participant guess: ", words)
    words = eval(words) if isinstance(words, str) else words

    # Load Hugging Face model for sentence embeddings
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading Hugging Face model: {e}")
        return ["ERROR", "IN", "MODEL", "LOAD"], True

    # Preprocess words to uppercase and exclude words already grouped correctly
    words = [word.upper() for word in words]
    correct_flattened = set(word.upper() for group in correctGroups for word in group) if correctGroups else set()
    remaining_words = [word for word in words if word not in correct_flattened]
    guess = []
    endTurn = False

    # Generate embeddings and calculate cosine similarity
    embeddings = model.encode(remaining_words)
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find the most similar pairs of words
    grouped_words = set()
    potential_groups = []
    for i in range(len(remaining_words)):
        if remaining_words[i] not in grouped_words:
            similar_words = sorted(
                [(remaining_words[j], similarity_matrix[i][j]) for j in range(len(remaining_words)) if i != j],
                key=lambda x: -x[1]
            )[:3]  # Top 3 most similar words to form a group of 4
            group = [remaining_words[i]] + [word for word, _ in similar_words]
            if len(group) == 4 and group not in previousGuesses:
                potential_groups.append(group)
                grouped_words.update(group)
                if len(grouped_words) == 4:
                    break

    # Choose the first potential group
    guess = potential_groups[0] if potential_groups else random.sample(remaining_words, 4)

    # Adjust guess based on "One Word Away" feedback
    if isOneAway and previousGuesses:
        last_guess = previousGuesses[-1]
        for i, word in enumerate(last_guess):
            new_guess = last_guess[:i] + [remaining_words[i]] + last_guess[i+1:]
            if new_guess not in previousGuesses:
                guess = new_guess
                break

    # Ensure exactly four words are selected
    guess = guess[:4]
    print("Final guess being returned:", guess)

    # End turn if strikes reach limit or all words are grouped correctly
    if strikes >= 3 or len(correct_flattened) + len(guess) == 16:
        endTurn = True

    return guess, endTurn
