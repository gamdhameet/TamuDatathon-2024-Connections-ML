import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import random

def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    Enhanced function for generating a guess in the Connections AI game based on previous guesses,
    current correct groups, and clustering of word embeddings.
    """

    print("Participant guess: ", words)
    words = eval(words) if isinstance(words, str) else words

    # Load the Sentence-BERT model
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading model: {e}")
        return ["ERROR", "IN", "MODEL", "LOAD"], True

    # Preprocess words to uppercase and track already guessed words
    words = [word.upper() for word in words]
    correct_flattened = set(word.upper() for group in correctGroups for word in group) if correctGroups else set()
    remaining_words = [word for word in words if word not in correct_flattened]
    guess = []
    endTurn = False

    # Generate embeddings for each word
    embeddings = model.encode(remaining_words)

    # Perform Agglomerative Clustering to group words based on embeddings
    clustering = AgglomerativeClustering(n_clusters=4, metric='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings)

    # Organize words by cluster
    word_to_cluster = {remaining_words[i]: labels[i] for i in range(len(remaining_words))}

    # Handle "One Word Away" cases by modifying only one word from the last guess
    if isOneAway and previousGuesses:
        last_guess = previousGuesses[-1]
        for i, word in enumerate(last_guess):
            for replacement in remaining_words:
                potential_group = last_guess[:i] + [replacement] + last_guess[i+1:]
                if potential_group not in previousGuesses and len(set(potential_group)) == 4:
                    guess = potential_group
                    break
            if guess:
                break

    # General guessing strategy to find a new group of four words
    if not guess:
        for cluster_id in set(labels):
            potential_group = [word for word in remaining_words if word_to_cluster[word] == cluster_id]
            if len(potential_group) == 4 and potential_group not in previousGuesses:
                guess = potential_group
                break

    # Fallback to random sampling if clustering doesn't yield a unique group of four
    if not guess:
        guess = random.sample(remaining_words, 4)

    # Ensure the guess contains exactly four unique words
    guess = list(set(guess[:4]))

    print("Final guess being returned:", guess)

    # End the turn if strikes reach the limit or all words are correctly grouped
    if strikes >= 3 or len(correct_flattened) + len(guess) == 16:
        endTurn = True

    return guess, endTurn
