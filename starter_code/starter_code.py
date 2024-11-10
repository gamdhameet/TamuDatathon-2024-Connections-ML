import numpy as np
from gensim.models import Word2Vec
import pickle

import random

def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    print("Participant guess: ", words)
    words = eval(words) if isinstance(words, str) else words

    def load_model(filepath="word2vec_nyt_connections.model"):
        model = Word2Vec.load(filepath)
        return model
    
    def load_clusters(filepath="word_clusters.pkl"):
        with open(filepath, "rb") as f:
            word_to_cluster = pickle.load(f)
        return word_to_cluster

    try:
        model_wv = load_model()
        word_to_cluster = load_clusters()
    except Exception as e:
        print(f"Error loading model or clusters: {e}")
        return ["ERROR", "IN", "MODEL", "LOAD"], True  

    # Initial setup of words and tracking guessed groups
    words = [word.upper() for word in words]
    correct_flattened = set(word.upper() for group in correctGroups for word in group) if correctGroups else set()
    remaining_words = [word for word in words if word not in correct_flattened]
    guess = []
    endTurn = False

    if isOneAway:
        # Try to improve by swapping one word
        last_guess = previousGuesses[-1] if previousGuesses else []
        for i, word in enumerate(last_guess):
            potential_group = last_guess[:i] + [remaining_words[i]] + last_guess[i+1:]
            if potential_group not in previousGuesses:
                guess = potential_group
                break
    else:
        # General guessing strategy, try finding a new group of four
        for word in remaining_words:
            if word in word_to_cluster:
                cluster_id = word_to_cluster[word]
                potential_group = [w for w in remaining_words if word_to_cluster.get(w) == cluster_id]
                if len(potential_group) == 4 and potential_group not in previousGuesses:
                    guess = potential_group
                    break

    # Fallback if no valid group of 4 found or "One Word Away" adjustments failed
    if not guess:
        guess = [word for word in random.sample(remaining_words, 4) if guess not in previousGuesses]

    # Ensure the guess is exactly four words
    guess = guess[:4]

    print("Final guess being returned:", guess)

    # Check if the turn should end due to lives or successful group completion
    if strikes >= 3 or len(correct_flattened) + len(guess) == 16:
        endTurn = True

    return guess, endTurn
