import numpy as np
from gensim.models import Word2Vec
import pickle

def load_model_and_clusters():
    print("Loading Word2Vec model...")
    model_wv = Word2Vec.load("word2vec_nyt_connections.model")
    print("Model loaded successfully.")
    
    print("Loading word clusters...")
    with open("word_clusters.pkl", "rb") as f:
        word_to_cluster = pickle.load(f)
    print("Clusters loaded successfully.")
    
    return model_wv, word_to_cluster

def test_model_functionality(words, model_wv, word_to_cluster):
    groups = {}
    
    print("Grouping words based on clusters...")
    for word in words:
        if word in word_to_cluster:
            cluster_id = word_to_cluster[word]
            if cluster_id not in groups:
                groups[cluster_id] = []
            groups[cluster_id].append(word)
        else:
            print(f"Warning: '{word}' not found in clusters.")  # Debugging line for missing words

    if groups:
        print("Generated groups:")
        for cluster_id, word_group in groups.items():
            print(f"Cluster {cluster_id}: {word_group}")
    else:
        print("No groups were formed. Check if word clusters are loaded correctly.")

# Main test
if __name__ == "__main__":
    model_wv, word_to_cluster = load_model_and_clusters()
    
    # Sample list of words present in word_to_cluster
    test_words = ["BALL", "RING", "HEART", "LEAD", "FLY", "WING", "FIRE", "LOVE", "JACK", "CHECK", "SUIT", "SCHOOL", "BANK", "PEN", "PAPER", "CUP", "BOTTLE", "BAG", "SHOE", "HAT", "GLASS", "PLATE", "BOWL", "SPOON", "FORK", "KNIFE", "TABLE", "CHAIR", "BED", "SOFA", "TV", "LAMP", "CLOCK", "PHONE", "COMPUTER", "KEYBOARD", "MOUSE", "SCREEN", "WINDOW", "DOOR", "FLOOR", "WALL", "CEILING", "ROOF", "STAIRS", "ELEVATOR", "TOILET", "SINK", "SHOWER", "BATHTUB", "TOWEL", "SOAP", "SHAMPOO", "TOOTHBRUSH", "TOOTHPASTE", "MIRROR", "RAZOR", "SCISSORS", "TWEZZERS", "NAIL", "HAIR", "SKIN", "BONE", "BLOOD", "HEART", "LUNG", "LIVER", "KIDNEY", "STOMACH", "INTESTINE", "BRAIN", "NERVE", "MUSCLE", "BONE", "SKIN", "HAIR", "NAIL", "BLOOD", "HEART", "LUNG", "LIVER", "KIDNEY", "STOMACH", "INTESTINE", "BRAIN", "NERVE", "MUSCLE", "BONE", "SKIN", "HAIR", "NAIL", "BLOOD", "HEART", "LUNG", "LIVER", "KIDNEY", "STOMACH", "INTESTINE", "BRAIN", "NERVE", "MUSCLE", "BONE", "SKIN", "HAIR", "NAIL", "BLOOD", "HEART", "LUNG", "LIVER", "KIDNEY", "STOMACH", "INTESTINE", "BRAIN", "NERVE", "MUSCLE", "BONE", "SKIN", "HAIR", "NAIL", "BLOOD", "HEART", "LUNG", "LIVER", "KIDNEY", "STOMACH", "INTESTINE", "BRAIN", "NERVE", "MUSCLE",]
    
    # Run the test
    test_model_functionality(test_words, model_wv, word_to_cluster)
