import pandas as pd
from gensim.models import Word2Vec
from datasets import load_dataset
from sklearn.cluster import KMeans
import numpy as np
import pickle

def load_data():
    dataset = load_dataset("eric27n/NYT-Connections", split="train")
    data = [item for item in dataset]
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    grouped = df.groupby('Group Name')['Word'].apply(list)
    sentences = grouped.tolist()
    return sentences

def train_word2vec(sentences):
    model_wv = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    model_wv.save("word2vec_nyt_connections.model")
    print("Word2Vec model trained and saved.")
    return model_wv

def create_clusters(model, num_clusters=100):
    # Initialize lists for words and vectors
    words = []
    word_vectors = []
    
    # Collect words and their vectors from the model
    for word in model.wv.key_to_index:  # Safely iterate over model's vocabulary
        try:
            vector = model.wv.get_vector(word)  # Safely retrieve vector
            if vector is not None:
                words.append(word)
                word_vectors.append(vector)
        except KeyError:
            print(f"Word '{word}' not found in the model vocabulary and is skipped.")
    
    # Ensure word_vectors is a numpy array for K-Means
    word_vectors = np.array(word_vectors)
    
    # Apply K-Means clustering on the valid word vectors
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(word_vectors)
    
    # Map each word to its cluster
    word_to_cluster = {words[i]: clusters[i] for i in range(len(words))}
    
    # Save clusters using pickle
    with open("word_clusters.pkl", "wb") as f:
        pickle.dump(word_to_cluster, f)
    print("Word clusters created and saved.")
    return word_to_cluster


# Main script
if __name__ == "__main__":
    df = load_data()
    sentences = preprocess_data(df)
    model_wv = train_word2vec(sentences)
    create_clusters(model_wv)
