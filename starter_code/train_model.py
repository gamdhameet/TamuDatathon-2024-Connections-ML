import pandas as pd
from gensim.models import Word2Vec
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

def train_word2vec(sentences, vector_size=50, window=7, epochs=20):
    model_wv = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=1, workers=4, epochs=epochs)
    model_wv.save("word2vec_nyt_connections.model")
    print("Word2Vec model trained and saved.")
    return model_wv

def create_clusters(model, min_clusters=50, max_clusters=200):
    words = []
    word_vectors = []
    for word in model.wv.key_to_index:
        try:
            vector = model.wv.get_vector(word)
            if vector is not None:
                words.append(word)
                word_vectors.append(vector)
        except KeyError:
            continue
    
    word_vectors = np.array(word_vectors)
    optimal_clusters = min_clusters
    best_score = -1

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(word_vectors)
        score = silhouette_score(word_vectors, clusters)
        print(f"Clusters: {n_clusters}, Silhouette Score: {score}")
        if score > best_score:
            best_score = score
            optimal_clusters = n_clusters

    # Final clustering with optimal clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    final_clusters = kmeans.fit_predict(word_vectors)
    word_to_cluster = {words[i]: final_clusters[i] for i in range(len(words))}

    # Save clusters
    with open("word_clusters.pkl", "wb") as f:
        pickle.dump(word_to_cluster, f)
    print(f"Word clusters created with {optimal_clusters} clusters and saved.")
    return word_to_cluster

# Main script
if __name__ == "__main__":
    df = load_data()
    sentences = preprocess_data(df)
    model_wv = train_word2vec(sentences)
    create_clusters(model_wv)
