import pickle

# Load and inspect clusters
def inspect_clusters():
    with open("word_clusters.pkl", "rb") as f:
        word_to_cluster = pickle.load(f)
    
    # Print the total number of clustered words and some samples
    print(f"Total words in clusters: {len(word_to_cluster)}")
    print("Sample of clustered words and their cluster IDs:")
    for word, cluster_id in list(word_to_cluster.items())[:10]:  # Show a sample of 10 words
        print(f"{word}: Cluster {cluster_id}")

if __name__ == "__main__":
    # Inspect clusters
    inspect_clusters()
