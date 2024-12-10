# Connections AI ML Model

This project provides a machine learning-based solution for generating optimal guesses in the **Connections AI Game**. The game challenges players to group related words into four sets of four. The model uses **Sentence-BERT** embeddings and **Agglomerative Clustering** to identify relationships between words and propose accurate guesses.

## Features

- **Semantic Understanding**: The model leverages a pre-trained **Sentence-BERT** model (`paraphrase-MiniLM-L6-v2`) to compute embeddings for each word, enabling the detection of subtle relationships and meanings.
  
- **Clustering-Based Grouping**: Using **Agglomerative Clustering** with cosine similarity as the metric, the model groups words into semantically related clusters, providing logical and data-driven guesses.

- **Adaptive Guessing**: The model incorporates game-specific parameters like:
  - Handling cases where the player is "one word away" from a correct group.
  - Tracking previously guessed groups to avoid repetition.
  - Modifying guesses dynamically based on feedback.

- **Robust Fallback Mechanism**: If clustering does not produce a valid guess, the model falls back to a randomized selection of four words, ensuring continuous gameplay.

- **Error Handling**: The system gracefully handles exceptions such as model loading errors and invalid inputs.

## Key Components

1. **Word Preprocessing**: 
   - Converts all words to uppercase for uniformity.
   - Excludes already grouped words from further consideration.

2. **Embedding Generation**: 
   - Computes word embeddings using Sentence-BERT, capturing semantic nuances.

3. **Clustering**: 
   - Groups words into four clusters based on similarity.
   - Uses cosine distance and average linkage for optimal clustering.

4. **Dynamic Guessing**: 
   - Adapts guesses based on the game's state (e.g., previous guesses, correct groups, and strikes).

5. **Game Logic Integration**: 
   - Ensures compliance with game rules, such as selecting exactly four unique words per guess.
   - Ends the turn when strikes reach the limit or all words are grouped.

## Challenges Addressed

- Identifying semantic relationships between words in diverse and ambiguous datasets.
- Balancing logical guess generation with fallback strategies for edge cases.
- Adapting machine learning techniques to a dynamic and rule-based game environment.

## Technologies Used

- **Python**: Programming language for implementation.
- **Sentence-BERT**: Pre-trained transformer model for embedding generation.
- **Scikit-learn**: For clustering and other machine learning utilities.
- **NumPy**: For efficient numerical computations.
- **Random**: For generating fallback guesses when necessary.

## Acknowledgments

- **Sentence-BERT**: Thanks to the creators of the `paraphrase-MiniLM-L6-v2` model for providing robust word embeddings.
- **Scikit-learn**: For providing the tools necessary for clustering and similarity computation.
- Game concept inspired by the **Connections AI Game** mechanics. 

--- 

This project demonstrates the application of machine learning techniques to create a fun and engaging game experience while exploring semantic relationships in natural language data.
