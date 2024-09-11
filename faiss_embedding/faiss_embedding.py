import faiss
import gensim.downloader as api
import numpy as np
import logging
import sys

# Set Gensim logging to show only errors and critical messages
logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('smart_open').setLevel(logging.ERROR)

# Redirect stdout and stderr to suppress any potential output
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')


# Pre-trained Word2Vec model from Gensim
word2vec_model = api.load('word2vec-google-news-300')  # Word2Vec with 300 dimensions, trained on Google News

# List of words to embed
words = ['king', 'queen', 'man', 'woman', 'infant', 'child', 'boy', 'girl', 
         'house', 'mansion', 'hotel', 'park', 'movie', 'artist', 'singer']

# Get the vector for each word using the pre-trained Word2Vec model
word_vectors = np.array([word2vec_model[word] for word in words if word in word2vec_model])

# Number of dimensions of the word embeddings (should be 300 for Word2Vec)
d = word_vectors.shape[1]

# Initialize a FAISS index for L2 (Euclidean) distance search
index = faiss.IndexFlatL2(d)

# Add word vectors to the FAISS index
index.add(word_vectors)

# Restore stdout and stderr (optional if you need them later)
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Search the nearest neighbors for a query word
def find_similar(word, top_n=5):
    if word in word2vec_model:
        query_vector = np.array([word2vec_model[word]])
        distances, indices = index.search(query_vector, top_n)
        print(f"Top {top_n} words similar to '{word}':")
        for i, idx in enumerate(indices[0]):
            print(f"{i+1}: {words[idx]} (distance: {distances[0][i]})")
    else:
        print(f"'{word}' not in the vocabulary")

# Example usage: Find top 5 similar words to 'king'
find_similar('king', top_n=5)

# Example usage: Find top 5 similar words to 'hotel'
find_similar('hotel', top_n=5)