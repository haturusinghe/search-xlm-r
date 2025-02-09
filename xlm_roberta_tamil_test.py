
from google.colab import drive
import os
drive.mount('/content/drive')

"""## Load model and tokenizer"""

from transformers import AutoModel, AutoTokenizer
import torch

# Load the updated tokenizer
updated_tokenizer_path = "/content/drive/MyDrive/xlm-roberta-dhivehi"
tokenizer = AutoTokenizer.from_pretrained(updated_tokenizer_path)

# Load the model from the checkpoint
checkpoint_path = "/content/drive/MyDrive/xlm-roberta-dhivehi/checkpoint-160000"
model = AutoModel.from_pretrained(checkpoint_path)

# Resize the model embeddings to match the updated tokenizer
model.resize_token_embeddings(len(tokenizer))
print("Model embeddings resized to match the updated tokenizer vocabulary.")

# Set the model to evaluation mode
model.eval()

"""## Get embedding"""

def get_embedding(sentence):
    """
    Generate sentence embeddings using CLS token instead of mean pooling.
    """
    # Tokenize the input
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the last hidden state
    last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

    # Use CLS token representation instead of mean pooling
    sentence_embedding = last_hidden_state[:, 0, :]  # CLS token representation

    return sentence_embedding

"""## Compare embeddings"""

from torch.nn.functional import cosine_similarity

def compare_embeddings(sentence_emb_1, sentence_emb_2):
  """
  take 2 sentence embeddings and compare them using cosine similarity
  """
  similarity = cosine_similarity(sentence_emb_1, sentence_emb_2, dim=1)
  return similarity.item()

"""## Example usage
Note: The similarity values can be used as is. However, if you're manually looking at the similarity values, look at it starting from the 3rd decimal place. For example if the similarity value is 0.9992804... then imagine the similarity is 0.92804... (ignoring the first 2 decimal places as it is always 99)
"""

sentence_1 = "ހަނިމާދޫ އެއަރޕޯޓުގެ އައު ރަންވޭގެ ބައެއް އަންނަ މަހު ބޭނުންކުރަން ފަށަނީ,30 ޑިސެމްބަރ 2023"
sentence_2 = "އެފްއޭއެމްގެ އިންތިޚާބެއް އޮންނާނީ ކޮން ދުވަހަކު؟"

sentence_embedding_1 = get_embedding(sentence_1)
sentence_embedding_2 = get_embedding(sentence_2)

similarity = compare_embeddings(sentence_embedding_1, sentence_embedding_2)
print("Cosine Similarity:", similarity)


import faiss
import numpy as np

import os

# Define the directory where your .txt files are stored
articles_directory = "/content/drive/MyDrive/DhivehiTxtfiles"  # Change this to your actual path

# List all .txt files in the directory
article_files = [f for f in os.listdir(articles_directory) if f.endswith('.txt')]

# Read the content of each file and store it in a list
articles = []
for file in article_files:
    file_path = os.path.join(articles_directory, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        articles.append(f.read())

# Display the number of articles loaded
print(f"Loaded {len(articles)} articles.")

# Optional: If you want to print the content of the first article
print("First article preview:\n", articles[0][:500])  # Display first 500 chars

import os
import textwrap

# Define the directory where your .txt files are stored
articles_directory = "/content/drive/MyDrive/DhivehiTxtfiles"  # Change to your actual path

# List all .txt files in the directory
article_files = [f for f in os.listdir(articles_directory) if f.endswith('.txt')]

# Read the content of each file and store it in a list
articles = []
for file in article_files:
    file_path = os.path.join(articles_directory, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        articles.append(f.read())

# Function to chunk articles into smaller segments
def chunk_text(text, chunk_size=300):
    return textwrap.wrap(text, width=chunk_size, break_long_words=False)

# Process all articles into chunks
article_chunks = []
chunk_to_article_map = {}  # To keep track of which chunk belongs to which article

for idx, article in enumerate(articles):
    chunks = chunk_text(article, chunk_size=300)
    for chunk in chunks:
        article_chunks.append(chunk)
        chunk_to_article_map[len(article_chunks) - 1] = idx  # Store article index

# Display results
print(f"Total chunks created: {len(article_chunks)}")
print(f"Sample chunk:\n{article_chunks[0]}")

# Normalize embeddings before adding to FAISS
chunk_embeddings = [get_embedding(chunk).squeeze().numpy() for chunk in article_chunks]
chunk_embeddings = np.array(chunk_embeddings)
chunk_embeddings /= np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)  # Normalize embeddings

# FAISS Index using Inner Product (which approximates Cosine Similarity)
embedding_dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)  # IP = Inner Product
index.add(chunk_embeddings)  # Add embeddings to FAISS

print(f"Stored {len(chunk_embeddings)} chunk embeddings in FAISS index.")

import pickle

# Save the FAISS index
faiss_index_path = "/content/drive/MyDrive/faiss_index.idx"
faiss.write_index(index, faiss_index_path)
print(f"FAISS index saved to {faiss_index_path}")

# Save the chunk-to-article mapping
chunk_map_path = "/content/drive/MyDrive/chunk_to_article_map.pkl"
with open(chunk_map_path, "wb") as f:
    pickle.dump(chunk_to_article_map, f)
print(f"Chunk-to-article map saved to {chunk_map_path}")

# Load the FAISS index
index = faiss.read_index(faiss_index_path)
print("FAISS index loaded.")

# Load the chunk-to-article mapping
with open(chunk_map_path, "rb") as f:
    chunk_to_article_map = pickle.load(f)
print("Chunk-to-article mapping loaded.")

"""# This is the test to check the retrieval on 10 documents"""

def search_similar_articles(query_text, top_k=5):
    # Convert query to embedding
    query_embedding = get_embedding(query_text).squeeze().numpy()

    # Normalize query embedding (important for FAISS with Inner Product)
    query_embedding /= np.linalg.norm(query_embedding)
    query_embedding = np.array([query_embedding])  # FAISS requires array input

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)  # Retrieve top_k results

    # Display results
    print(f"\nTop {top_k} similar chunks for query: {query_text}\n")
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
        article_index = chunk_to_article_map[idx]  # Get original article index
        print(f"Rank {rank}: Article {article_index}, Distance: {distance}\nChunk:\n{article_chunks[idx]}\n")

# Example Query
query = "އައިޖީއެމްއެޗްގެ ލެބޯޓްރީގެ އައު ގަޑިތަކަކީ ކޮބާ؟"
search_similar_articles(query, top_k=3)

"""# This is the test to combine embedding search similarity and keyword similarity"""

import numpy as np

def token_match_score(query, text):
    """
    Compute a simple token overlap score between the query and text.
    Returns the ratio of query tokens that appear in the text.
    """
    query_tokens = set(query.lower().split())
    text_tokens = set(text.lower().split())
    if not query_tokens:
        return 0.0
    return len(query_tokens.intersection(text_tokens)) / len(query_tokens)

def combined_search(query_text, top_k=5, candidate_multiplier=5, token_threshold=0.1, lambda_weight=0.5):
    """
    Combine embedding similarity (from FAISS) with a token-based string matching score.

    If not enough candidates pass the token threshold, the function falls back to the top candidates
    based solely on embedding similarity.

    Args:
      query_text (str): The input query string.
      top_k (int): Number of top results to return.
      candidate_multiplier (int): Factor to expand the candidate set from FAISS.
      token_threshold (float): Minimum fraction of query tokens that must appear in a candidate chunk.
      lambda_weight (float): Weight applied to the token match score when computing the combined score.
    """
    # Get the query embedding and normalize it.
    query_embedding = get_embedding(query_text).squeeze().numpy()
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        print("Query embedding has zero norm.")
        return
    query_embedding /= norm
    query_embedding = query_embedding.reshape(1, -1)  # FAISS requires a 2D array

    # Retrieve a larger set of candidate chunks from FAISS.
    candidate_k = top_k * candidate_multiplier
    distances, indices = index.search(query_embedding, candidate_k)

    results = []
    for idx, emb_similarity in zip(indices[0], distances[0]):
        candidate_chunk = article_chunks[idx]
        token_score = token_match_score(query_text, candidate_chunk)
        # Compute a combined score using a weighted sum.
        combined_score = emb_similarity + lambda_weight * token_score
        results.append((idx, emb_similarity, token_score, combined_score))

    # Sort candidates by the combined score (descending order)
    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)

    # Filter candidates based on the token threshold.
    results_filtered = [r for r in results_sorted if r[2] >= token_threshold]

    # If not enough candidates pass the token threshold, use the top candidates based on embedding similarity alone.
    if len(results_filtered) < top_k:
        print("Not enough candidates met the token match threshold. Falling back to embedding similarity ranking.\n")
        results_filtered = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    else:
        results_filtered = results_filtered[:top_k]

    # Display the results.
    print(f"\nTop {top_k} combined search results for query:\n'{query_text}'\n")
    for rank, (idx, emb_sim, token_score, comb_score) in enumerate(results_filtered, start=1):
        article_index = chunk_to_article_map[idx]  # Retrieve the original article index
        print(f"Rank {rank}: Article {article_index}")
        print(f"  Embedding Similarity: {emb_sim:.4f}")
        print(f"  Token Match Score:   {token_score:.4f}")
        print(f"  Combined Score:      {comb_score:.4f}")
        print("  Chunk:\n", article_chunks[idx], "\n")

# Example usage:
query_example = "ފަޒްނާ އަހުމަދުގެ މަންމާފުޅުގެ ނަމަކީ ކޮބާ؟"
combined_search(query_example, top_k=3, candidate_multiplier=5, token_threshold=0.1, lambda_weight=0.5)

"""---

# Test on the 1 million scraped dhivehi articles
"""

import textwrap

# Load text file
file_path = "/content/drive/MyDrive/CleanDhivehiNews.txt"  # Update this path if needed
with open(file_path, 'r', encoding='utf-8') as file:
    # Assuming each line is an article; strip removes any extra whitespace
    articles = [line.strip() for line in file if line.strip()]

print(f"Loaded {len(articles)} articles.")

# Function to chunk articles
def chunk_text(text, chunk_size=300):
    return textwrap.wrap(text, width=chunk_size, break_long_words=False)

# Process all articles into chunks
article_chunks = []
chunk_to_article_map = {}  # To keep track of which chunk belongs to which article

for idx, article in enumerate(articles):
    chunks = chunk_text(article, chunk_size=300)
    for chunk in chunks:
        article_chunks.append(chunk)
        # Map the chunk index (in article_chunks) back to the article index (in articles)
        chunk_to_article_map[len(article_chunks) - 1] = idx

print(f"Total chunks created: {len(article_chunks)}")

import pickle

# Save the list of article chunks.
chunks_save_path = "/content/drive/MyDrive/article_chunks2.pkl"
with open(chunks_save_path, "wb") as f:
    pickle.dump(article_chunks, f)

# Save the chunk-to-article mapping.
map_save_path = "/content/drive/MyDrive/chunk_to_article_map2.pkl"
with open(map_save_path, "wb") as f:
    pickle.dump(chunk_to_article_map, f)

print(f"Saved article chunks to {chunks_save_path}")
print(f"Saved chunk-to-article map to {map_save_path}")

import pickle

# Load the list of article chunks.
chunks_load_path = "/content/drive/MyDrive/article_chunks2.pkl"
with open(chunks_load_path, "rb") as f:
    article_chunks = pickle.load(f)

# Load the chunk-to-article mapping.
map_load_path = "/content/drive/MyDrive/chunk_to_article_map2.pkl"
with open(map_load_path, "rb") as f:
    chunk_to_article_map = pickle.load(f)

print(f"Loaded {len(article_chunks)} article chunks.")
print(f"Loaded chunk-to-article map with {len(chunk_to_article_map)} entries.")

import numpy as np
import torch
import h5py

# Ensure the model is on the proper device and in evaluation mode.
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Update or define the get_embedding function to move inputs to the same device.
def get_embedding(sentence):
    # Tokenize the sentence.
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    # Move inputs to the proper device.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Use mean pooling to obtain the sentence embedding.
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding

# Determine the embedding dimension using one sample.
sample_embedding = get_embedding(article_chunks[0]).squeeze().cpu().numpy()
embedding_dim = sample_embedding.shape[0]
num_chunks = len(article_chunks)
print(f"Number of chunks: {num_chunks}, Embedding dimension: {embedding_dim}")

# Set up parameters for batch processing.
batch_size = 32  # Adjust based on your available memory.

# Define the HDF5 file path where the embeddings will be saved.
h5_file_path = "/content/drive/MyDrive/chunk_embeddings2.h5"

# Create (or overwrite) the HDF5 file and dataset.
with h5py.File(h5_file_path, 'w') as h5f:
    # Create a dataset to hold all embeddings.
    # Using float32 for storage; if you need to save space, you could also consider float16.
    dset = h5f.create_dataset("embeddings", shape=(num_chunks, embedding_dim), dtype="float32")

    # Process the article chunks in batches.
    for i in range(0, num_chunks, batch_size):
        # Get the current batch of chunks.
        batch = article_chunks[i:i + batch_size]

        # Tokenize the batch.
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Use mean pooling to obtain embeddings for the batch.
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        batch_embeddings = batch_embeddings.cpu().numpy()

        # Normalize embeddings row-wise.
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings = batch_embeddings / (norms + 1e-10)  # Prevent division by zero.

        # Save the current batch of embeddings to the HDF5 dataset.
        dset[i:i + batch_size] = batch_embeddings

        # Optionally, print progress.
        if i % (batch_size * 100) == 0:
            print(f"Processed {i} / {num_chunks} chunks")

print("Embeddings computed and saved to disk at:", h5_file_path)

import faiss
import h5py
import numpy as np

# Define the HDF5 file path where your embeddings were saved.
h5_file_path = "/content/drive/MyDrive/chunk_embeddings.h5"

# Load the embeddings from the HDF5 file.
with h5py.File(h5_file_path, 'r') as h5f:
    # Load all embeddings into memory.
    # (Ensure your system has enough memory; if not, consider using memmap or incremental loading.)
    chunk_embeddings = h5f['embeddings'][:]

print("Loaded chunk embeddings with shape:", chunk_embeddings.shape)

# Since you normalized the embeddings when saving them, you can use FAISS's inner product (IP) index
# to approximate cosine similarity.
embedding_dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(chunk_embeddings)

print(f"FAISS index built with {index.ntotal} vectors.")


# Install rank_bm25 if needed:
# !pip install rank_bm25

from rank_bm25 import BM25Okapi
import numpy as np

def combined_search_bm25(query, top_k=10, candidate_multiplier=10, lambda_weight=1.0):
    """
    Combines FAISS embedding similarity with BM25 scores computed over candidate chunk texts.

    The candidate set is first retrieved using FAISS (using candidate_multiplier to expand the pool).
    Then, BM25 scores are computed for each candidate chunk (using simple whitespace tokenization).
    BM25 scores are normalized (by dividing by the maximum BM25 score in the candidate set).
    The final combined score is computed as:

        combined_score = embedding_similarity + lambda_weight * normalized_bm25_score

    Args:
      query (str): The input query string.
      top_k (int): The number of top results to return.
      candidate_multiplier (int): Factor to expand the candidate set from FAISS.
      lambda_weight (float): Weight given to the BM25 score in the combined ranking.
    """
    # Compute the query embedding and normalize it.
    query_embedding = get_embedding(query).squeeze().cpu().numpy()
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        print("Query embedding has zero norm!")
        return
    query_embedding = query_embedding / norm
    query_embedding = query_embedding.reshape(1, -1)  # FAISS requires a 2D array

    # Retrieve a larger candidate set from FAISS.
    candidate_k = top_k * candidate_multiplier
    distances, indices = index.search(query_embedding, candidate_k)

    # Filter out any invalid indices (e.g., indices that are negative or out of range)
    valid_indices = []
    valid_embedding_scores = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(article_chunks):
            valid_indices.append(idx)
            valid_embedding_scores.append(distances[0][i])

    if len(valid_indices) == 0:
        print("No valid candidates found!")
        return

    # Extract candidate texts from the valid indices.
    candidate_texts = [article_chunks[idx] for idx in valid_indices]
    embedding_scores = np.array(valid_embedding_scores)

    # Tokenize the candidate texts using a simple whitespace split.
    tokenized_candidates = [text.lower().split() for text in candidate_texts]

    # Create a BM25 object for the candidate texts.
    bm25 = BM25Okapi(tokenized_candidates)
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    # Normalize BM25 scores (if max > 0, otherwise leave as is).
    max_bm25 = np.max(bm25_scores)
    if max_bm25 > 0:
        normalized_bm25 = bm25_scores / max_bm25
    else:
        normalized_bm25 = bm25_scores

    # Combine scores: embedding similarity plus lambda_weight times normalized BM25.
    combined_scores = embedding_scores + lambda_weight * normalized_bm25

    # Prepare results: list of tuples (index, embedding_score, normalized_bm25, combined_score)
    results = []
    for i, idx in enumerate(valid_indices):
        results.append((idx, embedding_scores[i], normalized_bm25[i], combined_scores[i]))

    # Sort the candidates by combined score (descending order)
    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
    results_final = results_sorted[:top_k]

    # Display the results.
    print(f"\nTop {top_k} combined BM25 search results for query:\n'{query}'\n")
    for rank, (idx, emb_sim, bm25_score, comb_score) in enumerate(results_final, start=1):
        article_index = chunk_to_article_map.get(idx, "N/A")
        print(f"Rank {rank}: Article {article_index}")
        print(f"  Embedding Similarity: {emb_sim:.4f}")
        print(f"  Normalized BM25 Score: {bm25_score:.4f}")
        print(f"  Combined Score:       {comb_score:.4f}")
        print("  Chunk:")
        print(article_chunks[idx])
        print("-" * 50)

# Example usage:
query_example = "ހަނިމާދޫ އިންޓަރނޭޝަނަލް އެއަރޕޯޓުގެ އައު ރަންވޭ އަންނަ މަހު ކިހާ މިންވަރެއްގެ ކުރިއަށް އެބަދޭތޯ؟"
combined_search_bm25(query_example, top_k=10, candidate_multiplier=1000, lambda_weight=1.0)

def search(query, top_k=5):
    """
    Given a query string, compute its embedding, perform a FAISS search,
    and print the top-k most similar chunks along with their associated article index.
    """
    # Compute the query embedding using your existing get_embedding function.
    query_embedding = get_embedding(query).squeeze().cpu().numpy()
    # Normalize the query embedding.
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        print("Query embedding has zero norm!")
        return
    query_embedding = query_embedding / norm
    query_embedding = query_embedding.reshape(1, -1)

    # Perform the search on the FAISS index.
    distances, indices = index.search(query_embedding, top_k)

    print(f"\nTop {top_k} results for query: '{query}'\n")
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
        # Retrieve the original article index for this chunk (if available).
        article_index = chunk_to_article_map.get(idx, "N/A")
        # Retrieve the text of the chunk.
        chunk_text = article_chunks[idx]

        print(f"Rank {rank}:")
        print(f"  Article index: {article_index}")
        print(f"  FAISS similarity (inner product): {distance:.4f}")
        print("  Chunk text:")
        print(chunk_text)
        print("-" * 50)

# Example usage:
query_text = "ހަނިމާދޫ އިންޓަރނޭޝަނަލް އެއަރޕޯޓުގެ އައު ރަންވޭ އަންނަ މަހު ކިހާ މިންވަރެއްގެ ކުރިއަށް އެބަދޭތޯ؟"
search(query_text, top_k=10)

