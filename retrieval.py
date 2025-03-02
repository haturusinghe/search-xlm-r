import numpy as np
from torch.nn.functional import cosine_similarity
from rank_bm25 import BM25Okapi

def compare_embeddings(emb1, emb2):
    sim = cosine_similarity(emb1, emb2, dim=1)
    return sim.item()

def search_similar_articles(query, model, indexer, chunk_to_article_map, article_chunks, top_k=5):
    query_embedding = model.get_embedding(query).squeeze().cpu().numpy()
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        print("Query embedding has zero norm")
        return
    query_embedding /= norm
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = indexer.search(query_embedding, top_k)
    print(f"\nTop {top_k} similar chunks for query: {query}\n")
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
        article_index = chunk_to_article_map.get(idx, "N/A")
        print(f"Rank {rank}: Article {article_index}, Distance: {distance}\nChunk:\n{article_chunks[idx]}\n")

def combined_search(query, model, indexer, chunk_to_article_map, article_chunks, top_k=5, candidate_multiplier=5, token_threshold=0.1, lambda_weight=0.5):
    query_embedding = model.get_embedding(query).squeeze().cpu().numpy()
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        print("Query embedding has zero norm")
        return
    query_embedding /= norm
    query_embedding = query_embedding.reshape(1, -1)
    candidate_k = top_k * candidate_multiplier
    distances, indices = indexer.search(query_embedding, candidate_k)
    results = []
    for idx, emb_sim in zip(indices[0], distances[0]):
        candidate_chunk = article_chunks[idx]
        # Simple token match: ratio of shared tokens
        token_score = len(set(query.lower().split()).intersection(set(candidate_chunk.lower().split()))) / max(len(query.split()), 1)
        combined_score = emb_sim + lambda_weight * token_score
        results.append((idx, emb_sim, token_score, combined_score))
    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
    results_filtered = [r for r in results_sorted if r[2] >= token_threshold]
    if len(results_filtered) < top_k:
        results_filtered = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    else:
        results_filtered = results_filtered[:top_k]
    print(f"\nTop {top_k} combined search results for query:\n'{query}'\n")
    for rank, (idx, emb_sim, token_score, comb_score) in enumerate(results_filtered, start=1):
        article_index = chunk_to_article_map.get(idx, "N/A")
        print(f"Rank {rank}: Article {article_index}")
        print(f"  Embedding Similarity: {emb_sim:.4f}")
        print(f"  Token Match Score:   {token_score:.4f}")
        print(f"  Combined Score:      {comb_score:.4f}")
        print("  Chunk:\n", article_chunks[idx], "\n")

def combined_search_bm25(query, model, indexer, chunk_to_article_map, article_chunks, top_k=10, candidate_multiplier=1000, lambda_weight=1.0):
    query_embedding = model.get_embedding(query).squeeze().cpu().numpy()
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        print("Query embedding has zero norm!")
        return
    query_embedding /= norm
    query_embedding = query_embedding.reshape(1, -1)
    candidate_k = top_k * candidate_multiplier
    distances, indices = indexer.search(query_embedding, candidate_k)
    valid_indices = []
    valid_embedding_scores = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(article_chunks):
            valid_indices.append(idx)
            valid_embedding_scores.append(distances[0][i])
    if not valid_indices:
        print("No valid candidates found!")
        return
    candidate_texts = [article_chunks[idx] for idx in valid_indices]
    embedding_scores = np.array(valid_embedding_scores)
    tokenized_candidates = [text.lower().split() for text in candidate_texts]
    bm25 = BM25Okapi(tokenized_candidates)
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    max_bm25 = np.max(bm25_scores)
    normalized_bm25 = bm25_scores / max_bm25 if max_bm25 > 0 else bm25_scores
    combined_scores = embedding_scores + lambda_weight * normalized_bm25
    results = []
    for i, idx in enumerate(valid_indices):
        results.append((idx, embedding_scores[i], normalized_bm25[i], combined_scores[i]))
    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)[:top_k]
    print(f"\nTop {top_k} combined BM25 search results for query:\n'{query}'\n")
    for rank, (idx, emb_sim, bm25_score, comb_score) in enumerate(results_sorted, start=1):
        article_index = chunk_to_article_map.get(idx, "N/A")
        print(f"Rank {rank}: Article {article_index}")
        print(f"  Embedding Similarity: {emb_sim:.4f}")
        print(f"  Normalized BM25 Score: {bm25_score:.4f}")
        print(f"  Combined Score:       {comb_score:.4f}")
        print("  Chunk:")
        print(article_chunks[idx])
        print("-" * 50)

def search(query, model, indexer, chunk_to_article_map, article_chunks, top_k=10):
    query_embedding = model.get_embedding(query).squeeze().cpu().numpy()
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        print("Query embedding has zero norm!")
        return
    query_embedding /= norm
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = indexer.search(query_embedding, top_k)
    print(f"\nTop {top_k} results for query: '{query}'\n")
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
        article_index = chunk_to_article_map.get(idx, "N/A")
        print(f"Rank {rank}:")
        print(f"  Article index: {article_index}")
        print(f"  FAISS similarity: {distance:.4f}")
        print("  Chunk:")
        print(article_chunks[idx])
        print("-" * 50)
