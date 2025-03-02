import os
import numpy as np
import argparse
from model import EmbeddingModel
from utils import load_articles, chunk_text, load_masked_sentences
from faiss_indexer import FaissIndexer
from retrieval import search_similar_articles, combined_search, combined_search_bm25, search
from evaluation import evaluate_mlm, print_evaluation_results, save_evaluation_results

def main(args):
    # --- Configuration & Model Initialization ---
    tokenizer_path = args.tokenizer_path
    checkpoint_path = args.checkpoint_path
    model = EmbeddingModel(tokenizer_path, checkpoint_path)
    
    # --- MLM Evaluation Mode ---
    if args.eval_mlm:
        print(f"Loading masked sentences from {args.eval_mlm}")
        masked_sentences = load_masked_sentences(args.eval_mlm)
        print(f"Loaded {len(masked_sentences)} masked sentences for evaluation")
        
        # Load MLM model
        model.load_mlm_model()
        
        # Run evaluation
        metrics = evaluate_mlm(model, masked_sentences, top_k=args.top_k)
        
        # Print results
        print_evaluation_results(metrics)
        
        # Save results if output directory is specified
        if args.output_dir:
            save_evaluation_results(metrics, args.output_dir)
        
        return
    
    # --- Load Articles & Chunking ---
    articles = load_articles(args.articles_dir)
    article_chunks = []
    chunk_to_article_map = {}
    for idx, article in enumerate(articles):
        chunks = chunk_text(article, chunk_size=300)
        for chunk in chunks:
            article_chunks.append(chunk)
            chunk_to_article_map[len(article_chunks) - 1] = idx

    # --- FAISS Index Creation ---
    # Compute and normalize embeddings for all chunks
    chunk_embeddings = []
    for chunk in article_chunks:
        emb = model.get_embedding(chunk).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        chunk_embeddings.append(emb / norm if norm != 0 else emb)
    chunk_embeddings = np.array(chunk_embeddings)
    embedding_dim = chunk_embeddings.shape[1]

    indexer = FaissIndexer(embedding_dim)
    indexer.add_embeddings(chunk_embeddings)

    # --- Execute Query if provided ---
    if args.query:
        query = args.query
        print("=== FAISS Similarity Search ===")
        search_similar_articles(query, model, indexer, chunk_to_article_map, article_chunks, top_k=args.top_k)
        print("=== Combined Search ===")
        combined_search(query, model, indexer, chunk_to_article_map, article_chunks, top_k=args.top_k)
        print("=== Combined BM25 Search ===")
        combined_search_bm25(query, model, indexer, chunk_to_article_map, article_chunks, top_k=args.top_k)
        print("=== Basic FAISS Search ===")
        search(query, model, indexer, chunk_to_article_map, article_chunks, top_k=args.top_k)
    else:
        print("No query provided. Exiting.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search with Embedding Model and FAISS Index")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--articles_dir", type=str, default=None, help="Directory of articles (.txt files)")
    parser.add_argument("--query", type=str, default=None, help="Query string to search")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top search results or predictions")
    parser.add_argument("--eval_mlm", type=str, default=None, help="Path to file with masked sentences for MLM evaluation")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Make articles_dir optional if we're only doing MLM evaluation
    if not args.articles_dir and not args.eval_mlm:
        parser.error("Either --articles_dir or --eval_mlm must be specified")
        
    main(args)
