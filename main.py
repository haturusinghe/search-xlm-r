import os
import numpy as np
import argparse
import random
import h5py
import torch
from model import EmbeddingModel
from utils import load_articles, chunk_text, load_masked_sentences, save_pickle, load_pickle
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
    
    # --- Load Article Chunks and Mapping ---
    # Check if we should load pre-saved chunks and mapping
    article_chunks = None
    chunk_to_article_map = None
    
    # Try to load pre-saved chunks if path provided
    if args.load_chunks_path and os.path.exists(args.load_chunks_path):
        print(f"Loading pre-saved article chunks from {args.load_chunks_path}")
        article_chunks = load_pickle(args.load_chunks_path)
        print(f"Loaded {len(article_chunks)} chunks")
    
    # Try to load pre-saved mapping if path provided
    if args.load_map_path and os.path.exists(args.load_map_path):
        print(f"Loading pre-saved chunk-to-article map from {args.load_map_path}")
        chunk_to_article_map = load_pickle(args.load_map_path)
        print(f"Loaded mapping for {len(chunk_to_article_map)} chunks")
    
    # If chunks or mapping not loaded, process articles normally
    if article_chunks is None or chunk_to_article_map is None:
        if not args.articles_dir:
            parser.error("Either provide article directory or both load_chunks_path and load_map_path")
            
        print("Processing articles to create chunks...")
        articles = load_articles(args.articles_dir)
        article_chunks = []
        chunk_to_article_map = {}
        for idx, article in enumerate(articles):
            chunks = chunk_text(article, chunk_size=300)
            for chunk in chunks:
                article_chunks.append(chunk)
                chunk_to_article_map[len(article_chunks) - 1] = idx
        
        # Save chunk data if paths are provided
        if args.save_chunks_path:
            print(f"Saving article chunks to {args.save_chunks_path}")
            save_pickle(article_chunks, args.save_chunks_path)
        
        if args.save_map_path:
            print(f"Saving chunk-to-article map to {args.save_map_path}")
            save_pickle(chunk_to_article_map, args.save_map_path)
    
    # Display chunk statistics
    print(f"Total chunks created: {len(article_chunks)}")
    if len(article_chunks) > 0:
        # Display 5 random chunks (or fewer if there are less than 5 chunks)
        sample_size = min(5, len(article_chunks))
        sample_indices = random.sample(range(len(article_chunks)), sample_size)
        print(f"\nShowing {sample_size} random chunk samples:")
        for i, sample_idx in enumerate(sample_indices):
            article_idx = chunk_to_article_map[sample_idx]
            print(f"\nSample {i+1} (from article {article_idx}):")
            print("-" * 50)
            print(article_chunks[sample_idx][:150] + "..." if len(article_chunks[sample_idx]) > 150 else article_chunks[sample_idx])
            print("-" * 50)

    # --- FAISS Index Creation ---
    # Compute embeddings with batch processing and save to HDF5
    if args.h5_embeddings_path and os.path.exists(args.h5_embeddings_path) and not args.force_recompute:
        print(f"Loading pre-computed embeddings from {args.h5_embeddings_path}")
        with h5py.File(args.h5_embeddings_path, 'r') as h5f:
            chunk_embeddings = h5f['embeddings'][:]
            embedding_dim = chunk_embeddings.shape[1]
            print(f"Loaded {len(chunk_embeddings)} embeddings with dimension {embedding_dim}")
    else:
        # Determine embedding dimension using one sample
        sample_embedding = model.get_embedding(article_chunks[0]).squeeze().cpu().numpy()
        embedding_dim = sample_embedding.shape[0]
        num_chunks = len(article_chunks)
        print(f"Number of chunks: {num_chunks}, Embedding dimension: {embedding_dim}")
        
        # Set up parameters for batch processing
        batch_size = args.batch_size  # Adjustable batch size
        
        # If h5 path is provided, use it to store embeddings
        if args.h5_embeddings_path:
            print(f"Computing embeddings in batches and saving to {args.h5_embeddings_path}")
            with h5py.File(args.h5_embeddings_path, 'w') as h5f:
                # Create dataset for embeddings
                dset = h5f.create_dataset("embeddings", shape=(num_chunks, embedding_dim), dtype="float32")
                
                # Process chunks in batches
                for i in range(0, num_chunks, batch_size):
                    # Get current batch
                    batch = article_chunks[i:min(i + batch_size, num_chunks)]
                    
                    # Process each chunk in batch and get embedding
                    batch_embeddings = []
                    for chunk in batch:
                        emb = model.get_embedding(chunk).squeeze().cpu().numpy()
                        norm = np.linalg.norm(emb)
                        batch_embeddings.append(emb / norm if norm != 0 else emb)
                    
                    # Store in HDF5
                    batch_embeddings = np.array(batch_embeddings)
                    dset[i:i + len(batch)] = batch_embeddings
                    
                    # Print progress
                    if i % (batch_size * 10) == 0 or i + batch_size >= num_chunks:
                        print(f"Processed {i + len(batch)} / {num_chunks} chunks")
                
                print("Embeddings computed and saved to disk at:", args.h5_embeddings_path)
                
            # Load the embeddings back for indexing
            with h5py.File(args.h5_embeddings_path, 'r') as h5f:
                chunk_embeddings = h5f['embeddings'][:]
        else:
            # If no h5 path, compute embeddings in memory (original approach)
            print("Computing embeddings in memory")
            chunk_embeddings = []
            for i, chunk in enumerate(article_chunks):
                emb = model.get_embedding(chunk).squeeze().cpu().numpy()
                norm = np.linalg.norm(emb)
                chunk_embeddings.append(emb / norm if norm != 0 else emb)
                
                # Print progress
                if i % 100 == 0 or i + 1 == num_chunks:
                    print(f"Processed {i + 1} / {num_chunks} chunks")
                    
            chunk_embeddings = np.array(chunk_embeddings)

    # Create and populate FAISS index
    indexer = FaissIndexer(embedding_dim)
    indexer.add_embeddings(chunk_embeddings)
    print(f"Stored {len(chunk_embeddings)} chunk embeddings in FAISS index.")

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
    parser.add_argument("--save_chunks_path", type=str, default=None, help="Path to save article chunks as pickle file")
    parser.add_argument("--save_map_path", type=str, default=None, help="Path to save chunk-to-article map as pickle file")
    parser.add_argument("--load_chunks_path", type=str, default=None, help="Path to load pre-saved article chunks pickle file")
    parser.add_argument("--load_map_path", type=str, default=None, help="Path to load pre-saved chunk-to-article map pickle file")
    parser.add_argument("--h5_embeddings_path", type=str, default=None, 
                        help="Path to save/load embeddings using HDF5 format")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for processing embeddings")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Force recomputation of embeddings even if H5 file exists")
    
    args = parser.parse_args()
    
    # Update the articles_dir requirement check
    if not args.eval_mlm and not args.articles_dir and (not args.load_chunks_path or not args.load_map_path):
        parser.error("Either --articles_dir or (--load_chunks_path AND --load_map_path) or --eval_mlm must be specified")
        
    main(args)

