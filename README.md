# Search-XLM-R

A Python project for performing search over article text using transformer-based embeddings and FAISS indexing, combined with token matching and BM25 scoring.

## Repository Structure

- **main.py**  
  Entry point of the application. Uses argparse to pass parameters (model paths, articles directory, query, etc.).

- **model.py**  
  Contains the `EmbeddingModel` class for loading the tokenizer and model, and generating embeddings.

- **faiss_indexer.py**  
  Wraps FAISS index creation, search, save, and load operations.

- **retrieval.py**  
  Provides search functions combining embedding similarity, token matching, and BM25 scores.

- **utils.py**  
  Helper functions for chunking text, loading articles, and pickle utilities.

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- FAISS
- Rank-BM25
- Other standard libraries (os, numpy, argparse, pickle, textwrap)

## Usage

Run the main script with required arguments. For example:

```bash
python main.py \
  --tokenizer_path "/path/to/tokenizer" \
  --checkpoint_path "/path/to/checkpoint" \
  --articles_dir "/path/to/articles" \
  --query "Your search query" \
  --top_k 5
```

If no query is provided, the program will exit.

## Notes

- The embeddings are normalized before being added to the FAISS index.
- Several search functions are provided, including basic FAISS search, combined search with token matching, and BM25 search.
- The repository is structured for ease of testing and extension.
