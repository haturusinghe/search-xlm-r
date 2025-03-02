import os
import textwrap
import pickle
from tqdm import tqdm

def chunk_text(text, chunk_size=300):
    # Simple wrapping into chunks
    return textwrap.wrap(text, width=chunk_size, break_long_words=False)

def load_articles(directory, file_extension='.txt', show_sample=True, sample_length=200, max_articles=None, lines_as_articles=False):
    files = [f for f in os.listdir(directory) if f.endswith(file_extension)]
    
    # Limit the number of files if max_articles is specified
    if max_articles is not None and max_articles > 0 and not lines_as_articles:
        files = files[:max_articles]
    
    articles = []
    for f in tqdm(files, desc="Loading articles", unit="file"):
        with open(os.path.join(directory, f), 'r', encoding='utf-8') as fh:
            if lines_as_articles:
                # Treat each line as a separate article
                for line in fh:
                    line = line.strip()
                    if line:  # Skip empty lines
                        articles.append(line)
                        # Check if we've reached the max_articles limit
                        if max_articles is not None and len(articles) >= max_articles:
                            break
            else:
                # Treat the whole file as one article
                articles.append(fh.read())
        
        # Break outer loop too if we've reached max_articles
        if lines_as_articles and max_articles is not None and len(articles) >= max_articles:
            break
    
    print(f"Loaded {len(files)} {file_extension} files from {directory}")
    print(f"Loaded {len(articles)} articles.")
    
    if show_sample and articles:
        sample_text = articles[0][:sample_length] + ("..." if len(articles[0]) > sample_length else "")
        print(f"Sample from first article ({os.path.basename(files[0]) if not lines_as_articles else 'line 1'}):")
        print("-" * 40)
        print(sample_text)
        print("-" * 40)
    
    return articles

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_masked_sentences(filepath):
    """
    Load a file with masked sentences for MLM evaluation.
    Expected format: Each line contains a masked sentence with [MASK] or <mask> token and 
    the expected word separated by a double pipe (||).
    Example: "මම <mask> වෙත යනවා||ගෙදර" or "මම [MASK] වෙත යනවා||ගෙදර"
    
    Args:
        filepath: Path to the text file with masked sentences
        
    Returns:
        List of tuples (masked_sentence, expected_word)
    """
    masked_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '||' not in line:
                continue
            masked_sent, expected = line.split('||', 1)
            masked_data.append((masked_sent, expected))
    return masked_data
