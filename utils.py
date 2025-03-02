import os
import textwrap
import pickle

def chunk_text(text, chunk_size=300):
    # Simple wrapping into chunks
    return textwrap.wrap(text, width=chunk_size, break_long_words=False)

def load_articles(directory, file_extension='.txt'):
    files = [f for f in os.listdir(directory) if f.endswith(file_extension)]
    articles = []
    for f in files:
        with open(os.path.join(directory, f), 'r', encoding='utf-8') as fh:
            articles.append(fh.read())
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
