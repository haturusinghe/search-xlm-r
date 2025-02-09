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
