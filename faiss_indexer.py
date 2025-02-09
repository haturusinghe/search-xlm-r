import faiss
import numpy as np

class FaissIndexer:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add_embeddings(self, embeddings):
        # embeddings should be a numpy array with rows normalized
        self.index.add(embeddings)

    def save_index(self, path):
        faiss.write_index(self.index, path)

    def load_index(self, path):
        self.index = faiss.read_index(path)

    def search(self, query_embedding, top_k):
        # query_embedding expected as a 2D numpy array
        return self.index.search(query_embedding, top_k)
