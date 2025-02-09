import torch
from transformers import AutoModel, AutoTokenizer

class EmbeddingModel:
    def __init__(self, tokenizer_path, checkpoint_path, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(checkpoint_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, sentence, pooling="cls"):
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        if pooling == "cls":
            embedding = outputs.last_hidden_state[:, 0, :]
        else:
            embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding
