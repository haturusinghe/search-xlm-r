import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

class EmbeddingModel:
    def __init__(self, tokenizer_path, checkpoint_path, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(checkpoint_path)
        self.mlm_model = None
        
        # Set device if provided, otherwise use CUDA if available
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def to_device(self, device):
        """Move model to specified device"""
        if self.device != device:
            self.device = device
            self.model.to(device)
            if self.mlm_model:
                self.mlm_model.to(device)
            print(f"Model moved to {device}")
    
    def get_embedding(self, text, device=None):
        """Get embeddings for a single text"""
        if device is None:
            device = self.device
            
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token as the sentence embedding
            embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding
    
    def get_batch_embeddings(self, text_list, device=None):
        """
        Get embeddings for a batch of texts efficiently
        
        Args:
            text_list: List of texts to encode
            device: Device to use for computation
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        if device is None:
            device = self.device
            
        # Tokenize all texts in the batch at once
        inputs = self.tokenizer(
            text_list, 
            return_tensors='pt', 
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move inputs to the device
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Get embeddings without gradient calculation
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token as the sentence embedding for each item in batch
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def load_mlm_model(self, checkpoint_path=None):
        """
        Load a model for masked language modeling tasks
        """
        model_path = checkpoint_path or self.model.config._name_or_path
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.mlm_model.to(self.device)
        self.mlm_model.eval()
    
    def predict_masked_token(self, masked_text, top_k=5):
        """
        Predict the most likely tokens for a [MASK] or <mask> in the input text
        
        Args:
            masked_text: Text with [MASK] or <mask> token
            top_k: Number of top predictions to return
            
        Returns:
            List of (token, probability) pairs
        """
        if self.mlm_model is None:
            self.load_mlm_model()
            
        # Replace mask tokens with the actual mask token if needed
        if "[MASK]" in masked_text:
            masked_text = masked_text.replace("[MASK]", self.tokenizer.mask_token)
        elif "<mask>" in masked_text:
            masked_text = masked_text.replace("<mask>", self.tokenizer.mask_token)
            
        inputs = self.tokenizer(masked_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.mlm_model(**inputs)
        
        # Find position of mask token
        mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
        if len(mask_token_index) == 0:
            return []
            
        mask_token_index = mask_token_index[0].item()
        
        # Get predictions
        logits = outputs.logits[0, mask_token_index, :]
        probs = torch.nn.functional.softmax(logits, dim=0)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
        
        # Convert to tokens
        predictions = []
        for i, idx in enumerate(top_k_indices):
            token = self.tokenizer.convert_ids_to_tokens([idx])[0]
            probability = top_k_weights[i].item()
            predictions.append((token, probability))
            
        return predictions
