import torch
import numpy as np
import h5py
from tqdm import tqdm

def compute_embeddings_batch(model, chunks, batch_size=2048, device='cuda', show_progress=True):
    """
    Compute embeddings for text chunks in optimized batches using GPU acceleration.
    
    Args:
        model: The embedding model
        chunks: List of text chunks
        batch_size: Number of chunks to process at once
        device: Device to use ('cuda' or 'cpu')
        show_progress: Whether to show a progress bar
        
    Returns:
        numpy array of normalized embeddings
    """
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    # Move model to the device
    model.to_device(device)
    
    # Get total number of chunks
    num_chunks = len(chunks)
    
    # Determine embedding dimension with a single sample
    with torch.no_grad():
        sample_embedding = model.get_embedding(chunks[0], device=device)
        embedding_dim = sample_embedding.shape[-1]
    
    # Initialize the numpy array for all embeddings
    all_embeddings = np.zeros((num_chunks, embedding_dim), dtype=np.float32)
    
    # Process in batches with progress bar
    iterator = range(0, num_chunks, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing embeddings", unit="batch")
    
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, num_chunks)
        batch_chunks = chunks[start_idx:end_idx]
        
        try:
            # Process batch at once to maximize GPU utilization
            with torch.no_grad():  # Disable gradient calculation for inference
                # Get embeddings for the entire batch at once
                batch_embeddings = model.get_batch_embeddings(batch_chunks, device=device)
                
                # Normalize embeddings (using PyTorch for GPU acceleration)
                norms = torch.norm(batch_embeddings, dim=1, keepdim=True)
                batch_embeddings = torch.div(batch_embeddings, norms.clamp(min=1e-10))
                
                # Move to CPU and convert to numpy
                batch_embeddings_np = batch_embeddings.cpu().numpy()
                
            # Store in the pre-allocated array
            all_embeddings[start_idx:end_idx] = batch_embeddings_np
            
        except RuntimeError as e:  # Handle potential CUDA out of memory error
            if 'CUDA out of memory' in str(e):
                print(f"\nCUDA out of memory error. Trying with smaller batch...")
                # Try processing this batch with half the batch size
                half_batch_size = (end_idx - start_idx) // 2
                if half_batch_size == 0:  # If can't reduce further, process one by one
                    print("Processing individual items...")
                    for i, chunk in enumerate(batch_chunks):
                        with torch.no_grad():
                            emb = model.get_embedding(chunk, device=device).cpu().numpy()
                            norm = np.linalg.norm(emb)
                            all_embeddings[start_idx + i] = emb / norm if norm > 0 else emb
                else:
                    # Process first half
                    first_half = compute_embeddings_batch(
                        model, batch_chunks[:half_batch_size], 
                        batch_size=half_batch_size, device=device, show_progress=False
                    )
                    # Process second half
                    second_half = compute_embeddings_batch(
                        model, batch_chunks[half_batch_size:],
                        batch_size=half_batch_size, device=device, show_progress=False
                    )
                    # Combine results
                    all_embeddings[start_idx:start_idx+half_batch_size] = first_half
                    all_embeddings[start_idx+half_batch_size:end_idx] = second_half
            else:
                raise e
    
    return all_embeddings

def create_embeddings_h5(model, chunks, h5_path, batch_size=2048, device='cuda'):
    """
    Compute embeddings and save directly to H5 file with optimized GPU batch processing.
    
    Args:
        model: The embedding model
        chunks: List of text chunks
        h5_path: Path to save the H5 file
        batch_size: Number of chunks to process at once
        device: Device to use ('cuda' or 'cpu')
    """
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    # Move model to the device
    model.to_device(device)
    
    # Get total number of chunks
    num_chunks = len(chunks)
    
    # Determine embedding dimension with a single sample
    with torch.no_grad():
        sample_embedding = model.get_embedding(chunks[0], device=device)
        embedding_dim = sample_embedding.shape[-1]
    
    print(f"Creating H5 file with {num_chunks} embeddings of dimension {embedding_dim}")
    
    # Create H5 file and dataset
    with h5py.File(h5_path, 'w') as h5f:
        # Create dataset with chunks for efficient I/O
        dset = h5f.create_dataset(
            "embeddings", 
            shape=(num_chunks, embedding_dim), 
            dtype="float32",
            chunks=(min(1024, num_chunks), embedding_dim)  # Optimized chunk size for HDF5
        )
        
        # Process in batches with progress bar
        for start_idx in tqdm(range(0, num_chunks, batch_size), desc="Saving embeddings", unit="batch"):
            end_idx = min(start_idx + batch_size, num_chunks)
            batch_chunks = chunks[start_idx:end_idx]
            
            try:
                # Process batch at once to maximize GPU utilization
                with torch.no_grad():  # Disable gradient calculation for inference
                    # Get embeddings for the entire batch at once
                    batch_embeddings = model.get_batch_embeddings(batch_chunks, device=device)
                    
                    # Normalize embeddings
                    norms = torch.norm(batch_embeddings, dim=1, keepdim=True)
                    batch_embeddings = torch.div(batch_embeddings, norms.clamp(min=1e-10))
                    
                    # Move to CPU and convert to numpy
                    batch_embeddings_np = batch_embeddings.cpu().numpy()
                
                # Store in HDF5 file
                dset[start_idx:end_idx] = batch_embeddings_np
                
            except RuntimeError as e:  # Handle potential CUDA out of memory error
                if 'CUDA out of memory' in str(e):
                    print(f"\nCUDA out of memory error. Processing this batch with smaller batches...")
                    # Process this batch with adaptive batch sizing
                    current_batch_size = (end_idx - start_idx) // 2
                    while current_batch_size > 0:
                        success = False
                        for sub_start in range(start_idx, end_idx, current_batch_size):
                            sub_end = min(sub_start + current_batch_size, end_idx)
                            try:
                                with torch.no_grad():
                                    sub_batch = chunks[sub_start:sub_end]
                                    sub_embeddings = model.get_batch_embeddings(sub_batch, device=device)
                                    norms = torch.norm(sub_embeddings, dim=1, keepdim=True)
                                    sub_embeddings = torch.div(sub_embeddings, norms.clamp(min=1e-10))
                                    dset[sub_start:sub_end] = sub_embeddings.cpu().numpy()
                                success = True
                            except RuntimeError:
                                success = False
                                break
                        
                        if success:
                            break
                        current_batch_size //= 2
                    
                    # If all attempts with batching fail, process one by one
                    if current_batch_size == 0:
                        print("Processing individual items...")
                        for i in range(start_idx, end_idx):
                            with torch.no_grad():
                                emb = model.get_embedding(chunks[i], device=device).cpu().numpy()
                                norm = np.linalg.norm(emb)
                                dset[i] = emb / norm if norm > 0 else emb
                else:
                    raise e
