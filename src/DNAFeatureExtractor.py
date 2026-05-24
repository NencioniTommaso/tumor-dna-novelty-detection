import logging
from contextlib import contextmanager

import torch
import numpy as np
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from torch.utils.data import DataLoader, Dataset
import os

class _SequenceDataset(Dataset):
    def __init__(self, sequences: List[str]):
        self.sequences = sequences
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx]

class TokenizerCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return self.tokenizer(
            batch, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=1500
        )


logger = logging.getLogger(__name__)


@contextmanager
def _patched_torch_arange():
    # Temporary patch to ensure torch.arange respects the active device.
    original_arange = torch.arange

    def patched_arange(*args, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = torch.empty(0).device
        return original_arange(*args, **kwargs)

    torch.arange = patched_arange
    try:
        yield
    finally:
        torch.arange = original_arange

class DNAFoundationExtractor:
    """
    Extracts deep learning embeddings from DNA sequences using state-of-the-art 
    genomic foundation models. Defaults to DNABERT-2, but supports models like Caduceus.
    """
    def __init__(self, model_name: str = "quietflamingo/dnabert2-no-flashattention", device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"[Initialization] Loading Foundation Model: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = self.tokenizer.pad_token_id

        # --- THE ULTIMATE HOTFIX ---
        # DNABERT-2 uses torch.arange, which builds on the CPU and ignores the meta device.
        # Temporarily patch it to respect the current memory context.
        with _patched_torch_arange():
            self.model = AutoModel.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True
            ).to(self.device)
            
        self.model.eval()

    def _mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Calculates the mean token embedding."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def extract_embeddings(self, sequences: List[str], batch_size: int = 32) -> np.ndarray:
        """Iterates through the dataset using a DataLoader to extract latent representations."""
        all_embeddings = []
        
        # Prevent tokenizers from deadlocking in subprocesses
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        collate_fn = TokenizerCollate(self.tokenizer)
            
        num_workers = min(2, (os.cpu_count() or 2) - 1)
        num_workers = max(0, num_workers)
        
        dataset = _SequenceDataset(sequences)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=(self.device.type == 'cuda')
        )
        
        for inputs in dataloader:
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif 'hidden_states' in outputs:
                hidden_states = outputs['hidden_states'][-1]
            else:
                hidden_states = outputs[0]
            
            pooled_batch = self._mean_pooling(hidden_states, attention_mask)
            all_embeddings.append(pooled_batch.cpu().numpy())
            
        return np.vstack(all_embeddings)


def compute_train_test_kernels(
    train_sequences: List[str], 
    test_sequences: List[str],
    model_name: str = "quietflamingo/dnabert2-no-flashattention", 
    kernel_type: str = "rbf", 
    gamma: Optional[float] = None,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes rigorous, leak-proof Gram matrices for train and test sets.
    Returns (K_train, K_test).
    """
    if not torch.cuda.is_available():
        max_threads = os.cpu_count() or 4
        torch.set_num_threads(max_threads)
        logger.info(f"[Optimization] Configured PyTorch to use {max_threads} CPU threads.")
        
    logger.info(f"\n[Feature Extraction] Initializing {model_name}...")
    extractor = DNAFoundationExtractor(model_name=model_name)
    
    logger.info(f"[Feature Extraction] Extracting embeddings for {len(train_sequences)} Training sequences...")
    X_train = extractor.extract_embeddings(train_sequences, batch_size=batch_size)
    
    logger.info(f"[Feature Extraction] Extracting embeddings for {len(test_sequences)} Testing sequences...")
    X_test = extractor.extract_embeddings(test_sequences, batch_size=batch_size)
    
    logger.info(f"[Gram Matrix] Computing {kernel_type.upper()} kernels...")
    if kernel_type == "linear":
        K_train = linear_kernel(X_train, X_train)
        K_test = linear_kernel(X_test, X_train)
        
    elif kernel_type == "rbf":
        # Strictly calculate gamma based ONLY on the training distribution
        if gamma is None:
            gamma = 1.0 / (X_train.shape[1] * X_train.var())
            logger.info(f"[Gram Matrix] Auto-computed gamma from training variance: {gamma:.6e}")
            
        K_train = rbf_kernel(X_train, X_train, gamma=gamma)
        K_test = rbf_kernel(X_test, X_train, gamma=gamma)
        
    else:
        raise ValueError("Unsupported kernel_type. Use 'linear' or 'rbf'.")
        
    return K_train, K_test