"""
Utility functions for emotion classification training.
"""
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EvalPrediction


class MetricUpdater:
    """
    Metric tracker for emotion classification.
    Computes accuracy@k metrics instead of NDCG.
    """
    
    def __init__(self, ks=None, num_emotions=33):
        self.num_emotions = num_emotions
        
        if ks is None:
            ks = [1, 3, 5]
        # Filter k values to not exceed num_emotions
        self.ks = [k for k in ks if k <= num_emotions]
        if not self.ks:
            self.ks = [1]
        self.max_k = max(self.ks)
        
        self._init_metrics()

    def _init_metrics(self):
        """Initialize metric accumulators."""
        self.accuracy_metric = {k: 0. for k in self.ks}
        self.sample_count = 0

    def update(self, logits: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor):
        """
        Update metrics with a batch of predictions.
        
        Args:
            logits: Prediction scores, shape (batch_size, num_emotions)
            labels: Ground truth labels, shape (batch_size,)
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # Validate input
        if not self._check_valid_input(logits, labels):
            return

        batch_size = labels.size(0)
        num_classes = logits.size(-1)
        
        # Compute top-k predictions
        actual_max_k = min(self.max_k, num_classes)
        _, top_indices = logits.topk(actual_max_k, dim=-1)
        
        # Compute accuracy@k
        for k in self.ks:
            actual_k = min(k, num_classes)
            top_k_indices = top_indices[:, :actual_k]
            labels_expanded = labels.unsqueeze(1).expand(-1, actual_k)
            hits = (top_k_indices == labels_expanded).any(dim=1).float()
            self.accuracy_metric[k] += hits.sum().item()

        self.sample_count += batch_size

    def _check_valid_input(self, logits, labels) -> bool:
        """Validate input tensors."""
        if not logits.numel() or not labels.numel():
            return False

        if logits.size(0) != labels.size(0):
            raise ValueError(
                f"Batch dimension mismatch. logits: {logits.size(0)}, labels: {labels.size(0)}"
            )
        
        if torch.isnan(logits).any():
            raise ValueError("logits contains NaN values")

        if labels.max().item() >= logits.size(-1):
            raise ValueError(
                f"Label value {labels.max().item()} exceeds num_classes {logits.size(-1)}"
            )

        return True

    def compute(self) -> Dict[str, float]:
        """Compute and return accumulated metrics."""
        result = {}
        
        if self.sample_count == 0:
            for k in self.accuracy_metric:
                result[f"accuracy@{k}"] = 0.0
            return result
        
        for k in self.accuracy_metric:
            result[f"accuracy@{k}"] = self.accuracy_metric[k] / self.sample_count
        
        # Reset metrics for next evaluation
        self._init_metrics()
        
        return result


def get_compute_metrics(
    metric_updater: MetricUpdater, 
    num_negatives: Optional[int] = None
) -> Callable[[EvalPrediction, bool], Dict[str, float]]:
    """
    Create a compute_metrics function for the Trainer.
    
    Args:
        metric_updater: MetricUpdater instance
        num_negatives: Number of negative samples (optional)
        
    Returns:
        Callable that computes metrics from EvalPrediction
    """
    
    def compute_metrics(eval_pred: EvalPrediction, compute_result=False) -> Dict[str, float]:
        logits = eval_pred.predictions  # (B, num_items) similarity scores
        labels = eval_pred.label_ids  # (B,) or (B, seq)

        # Handle tuple predictions (shouldn't happen now but be safe)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # Convert numpy to torch if needed
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        
        # Ensure labels and logits have matching batch dimension
        # This can happen with the last batch being smaller
        batch_size = min(logits.shape[0], labels.shape[0])
        logits = logits[:batch_size]
        labels = labels[:batch_size]

        # Flatten labels if needed and filter valid samples
        if labels.dim() > 1:
            labels = labels.view(-1)
            idx = labels.ne(-100)
            labels = labels[idx]
            if logits.dim() > 2:
                logits = logits.view(-1, logits.size(-1))[:len(idx)][idx]
            else:
                logits = logits.view(-1, logits.size(-1))[:len(idx)][idx] if logits.dim() == 2 else logits[:len(idx)][idx]
        else:
            # Labels are already 1D, filter -100
            idx = labels.ne(-100)
            labels = labels[idx]
            logits = logits[:len(idx)][idx]

        if num_negatives is not None and num_negatives > 0:
            sampled_labels, sampled_logits = _negative_sampling(labels, logits, num_negatives)
            metric_updater.update(logits=sampled_logits, labels=sampled_labels)
        else:
            metric_updater.update(logits=logits, labels=labels)

        if compute_result:
            return metric_updater.compute()
        return {}

    def _negative_sampling(labels, logits, num_negatives):
        """Sample negative examples for evaluation."""
        B, num_items = logits.shape
        sampling_prob = torch.ones((B, num_items), dtype=torch.float, device=labels.device)
        sampling_prob[torch.arange(B), labels] = 0
        negative_items = torch.multinomial(sampling_prob, num_samples=num_negatives, replacement=False)
        sampled_items = torch.cat([labels.view(-1, 1), negative_items], dim=-1)
        sampled_logits = torch.gather(logits, dim=-1, index=sampled_items)
        sampled_labels = torch.zeros(B, dtype=torch.long, device=labels.device)
        return sampled_labels, sampled_logits

    return compute_metrics


def calculate_accuracy(
    logits: torch.FloatTensor,
    labels: torch.IntTensor,
    ks: list = [1, 3, 5],
) -> Dict[str, float]:
    """
    Calculate accuracy@k for emotion classification.

    Args:
        logits: Prediction scores, shape (batch_size, num_emotions)
        labels: Ground truth labels, shape (batch_size,)
        ks: List of k values for top-k accuracy

    Returns:
        Dictionary of accuracy@k values
    """
    batch_size = logits.size(0)
    num_classes = logits.size(-1)
    
    results = {}
    
    for k in ks:
        actual_k = min(k, num_classes)
        _, top_indices = logits.topk(actual_k, dim=-1)
        labels_expanded = labels.unsqueeze(1).expand(-1, actual_k)
        hits = (top_indices == labels_expanded).any(dim=1).float()
        results[f'accuracy@{k}'] = hits.mean().item()
    
    return results


class Similarity:
    """
    Similarity function for embedding comparison.
    Supports dot product, cosine, and L2 distance.
    Returns softmax similarity for reward computation.
    """

    def __init__(self, config):
        similarity_type = config.similarity_type
        if similarity_type == "cosine":
            self.forward_func = self.forward_cos
        elif similarity_type == "dot":
            self.forward_func = self.forward_dot
        elif similarity_type == "L2":
            self.forward_func = self.forward_l2
        else:
            raise NotImplementedError(f"Similarity type {similarity_type} not implemented")
        
        self.similarity_type = similarity_type
        self.temp = config.similarity_temperature
        self.do_normalize = config.similarity_normalization

    def forward_dot(self, x, y):
        """Dot product similarity."""
        return torch.matmul(x, y.t())

    def forward_cos(self, x, y):
        """Cosine similarity."""
        return nn.CosineSimilarity(dim=-1)(x.unsqueeze(1), y.unsqueeze(0))

    def forward_l2(self, x, y):
        """Negative L2 distance (so higher is more similar)."""
        return -torch.norm(x.unsqueeze(1) - y.unsqueeze(0), p=2, dim=-1)

    def forward(self, user_hs, item_hs, seq_labels=None, gather_negs_across_processes=False):
        """
        Compute similarity between user (reasoning) and item (emotion) embeddings.
        
        Args:
            user_hs: User/reasoning embeddings, shape (batch_size, hidden_dim)
            item_hs: Item/emotion embeddings, shape (num_items, hidden_dim)
            seq_labels: Ground truth labels for computing loss (optional)
            gather_negs_across_processes: Whether to gather negatives across processes
            
        Returns:
            Dictionary with sim_matrix and softmax_sim
        """
        x = user_hs
        y = item_hs
        
        # Ensure same dtype
        if x.dtype != y.dtype:
            x = x.to(y.dtype)
        
        # Normalize if configured
        if self.do_normalize:
            x = nn.functional.normalize(x, p=2, dim=-1)
            y = nn.functional.normalize(y, p=2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = self.forward_func(x, y) / self.temp
        
        assert sim_matrix.shape == (x.shape[0], y.shape[0])
        
        # Compute softmax similarity (probability distribution over items/emotions)
        softmax_sim = torch.softmax(sim_matrix, dim=-1)
        
        return {
            'sim_matrix': sim_matrix,
            'softmax_sim': softmax_sim,
        }

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def calculate_metrics(sim_matrix: torch.Tensor, labels: torch.Tensor, k: int = 1) -> Dict[str, float]:
    """
    Calculate ranking metrics from similarity matrix.
    
    Args:
        sim_matrix: Similarity scores, shape (batch_size, num_items)
        labels: Ground truth labels, shape (batch_size,)
        k: Top-k for accuracy calculation
        
    Returns:
        Dictionary with accuracy@k
    """
    batch_size = sim_matrix.size(0)
    num_items = sim_matrix.size(1)
    
    # Clamp k to not exceed num_items
    actual_k = min(k, num_items)
    
    _, top_indices = sim_matrix.topk(actual_k, dim=-1)
    labels_expanded = labels.unsqueeze(1).expand(-1, actual_k)
    hits = (top_indices == labels_expanded).any(dim=1).float()
    
    return {
        f'accuracy@{k}': hits.mean().item(),
    }


def get_tokenizer(model_name):
    """
    Get tokenizer for a model with proper configuration.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    if "qwen" in str(tokenizer.__class__).lower() or "qwen" in model_name.lower():
        tokenizer.__setattr__('generation_prompt', "<|im_start|>assistant\n")
        tokenizer.__setattr__('generation_end', "<|im_end|>")

    if "gemma" in str(tokenizer.__class__).lower() or "gemma" in model_name.lower():
        tokenizer.__setattr__('generation_prompt', "<start_of_turn>model\n")
        tokenizer.__setattr__('generation_end', "<end_of_turn>")

    # Decoder-only models need left padding for correct generation/embeddings.
    tokenizer.padding_side = "left"

    return tokenizer
