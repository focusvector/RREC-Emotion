"""
RecPOTrainer - RL-based trainer for recommendation with policy optimization.
Uses GRPO-style policy gradient with similarity-based rewards.
Adapted from R2EC (https://github.com/YRYangang/RRec)
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List

import torch
from torch import nn
import torch.nn.functional as F

from transformers.utils import logging
from transformers import GenerationConfig

from trainers.GRecTrainer import GenRecTrainer, GenRecTrainingArguments

logger = logging.get_logger(__name__)


@dataclass
class RecPOTrainingArguments(GenRecTrainingArguments):
    epsilon_low: float = field(
        default=0.2,
        metadata={"help": "Lower bound for PPO clipping."},
    )
    epsilon_high: float = field(
        default=0.28,
        metadata={"help": "Upper bound for PPO clipping."},
    )
    reward_softmax_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for softmax reward component (embedding similarity)."},
    )
    reward_format_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for format reward component (exact emotion match)."},
    )
    advantage_type: str = field(
        default="gaussian",
        metadata={"help": "Type of advantage normalization: 'gaussian' or 'loo' (leave-one-out)."},
    )
    item_emb_refresh_steps: int = field(
        default=0,
        metadata={"help": "Recompute item embeddings every N steps during training (0 = only once)."},
    )


class RecPOTrainer(GenRecTrainer):
    """
    RL trainer with GRPO-style policy gradient.
    
    Training loop:
    1. Generate K reasoning samples per input using vLLM
    2. Compute rewards based on:
       - Embedding similarity between generated reasoning and target emotion
    3. Compute advantages (gaussian or leave-one-out normalization)
    4. Apply PPO-style clipped policy gradient update
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args: RecPOTrainingArguments

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, any]],
                      num_items_in_batch: int = None) -> torch.Tensor:
        """
        Full RL training step:
        1. Generate K samples per input using vLLM
        2. Compute rewards and advantages
        3. Compute PPO-style clipped policy gradient loss
        """
        model.train()

        # Step 1: Generate K reasoning samples per input
        augmented_input = self._generate_in_train(model, inputs)

        # Step 2: Compute rewards and advantages
        augmented_input = self.compute_rec_score(model, augmented_input)

        # Step 3: Train with mini-batches using policy gradient
        mini_batch_size = self.args.mini_batch_size
        group_size = self.args.generation_config.num_return_sequences
        batch_size = augmented_input['seq_labels'].shape[0]

        losses = []
        for i in range(0, batch_size, mini_batch_size):
            mini_batch = {k: v[i * group_size:(i + mini_batch_size) * group_size]
                         if isinstance(v, torch.Tensor) and v.shape[0] == batch_size * group_size
                         else v[i:i + mini_batch_size]
                         for k, v in augmented_input.items()}
            loss = self.batch_forward(model, mini_batch)
            losses.append(loss)

        total_loss = sum(losses) / len(losses)
        return total_loss

    def compute_rec_score(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute rewards and advantages for each generated sample.
        
        Reward uses embedding similarity: probability assigned to correct emotion.
        """
        group_size = self.args.generation_config.num_return_sequences
        batch_size = batch['seq_labels'].shape[0]  # Original batch size (before K samples per input)

        # Compute similarity scores for all samples
        with torch.no_grad():
            sim_result = self.compute_sim_train(model, batch)
            softmax_sim = sim_result['softmax_sim']  # [batch_size * group_size, num_emotions]
            seq_labels = sim_result['seq_labels']    # [batch_size * group_size]

        # Embedding-based reward: probability assigned to correct emotion
        embedding_probs = torch.gather(softmax_sim, dim=1, index=seq_labels.unsqueeze(1)).squeeze(1)
        embedding_rewards = embedding_probs * self.args.reward_softmax_weight
        rewards = embedding_rewards

        # Store old decision log-probabilities for PPO ratio (policy over emotions)
        decision_logps = torch.log(embedding_probs.clamp_min(1e-12))
        batch['old_decision_logps'] = decision_logps.detach()

        # Reshape rewards for advantage computation: [batch_size, group_size]
        rewards = rewards.view(batch_size, group_size)

        # Compute advantages
        if self.args.advantage_type == "gaussian":
            # Gaussian normalization: (r - mean) / std
            mean_rewards = rewards.mean(dim=1, keepdim=True)
            std_rewards = rewards.std(dim=1, keepdim=True) + 1e-8
            advantages = (rewards - mean_rewards) / std_rewards
        elif self.args.advantage_type == "loo":
            # Leave-one-out baseline: r_i - mean(r_j for j != i)
            sum_rewards = rewards.sum(dim=1, keepdim=True)
            baseline = (sum_rewards - rewards) / (group_size - 1)
            advantages = rewards - baseline
        else:
            raise ValueError(f"Unknown advantage type: {self.args.advantage_type}")

        # Flatten advantages back: [batch_size * group_size]
        advantages = advantages.view(-1)

        # Store for policy gradient
        batch['advantages'] = advantages.detach()
        batch['rewards'] = rewards.view(-1).detach()

        # Log metrics
        self.store_metrics({
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'embedding_reward': embedding_rewards.mean().item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
        })

        return batch

    def batch_forward(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        PPO-style clipped policy gradient update.
        
        Uses advantages computed from similarity rewards to update policy.
        """
        group_size = self.args.generation_config.num_return_sequences

        # Compute current policy log probabilities and user embeddings
        per_token_logps, loss_mask, user_hs = self._efficient_forward(
            model, batch, prefix="multi_user", return_with_last_hidden_states=True
        )

        # Compute per-sequence log probability (sum of token log probs)
        seq_logps = (per_token_logps * loss_mask).sum(dim=1)
        seq_lengths = loss_mask.sum(dim=1)
        seq_logps = seq_logps / (seq_lengths + 1e-8)

        # Get advantages and old log probabilities for PPO ratio
        advantages = batch['advantages']
        old_decision_logps = batch['old_decision_logps']
        old_per_token_logps = batch['old_per_token_logps']

        # PPO-style clipped objective
        epsilon_low = self.args.epsilon_low
        epsilon_high = self.args.epsilon_high

        # Token-level PPO loss for reasoning (all trajectories)
        token_ratio = torch.exp(per_token_logps - old_per_token_logps)
        token_clipped_ratio = torch.clamp(token_ratio, 1 - epsilon_low, 1 + epsilon_high)
        token_adv = advantages.unsqueeze(1)
        token_pg = -torch.min(token_ratio * token_adv, token_clipped_ratio * token_adv)
        token_pg = token_pg * loss_mask
        token_loss = token_pg.sum() / loss_mask.sum().clamp(min=1.0)

        # Decision-level PPO loss (winner-only) based on similarity policy
        if self.item_hs is None:
            self._generate_item_embeddings(model)
        seq_labels = batch['seq_labels'].repeat_interleave(group_size)
        sim_out = self.similarity(
            user_hs,
            self.item_hs,
            seq_labels,
            gather_negs_across_processes=False
        )
        decision_logps = torch.log(
            sim_out['softmax_sim']
            .gather(1, seq_labels.unsqueeze(1))
            .squeeze(1)
            .clamp_min(1e-12)
        )
        ratio = torch.exp(decision_logps - old_decision_logps)
        clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)
        decision_pg = -torch.min(ratio * advantages, clipped_ratio * advantages)

        # Winner-only decision update per group
        batch_size = advantages.numel() // group_size
        advantages_grouped = advantages.view(batch_size, group_size)
        best_idx = advantages_grouped.argmax(dim=1)
        mask = torch.zeros_like(advantages_grouped)
        mask.scatter_(1, best_idx.unsqueeze(1), 1.0)
        mask = mask.view(-1)

        decision_pg = decision_pg * mask
        decision_loss = decision_pg.sum() / mask.sum().clamp(min=1.0)

        loss = token_loss + decision_loss

        # Backward pass
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        # Log metrics
        self.store_metrics({
            'pg_loss': decision_pg.mean().item(),
            'token_pg_loss': token_pg.mean().item(),
            'decision_pg_loss': decision_pg.mean().item(),
            'seq_logps': seq_logps.mean().item(),
        })

        return loss.detach()

    def compute_sim_train(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute similarity between generated reasoning embeddings and emotion embeddings.
        Used for computing rewards during training.
        """
        group_size = self.args.generation_config.num_return_sequences

        # Get user (reasoning) embeddings
        _, _, user_hs = self._efficient_forward(
            model, batch, prefix="multi_user", return_with_last_hidden_states=True
        )

        # Get emotion (item) embeddings; refresh periodically to avoid stale embeddings
        if self.item_hs is None or (
            self.args.item_emb_refresh_steps > 0
            and self.state.global_step % self.args.item_emb_refresh_steps == 0
        ):
            self._generate_item_embeddings(model)

        # Compute similarity
        sim_output = self.similarity(
            user_hs,
            self.item_hs,
            batch['seq_labels'].repeat_interleave(group_size),  # Expand labels for all K samples
            gather_negs_across_processes=False  # Don't gather during training
        )

        return {
            'softmax_sim': sim_output['softmax_sim'],
            'seq_labels': batch['seq_labels'].repeat_interleave(group_size),
            'sim_matrix': sim_output['sim_matrix'],
        }

    def compute_sim_val(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute similarity for validation/evaluation.
        No sampling, just single forward pass.
        """
        # Get user embeddings
        user_hs = model(
            attention_mask=batch["user_attention_mask"],
            input_ids=batch["user_input_ids"],
            return_with_last_hidden_states=True,
            return_causal_output=False,
        )

        # Compute similarity
        sim_output = self.similarity(
            user_hs,
            self.item_hs,
            batch['seq_labels'],
            gather_negs_across_processes=self.args.gather_negs_across_processes
        )

        return sim_output

    def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, any]],
                     return_outputs: bool = False, num_items_in_batch: int = None):
        """
        Compute loss during evaluation (not used during training).
        """
        if not model.training:
            # Evaluation mode: compute similarity and metrics
            sim_output = self.compute_sim_val(model, inputs)
            
            # Compute metrics
            preds = sim_output['softmax_sim'].argmax(dim=1)
            labels = inputs['seq_labels']
            
            acc = (preds == labels).float().mean()
            self.store_metrics({'val_acc': acc.item()}, metric_key_prefix="eval")
            
            # Dummy loss for evaluation
            loss = torch.tensor(0.0, device=self.accelerator.device)
            
            if return_outputs:
                # Return softmax_sim as predictions for compute_metrics
                return loss, sim_output['softmax_sim']
            return loss
        else:
            # This shouldn't be called during training (training_step handles it)
            raise ValueError("compute_loss should not be called during training. Use training_step instead.")
