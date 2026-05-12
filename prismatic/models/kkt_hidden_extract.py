"""
Helper utilities for extracting action-token hidden states for KKT heads.

These functions are independent of OpenVLA forward and are safe for CPU dry-run.
"""

from typing import Dict, Optional

import torch


def validate_action_token_mask(
    mask: torch.Tensor,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.dim() != 2:
        raise ValueError("mask must have shape (B, L)")
    if batch_size is not None and mask.shape[0] != batch_size:
        raise ValueError("mask batch size does not match")
    if seq_len is not None and mask.shape[1] != seq_len:
        raise ValueError("mask seq_len does not match")

    if mask.dtype != torch.bool:
        is_binary = torch.all((mask == 0) | (mask == 1))
        if not bool(is_binary):
            raise ValueError("mask must contain only 0/1 values")
        mask = mask.to(dtype=torch.bool)

    return mask


def extract_action_token_hidden_states(
    last_hidden_states: torch.Tensor,
    action_token_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(last_hidden_states, torch.Tensor):
        raise TypeError("last_hidden_states must be a torch.Tensor")
    if last_hidden_states.dim() != 3:
        raise ValueError("last_hidden_states must have shape (B, L, H)")

    batch_size, seq_len, hidden_dim = last_hidden_states.shape
    action_token_mask = validate_action_token_mask(action_token_mask, batch_size, seq_len)
    action_token_mask = action_token_mask.to(device=last_hidden_states.device)

    token_counts = action_token_mask.sum(dim=1)
    if torch.any(token_counts == 0):
        raise ValueError("action_token_mask must include at least one token per batch")
    if not torch.all(token_counts == token_counts[0]):
        raise ValueError("action_token_mask must select the same number of tokens per batch")

    num_action_tokens = int(token_counts[0].item())
    action_hidden = last_hidden_states[action_token_mask].reshape(batch_size, num_action_tokens, hidden_dim)
    return action_hidden


def split_current_and_chunk_hidden(
    action_hidden: torch.Tensor,
    current_index: int = 0,
    chunk_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if not isinstance(action_hidden, torch.Tensor):
        raise TypeError("action_hidden must be a torch.Tensor")
    if action_hidden.dim() != 3:
        raise ValueError("action_hidden must have shape (B, T, H)")

    batch_size, num_tokens, hidden_dim = action_hidden.shape
    if current_index < 0 or current_index >= num_tokens:
        raise ValueError("current_index is out of range")

    if chunk_size is not None:
        if num_tokens < chunk_size:
            raise ValueError("action_hidden does not have enough tokens for chunk_size")
        chunk_hidden = action_hidden[:, :chunk_size, :]
    else:
        chunk_hidden = action_hidden

    current_hidden = action_hidden[:, current_index, :]
    return {
        "current_hidden": current_hidden,
        "chunk_hidden": chunk_hidden,
    }


def extract_kkt_head_inputs(
    last_hidden_states: torch.Tensor,
    action_token_mask: torch.Tensor,
    chunk_size: int,
    current_index: int = 0,
) -> Dict[str, torch.Tensor]:
    action_hidden = extract_action_token_hidden_states(last_hidden_states, action_token_mask)
    split = split_current_and_chunk_hidden(action_hidden, current_index=current_index, chunk_size=chunk_size)
    return {
        "action_hidden": action_hidden,
        "current_hidden": split["current_hidden"],
        "chunk_hidden": split["chunk_hidden"],
        "num_action_tokens": action_hidden.shape[1],
    }
