"""Inspect action-token hidden-state extraction for KKT heads.

This script performs CPU-only dry-run with fake tensors and does not call model.forward.
"""

import argparse

import torch

from prismatic.models.kkt_heads import KKTHeadConfig, KKTMultiHead
from prismatic.models.kkt_hidden_extract import extract_kkt_head_inputs
from prismatic.training.kkt_losses import KKTTrainingLossConfig, compute_total_kkt_sense_loss
from prismatic.vla.constants import ACTION_DIM


def build_action_token_mask(
    batch_size: int,
    seq_len: int,
    num_action_tokens: int,
    pattern: str,
    make_ragged: bool,
) -> torch.Tensor:
    if num_action_tokens <= 0:
        raise ValueError("num_action_tokens must be positive")
    if num_action_tokens > seq_len:
        raise ValueError("num_action_tokens must be <= seq_len")

    counts = [num_action_tokens for _ in range(batch_size)]
    if make_ragged and batch_size > 1:
        if num_action_tokens > 1:
            counts[1] = num_action_tokens - 1
        elif num_action_tokens < seq_len:
            counts[1] = num_action_tokens + 1

    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for b in range(batch_size):
        count = counts[b]
        if pattern == "tail":
            indices = list(range(seq_len - count, seq_len))
        elif pattern == "interleaved":
            step = max(1, (seq_len - 1) // max(1, count - 1))
            indices = [i * step for i in range(count)]
        else:
            raise ValueError("Unknown mask pattern")
        mask[b, indices] = True

    return mask


def build_fake_kkt_batch(
    batch_size: int,
    chunk_size: int,
    direction_dim: int,
) -> dict:
    actions = torch.randn(batch_size, chunk_size, ACTION_DIM)
    action_chunk_mask = torch.ones(batch_size, chunk_size)

    current_targets = {
        "dual_cbf_main": torch.randn(batch_size, 1),
        "active_cbf_main": torch.randint(0, 2, (batch_size, 1)).float(),
        "h": torch.randn(batch_size, 1),
        "constraint_direction": torch.randn(batch_size, direction_dim),
    }
    chunk_targets = {
        "dual_cbf_main": torch.randn(batch_size, chunk_size, 1),
        "active_cbf_main": torch.randint(0, 2, (batch_size, chunk_size, 1)).float(),
        "h": torch.randn(batch_size, chunk_size, 1),
        "constraint_direction": torch.randn(batch_size, chunk_size, direction_dim),
    }

    kkt_masks = {
        "current_has_kkt": torch.ones(batch_size, 1),
        "current_qp_valid": torch.ones(batch_size, 1),
        "chunk_has_kkt": torch.ones(batch_size, chunk_size, 1),
        "chunk_qp_valid": torch.ones(batch_size, chunk_size, 1),
        "action_chunk_mask": action_chunk_mask,
    }

    batch = {
        "actions": actions,
        "action_chunk_mask": action_chunk_mask,
        "kkt_targets": {
            "current": current_targets,
            "chunk": chunk_targets,
        },
        "kkt_masks": kkt_masks,
    }
    return batch


def print_kkt_outputs(outputs: dict) -> None:
    current = outputs.get("kkt_current")
    chunk = outputs.get("kkt_chunk")

    if current is not None:
        for name, tensor in current.items():
            print(f"kkt_current.{name} shape: {tuple(tensor.shape)}")
    if chunk is not None:
        for name, tensor in chunk.items():
            print(f"kkt_chunk.{name} shape: {tuple(tensor.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect KKT action-token hidden extraction")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--num-action-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-pattern", choices=["tail", "interleaved"], default="tail")
    parser.add_argument("--make-ragged-mask", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    last_hidden_states = torch.randn(args.batch_size, args.seq_len, args.hidden_dim)
    action_token_mask = build_action_token_mask(
        args.batch_size,
        args.seq_len,
        args.num_action_tokens,
        args.mask_pattern,
        args.make_ragged_mask,
    )

    print(f"last_hidden_states shape: {tuple(last_hidden_states.shape)}")
    print(f"action_token_mask shape: {tuple(action_token_mask.shape)}")

    try:
        head_inputs = extract_kkt_head_inputs(
            last_hidden_states,
            action_token_mask,
            chunk_size=args.chunk_size,
            current_index=0,
        )
    except ValueError as exc:
        if args.make_ragged_mask:
            print(f"Expected failure: {exc}")
            return
        raise

    print(f"num_action_tokens: {head_inputs['num_action_tokens']}")
    print(f"current_hidden shape: {tuple(head_inputs['current_hidden'].shape)}")
    print(f"chunk_hidden shape: {tuple(head_inputs['chunk_hidden'].shape)}")

    config = KKTHeadConfig(
        hidden_dim=args.hidden_dim,
        chunk_size=args.chunk_size,
    )
    heads = KKTMultiHead(config)
    outputs = heads(
        current_hidden=head_inputs["current_hidden"],
        chunk_hidden=head_inputs["chunk_hidden"],
    )

    print_kkt_outputs(outputs)

    predictions = {
        "actions": torch.randn(args.batch_size, args.chunk_size, ACTION_DIM),
        "kkt_current": outputs.get("kkt_current"),
        "kkt_chunk": outputs.get("kkt_chunk"),
    }
    batch = build_fake_kkt_batch(args.batch_size, args.chunk_size, config.direction_dim)
    loss_config = KKTTrainingLossConfig()
    losses = compute_total_kkt_sense_loss(predictions, batch, loss_config)
    print(f"total_loss: {float(losses['total_loss'])}")


if __name__ == "__main__":
    main()
