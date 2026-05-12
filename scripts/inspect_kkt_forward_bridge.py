"""Inspect KKT forward bridge with fake OpenVLA hidden states.

This script builds a Phase 8E dummy processor batch and runs CPU-only dry-run.
"""

import argparse

import numpy as np
import torch

from prismatic.training.kkt_forward_bridge import (
    KKTForwardBridge,
    KKTForwardBridgeConfig,
    compute_kkt_forward_bridge_loss,
)
from prismatic.training.kkt_losses import KKTTrainingLossConfig
from prismatic.vla.constants import ACTION_DIM
from prismatic.vla.datasets.kkt_openvla_chunk_dataset import (
    KKTChunkedOpenVLADataset,
    collate_kkt_openvla_chunk_samples,
)
from prismatic.vla.datasets.kkt_openvla_processor_batch import (
    KKTProcessorBatchConfig,
    build_kkt_processor_batch,
)


def build_action_token_mask(
    batch_size: int,
    seq_len: int,
    num_action_tokens: int,
    pattern: str,
) -> torch.Tensor:
    if num_action_tokens <= 0:
        raise ValueError("num_action_tokens must be positive")
    if num_action_tokens > seq_len:
        raise ValueError("num_action_tokens must be <= seq_len")

    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for b in range(batch_size):
        if pattern == "tail":
            indices = list(range(seq_len - num_action_tokens, seq_len))
        elif pattern == "interleaved":
            step = max(1, (seq_len - 1) // max(1, num_action_tokens - 1))
            indices = [i * step for i in range(num_action_tokens)]
        else:
            raise ValueError("Unknown mask pattern")
        mask[b, indices] = True
    return mask


def get_batch_samples(dataset, start_index: int, batch_size: int):
    samples = []
    end_index = min(len(dataset), start_index + batch_size)
    for idx in range(start_index, end_index):
        samples.append(dataset[idx])
    if len(samples) < batch_size:
        raise ValueError("Not enough samples to build batch")
    return samples


def print_kkt_prediction_shapes(predictions: dict) -> None:
    current = predictions.get("kkt_current")
    chunk = predictions.get("kkt_chunk")

    if current is not None:
        for name, tensor in current.items():
            print(f"kkt_current.{name} shape: {tuple(tensor.shape)}")
    if chunk is not None:
        for name, tensor in chunk.items():
            print(f"kkt_chunk.{name} shape: {tuple(tensor.shape)}")


def as_mean(value):
    if isinstance(value, torch.Tensor):
        return float(value.float().mean())
    if isinstance(value, np.ndarray):
        return float(value.mean())
    return float(value)


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite loss detected: %s" % name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect standalone KKT forward bridge")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--require-kkt", action="store_true")
    parser.add_argument("--action-target", choices=["safe", "delta", "nominal"], default="safe")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-action-tokens", type=int, default=8)
    parser.add_argument("--mask-pattern", choices=["tail", "interleaved"], default="tail")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction-loss-type", choices=["cosine", "mse"], default="cosine")
    parser.add_argument("--disable-current-kkt", action="store_true")
    parser.add_argument("--disable-chunk-kkt", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset = KKTChunkedOpenVLADataset(
        manifest_path=args.manifest_path,
        require_kkt=args.require_kkt,
        action_target=args.action_target,
        chunk_size=args.chunk_size,
        load_images=False,
    )

    samples = get_batch_samples(dataset, args.start_index, args.batch_size)
    chunk_batch = collate_kkt_openvla_chunk_samples(samples)

    processor_cfg = KKTProcessorBatchConfig(processor_mode="dummy")
    batch = build_kkt_processor_batch(chunk_batch, processor_cfg)

    last_hidden_states = torch.randn(args.batch_size, args.seq_len, args.hidden_dim)
    action_token_mask = build_action_token_mask(
        args.batch_size,
        args.seq_len,
        args.num_action_tokens,
        args.mask_pattern,
    )

    direction_dim = int(batch["kkt_targets"]["current"]["constraint_direction"].shape[-1])
    bridge_cfg = KKTForwardBridgeConfig(
        hidden_dim=args.hidden_dim,
        action_dim=ACTION_DIM,
        chunk_size=args.chunk_size,
        direction_dim=direction_dim,
        use_current_kkt=not args.disable_current_kkt,
        use_chunk_kkt=not args.disable_chunk_kkt,
    )
    bridge = KKTForwardBridge(bridge_cfg)

    loss_cfg = KKTTrainingLossConfig(
        direction_loss_type=args.direction_loss_type,
        use_current_kkt_loss=not args.disable_current_kkt,
        use_chunk_kkt_loss=not args.disable_chunk_kkt,
    )

    output = compute_kkt_forward_bridge_loss(
        bridge,
        last_hidden_states,
        action_token_mask,
        batch,
        loss_cfg,
    )

    bridge_output = output["bridge_output"]
    predictions = bridge_output["predictions"]
    hidden = bridge_output["hidden"]
    losses = output["losses"]

    print(f"batch actions shape: {tuple(batch['actions'].shape)}")
    print(f"last_hidden_states shape: {tuple(last_hidden_states.shape)}")
    print(f"action_token_mask shape: {tuple(action_token_mask.shape)}")
    print(f"action_hidden shape: {tuple(hidden['action_hidden'].shape)}")
    print(f"current_hidden shape: {tuple(hidden['current_hidden'].shape)}")
    print(f"chunk_hidden shape: {tuple(hidden['chunk_hidden'].shape)}")
    print(f"pred_actions shape: {tuple(predictions['actions'].shape)}")
    print_kkt_prediction_shapes(predictions)
    print(f"action_chunk_mask mean: {as_mean(batch['action_chunk_mask'])}")

    expected_shape = tuple(batch["actions"].shape)
    if tuple(predictions["actions"].shape) != expected_shape:
        raise ValueError("pred_actions shape mismatch: %s != %s" % (predictions["actions"].shape, expected_shape))

    for name, value in losses.items():
        assert_finite(name, value)
        print(f"{name}: {float(value)}")


if __name__ == "__main__":
    main()
