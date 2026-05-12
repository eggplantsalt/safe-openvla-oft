"""Inspect KKT finetune hooks with fake finetune outputs.

This script does not call real model.forward and runs on CPU.
"""

import argparse

import numpy as np
import torch

from prismatic.training.kkt_finetune_hooks import (
    KKTFinetuneHookConfig,
    compute_kkt_auxiliary_loss_from_finetune_outputs,
    init_kkt_heads_for_model,
)
from prismatic.vla.datasets.kkt_openvla_chunk_dataset import (
    KKTChunkedOpenVLADataset,
    collate_kkt_openvla_chunk_samples,
)
from prismatic.vla.datasets.kkt_openvla_processor_batch import (
    KKTProcessorBatchConfig,
    build_kkt_processor_batch,
)


def build_action_masks(
    batch_size: int,
    seq_len: int,
    num_action_tokens: int,
) -> tuple:
    if num_action_tokens <= 0:
        raise ValueError("num_action_tokens must be positive")
    if num_action_tokens > seq_len:
        raise ValueError("num_action_tokens must be <= seq_len")

    current_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    next_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    action_indices = list(range(seq_len - num_action_tokens, seq_len))
    current_index = action_indices[0]
    next_indices = action_indices[1:]

    current_mask[:, current_index] = True
    if next_indices:
        next_mask[:, next_indices] = True

    return current_mask, next_mask


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
    parser = argparse.ArgumentParser(description="Inspect KKT finetune hooks with fake outputs")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--require-kkt", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-action-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction-loss-type", choices=["cosine", "mse"], default="cosine")
    parser.add_argument("--disable-current-kkt", action="store_true")
    parser.add_argument("--disable-chunk-kkt", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset = KKTChunkedOpenVLADataset(
        manifest_path=args.manifest_path,
        require_kkt=args.require_kkt,
        action_target="safe",
        chunk_size=args.chunk_size,
        load_images=False,
    )
    samples = get_batch_samples(dataset, args.start_index, args.batch_size)
    chunk_batch = collate_kkt_openvla_chunk_samples(samples)

    processor_cfg = KKTProcessorBatchConfig(processor_mode="dummy")
    batch = build_kkt_processor_batch(chunk_batch, processor_cfg)

    output_hidden_states_last = torch.randn(args.batch_size, args.seq_len, args.hidden_dim)
    current_action_mask, next_actions_mask = build_action_masks(
        args.batch_size,
        args.seq_len,
        args.num_action_tokens,
    )

    direction_dim = int(batch["kkt_targets"]["current"]["constraint_direction"].shape[-1])
    kkt_head = init_kkt_heads_for_model(
        hidden_dim=args.hidden_dim,
        chunk_size=args.chunk_size,
        direction_dim=direction_dim,
        use_current=not args.disable_current_kkt,
        use_chunk=not args.disable_chunk_kkt,
    )

    hook_config = KKTFinetuneHookConfig(
        enable_kkt_sense_training=True,
        kkt_chunk_size=args.chunk_size,
        kkt_hidden_dim=args.hidden_dim,
        kkt_direction_dim=direction_dim,
        use_current_kkt_loss=not args.disable_current_kkt,
        use_chunk_kkt_loss=not args.disable_chunk_kkt,
        direction_loss_type=args.direction_loss_type,
    )

    output = compute_kkt_auxiliary_loss_from_finetune_outputs(
        kkt_head=kkt_head,
        output_hidden_states_last=output_hidden_states_last,
        current_action_mask=current_action_mask,
        next_actions_mask=next_actions_mask,
        batch=batch,
        hook_config=hook_config,
        predicted_actions=None,
    )

    if output is None:
        raise ValueError("Expected KKT hook output")

    print(f"batch actions shape: {tuple(batch['actions'].shape)}")
    print(f"current_action_mask shape: {tuple(current_action_mask.shape)}")
    print(f"next_actions_mask shape: {tuple(next_actions_mask.shape)}")
    print(f"action_chunk_mask mean: {as_mean(batch['action_chunk_mask'])}")
    print(f"kkt_debug: {output['kkt_debug']}")
    print_kkt_prediction_shapes(output["kkt_predictions"])

    for name, value in output["kkt_loss_components"].items():
        if isinstance(value, dict):
            print(f"{name} components: {list(value.keys())}")

    print(f"kkt_loss: {float(output['kkt_loss'])}")
    print(f"kkt_current_loss: {float(output['kkt_current_loss'])}")
    print(f"kkt_chunk_loss: {float(output['kkt_chunk_loss'])}")

    assert_finite("kkt_loss", output["kkt_loss"])
    assert_finite("kkt_current_loss", output["kkt_current_loss"])
    assert_finite("kkt_chunk_loss", output["kkt_chunk_loss"])

    if args.disable_current_kkt and abs(float(output["kkt_current_loss"])) > 1e-6:
        raise ValueError("Expected kkt_current_loss to be 0.0 when disable-current-kkt is set")
    if args.disable_chunk_kkt and abs(float(output["kkt_chunk_loss"])) > 1e-6:
        raise ValueError("Expected kkt_chunk_loss to be 0.0 when disable-chunk-kkt is set")


if __name__ == "__main__":
    main()
