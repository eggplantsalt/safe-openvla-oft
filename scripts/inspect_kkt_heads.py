"""
inspect_kkt_heads.py

Dry-run KKT auxiliary heads with fake hidden states and loss computation.
This script does not call model forward or backward.
"""

import argparse
import os
import sys

import numpy as np
import torch

# Allow running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prismatic.models.kkt_heads import KKTHeadConfig, KKTMultiHead
from prismatic.training.kkt_losses import KKTTrainingLossConfig, compute_total_kkt_sense_loss
from prismatic.vla.datasets.kkt_openvla_chunk_dataset import (
    KKTChunkedOpenVLADataset,
    collate_kkt_openvla_chunk_samples,
)
from prismatic.vla.datasets.kkt_openvla_processor_batch import (
    KKTProcessorBatchConfig,
    build_kkt_processor_batch,
)


def _is_finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run KKT heads with fake hidden states.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--require-kkt", action="store_true", default=False)
    parser.add_argument("--action-target", default="safe", choices=["safe", "delta", "nominal"])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction-loss-type", default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--disable-current-head", action="store_true", default=False)
    parser.add_argument("--disable-chunk-head", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = KKTChunkedOpenVLADataset(
        manifest_path=args.manifest_path,
        require_kkt=args.require_kkt,
        action_target=args.action_target,
        chunk_size=args.chunk_size,
        load_images=False,
        max_samples=None,
    )

    if len(dataset) == 0:
        print("No records found.")
        raise SystemExit(1)

    start_index = min(args.start_index, len(dataset) - 1)
    batch_size = min(args.batch_size, len(dataset) - start_index)
    samples = [dataset[i] for i in range(start_index, start_index + batch_size)]
    chunk_batch = collate_kkt_openvla_chunk_samples(samples)

    processor_config = KKTProcessorBatchConfig(
        processor_mode="dummy",
        image_layout="chw",
        include_wrist_image=False,
        max_text_length=64,
    )
    batch = build_kkt_processor_batch(chunk_batch, processor_config)

    current_hidden = torch.randn(batch_size, args.hidden_dim)
    chunk_hidden = torch.randn(batch_size, args.chunk_size, args.hidden_dim)

    head_config = KKTHeadConfig(
        hidden_dim=args.hidden_dim,
        chunk_size=args.chunk_size,
        direction_dim=6,
        predict_current=not args.disable_current_head,
        predict_chunk=not args.disable_chunk_head,
    )
    head = KKTMultiHead(head_config)
    head_outputs = head(
        current_hidden=current_hidden if head_config.predict_current else None,
        chunk_hidden=chunk_hidden if head_config.predict_chunk else None,
    )

    pred_actions = torch.from_numpy(batch["actions"]).float() + torch.randn_like(torch.from_numpy(batch["actions"]).float()) * 0.01

    predictions = {
        "actions": pred_actions,
        "kkt_current": head_outputs.get("kkt_current"),
        "kkt_chunk": head_outputs.get("kkt_chunk"),
    }

    loss_config = KKTTrainingLossConfig(
        direction_loss_type=args.direction_loss_type,
        use_current_kkt_loss=not args.disable_current_head,
        use_chunk_kkt_loss=not args.disable_chunk_head,
    )

    target_actions = torch.from_numpy(batch["actions"]).float()
    action_chunk_mask = torch.from_numpy(batch["action_chunk_mask"]).float()

    kkt_targets_current = {k: torch.from_numpy(v).float() for k, v in batch["kkt_targets"]["current"].items()}
    kkt_targets_chunk = {k: torch.from_numpy(v).float() for k, v in batch["kkt_targets"]["chunk"].items()}

    kkt_masks = {
        "current_has_kkt": torch.from_numpy(batch["kkt_masks"]["current_has_kkt"]).float(),
        "current_qp_valid": torch.from_numpy(batch["kkt_masks"]["current_qp_valid"]).float(),
        "chunk_has_kkt": torch.from_numpy(batch["kkt_masks"]["chunk_has_kkt"]).float(),
        "chunk_qp_valid": torch.from_numpy(batch["kkt_masks"]["chunk_qp_valid"]).float(),
        "action_chunk_mask": action_chunk_mask,
    }

    losses = compute_total_kkt_sense_loss(
        predictions,
        {
            "actions": target_actions,
            "action_chunk_mask": action_chunk_mask,
            "kkt_targets": {
                "current": kkt_targets_current,
                "chunk": kkt_targets_chunk,
            },
            "kkt_masks": kkt_masks,
        },
        loss_config,
    )

    print("current_hidden shape:", tuple(current_hidden.shape))
    print("chunk_hidden shape:", tuple(chunk_hidden.shape))

    if head_outputs.get("kkt_current") is not None:
        for k, v in head_outputs["kkt_current"].items():
            print("current head %s shape: %s" % (k, tuple(v.shape)))

    if head_outputs.get("kkt_chunk") is not None:
        for k, v in head_outputs["kkt_chunk"].items():
            print("chunk head %s shape: %s" % (k, tuple(v.shape)))

    print("action_loss:", float(losses["action_loss"].item()))
    if loss_config.use_current_kkt_loss:
        print("current kkt loss components:")
        print("  dual_loss:", float(losses["current_dual_loss"].item()))
        print("  active_loss:", float(losses["current_active_loss"].item()))
        print("  h_loss:", float(losses["current_h_loss"].item()))
        print("  direction_loss:", float(losses["current_direction_loss"].item()))
        print("  total_current_kkt_loss:", float(losses["current_total_current_kkt_loss"].item()))

    if loss_config.use_chunk_kkt_loss:
        print("chunk kkt loss components:")
        print("  dual_loss:", float(losses["chunk_dual_loss"].item()))
        print("  active_loss:", float(losses["chunk_active_loss"].item()))
        print("  h_loss:", float(losses["chunk_h_loss"].item()))
        print("  direction_loss:", float(losses["chunk_direction_loss"].item()))
        print("  total_chunk_kkt_loss:", float(losses["chunk_total_chunk_kkt_loss"].item()))

    print("total_loss:", float(losses["total_loss"].item()))

    finite_checks = [losses["total_loss"]]
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            finite_checks.append(value)

    if not all(_is_finite(v) for v in finite_checks):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
