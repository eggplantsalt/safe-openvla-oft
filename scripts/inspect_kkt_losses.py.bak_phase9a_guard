"""
inspect_kkt_losses.py

Dry-run KKT loss inspection using fake predictions on CPU.
This script does not call model forward or backward.
"""

import argparse
import os
import sys

import numpy as np
import torch

# Allow running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prismatic.training.kkt_losses import KKTTrainingLossConfig, compute_total_kkt_sense_loss
from prismatic.vla.datasets.kkt_openvla_chunk_dataset import (
    KKTChunkedOpenVLADataset,
    collate_kkt_openvla_chunk_samples,
)
from prismatic.vla.datasets.kkt_openvla_processor_batch import (
    KKTProcessorBatchConfig,
    build_kkt_processor_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run KKT loss inspection.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--require-kkt", action="store_true", default=False)
    parser.add_argument("--action-target", default="safe", choices=["safe", "delta", "nominal"])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction-loss-type", default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--disable-current-kkt-loss", action="store_true", default=False)
    parser.add_argument("--disable-chunk-kkt-loss", action="store_true", default=False)
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
        return

    start_index = min(args.start_index, len(dataset) - 1)
    batch_size = min(args.batch_size, len(dataset) - start_index)
    samples = [dataset[i] for i in range(start_index, start_index + batch_size)]
    chunk_batch = collate_kkt_openvla_chunk_samples(samples)

    processor_config = KKTProcessorBatchConfig(
        processor_mode="dummy",
        image_layout="chw",
        include_wrist_image=False,
        max_text_length=args.max_text_length,
    )
    batch = build_kkt_processor_batch(chunk_batch, processor_config)

    target_actions = torch.from_numpy(batch["actions"]).float()
    action_mask = torch.from_numpy(batch["action_chunk_mask"]).float()

    noise = torch.randn_like(target_actions) * 0.01
    pred_actions = target_actions + noise

    target_current = {
        k: torch.from_numpy(v).float()
        for k, v in batch["kkt_targets"]["current"].items()
    }
    target_chunk = {
        k: torch.from_numpy(v).float()
        for k, v in batch["kkt_targets"]["chunk"].items()
    }

    pred_current = {
        "dual_cbf_main": target_current["dual_cbf_main"] + torch.randn_like(target_current["dual_cbf_main"]) * 0.01,
        "active_cbf_main_logit": (target_current["active_cbf_main"] * 2.0 - 1.0) + torch.randn_like(target_current["active_cbf_main"]) * 0.01,
        "h": target_current["h"] + torch.randn_like(target_current["h"]) * 0.01,
        "constraint_direction": target_current["constraint_direction"] + torch.randn_like(target_current["constraint_direction"]) * 0.01,
    }

    pred_chunk = {
        "dual_cbf_main": target_chunk["dual_cbf_main"] + torch.randn_like(target_chunk["dual_cbf_main"]) * 0.01,
        "active_cbf_main_logit": (target_chunk["active_cbf_main"] * 2.0 - 1.0) + torch.randn_like(target_chunk["active_cbf_main"]) * 0.01,
        "h": target_chunk["h"] + torch.randn_like(target_chunk["h"]) * 0.01,
        "constraint_direction": target_chunk["constraint_direction"] + torch.randn_like(target_chunk["constraint_direction"]) * 0.01,
    }

    kkt_masks = {
        "current_has_kkt": torch.from_numpy(batch["kkt_masks"]["current_has_kkt"]).float(),
        "current_qp_valid": torch.from_numpy(batch["kkt_masks"]["current_qp_valid"]).float(),
        "chunk_has_kkt": torch.from_numpy(batch["kkt_masks"]["chunk_has_kkt"]).float(),
        "chunk_qp_valid": torch.from_numpy(batch["kkt_masks"]["chunk_qp_valid"]).float(),
        "action_chunk_mask": action_mask,
    }

    loss_config = KKTTrainingLossConfig(
        direction_loss_type=args.direction_loss_type,
        use_current_kkt_loss=not args.disable_current_kkt_loss,
        use_chunk_kkt_loss=not args.disable_chunk_kkt_loss,
    )

    predictions = {
        "actions": pred_actions,
        "kkt_current": pred_current,
        "kkt_chunk": pred_chunk,
    }

    loss_output = compute_total_kkt_sense_loss(
        predictions,
        {
            "actions": target_actions,
            "action_chunk_mask": action_mask,
            "kkt_targets": {
                "current": target_current,
                "chunk": target_chunk,
            },
            "kkt_masks": kkt_masks,
        },
        loss_config,
    )

    print("Action shape:", tuple(target_actions.shape))
    print("Action chunk mask mean:", float(action_mask.mean().item()))
    print("Action loss:", float(loss_output["action_loss"].item()))

    if loss_config.use_current_kkt_loss:
        print("Current KKT loss components:")
        print("  dual_loss:", float(loss_output["current_dual_loss"].item()))
        print("  active_loss:", float(loss_output["current_active_loss"].item()))
        print("  h_loss:", float(loss_output["current_h_loss"].item()))
        print("  direction_loss:", float(loss_output["current_direction_loss"].item()))
        print("  total_current_kkt_loss:", float(loss_output["current_total_current_kkt_loss"].item()))

    if loss_config.use_chunk_kkt_loss:
        print("Chunk KKT loss components:")
        print("  dual_loss:", float(loss_output["chunk_dual_loss"].item()))
        print("  active_loss:", float(loss_output["chunk_active_loss"].item()))
        print("  h_loss:", float(loss_output["chunk_h_loss"].item()))
        print("  direction_loss:", float(loss_output["chunk_direction_loss"].item()))
        print("  total_chunk_kkt_loss:", float(loss_output["chunk_total_chunk_kkt_loss"].item()))

    print("Total loss:", float(loss_output["total_loss"].item()))


if __name__ == "__main__":
    main()
