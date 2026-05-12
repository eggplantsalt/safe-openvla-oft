"""
inspect_kkt_openvla_samples.py

Inspect OpenVLA-style KKT samples exported by safetydistill.
This script does not enter any training loop or download models/data.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

# Allow running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prismatic.vla.datasets.kkt_openvla_dataset import (
    KKTOpenVLASampleDataset,
    collate_kkt_openvla_samples,
)


def _stats(arr: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
        "mean": float(np.mean(arr)) if arr.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect OpenVLA-style KKT samples.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--require-kkt", action="store_true", default=False)
    parser.add_argument("--action-target", default="safe", choices=["safe", "delta", "nominal"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--load-images", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    dataset = KKTOpenVLASampleDataset(
        manifest_path=args.manifest_path,
        require_kkt=args.require_kkt,
        action_target=args.action_target,
        load_images=args.load_images,
        max_samples=args.max_samples,
    )

    print("Dataset length:", len(dataset))
    if len(dataset) == 0:
        raise SystemExit("error: no records found. Check manifest path, require_kkt filtering, and exported steps.jsonl files.")

    sample = dataset[0]
    print("Sample metadata:", json.dumps(sample["metadata"], indent=2))
    print("Instruction:", sample["instruction"])
    print("Agentview image path:", sample["agentview_image_path"])
    print("Wrist image path:", sample["wrist_image_path"])

    if args.load_images:
        if sample["agentview_image"] is None:
            raise SystemExit("error: failed to load agentview image: %s" % sample["agentview_image_path"])
        if sample["wrist_image"] is None:
            raise SystemExit("error: failed to load wrist image: %s" % sample["wrist_image_path"])

    print("State shape:", sample["state"].shape)
    print("Action shape:", sample["action"].shape)
    print("KKT shapes:")
    print("  dual_cbf_main:", sample["dual_cbf_main"].shape)
    print("  active_cbf_main:", sample["active_cbf_main"].shape)
    print("  h:", sample["h"].shape)
    print("  linear_cbf_lhs:", sample["linear_cbf_lhs"].shape)
    print("  constraint_direction:", sample["constraint_direction"].shape)

    batch_size = min(args.batch_size, len(dataset))
    batch_samples = [dataset[i] for i in range(batch_size)]
    batch = collate_kkt_openvla_samples(batch_samples)

    print("Batch stats:")
    numeric_fields = [
        "states",
        "actions",
        "action_nominal",
        "action_safe",
        "action_delta",
        "dual_cbf_main",
        "active_cbf_main",
        "h",
        "linear_cbf_lhs",
        "constraint_direction",
    ]
    for field in numeric_fields:
        stats = _stats(batch[field])
        print("  %s: %s" % (field, stats))

    mask_stats = {k: _stats(v) for k, v in batch["masks"].items()}
    print("  masks:", mask_stats)

    if args.load_images:
        if batch["agentview_images"] is None:
            raise SystemExit("error: agentview image batch is None despite --load-images")
        if batch["wrist_images"] is None:
            raise SystemExit("error: wrist image batch is None despite --load-images")
        print("  agentview_images:", batch["agentview_images"].shape)
        print("  wrist_images:", batch["wrist_images"].shape)


if __name__ == "__main__":
    main()
