"""
inspect_kkt_openvla_chunks.py

Inspect chunked OpenVLA-style KKT samples.
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

from prismatic.vla.datasets.kkt_openvla_chunk_dataset import (
    KKTChunkedOpenVLADataset,
    collate_kkt_openvla_chunk_samples,
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
    parser = argparse.ArgumentParser(description="Inspect chunked OpenVLA-style KKT samples.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--require-kkt", action="store_true", default=False)
    parser.add_argument("--action-target", default="safe", choices=["safe", "delta", "nominal"])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--load-images", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    dataset = KKTChunkedOpenVLADataset(
        manifest_path=args.manifest_path,
        require_kkt=args.require_kkt,
        action_target=args.action_target,
        chunk_size=args.chunk_size,
        load_images=args.load_images,
        max_samples=args.max_samples,
    )

    print("Dataset length:", len(dataset))
    if len(dataset) == 0:
        print("No records found.")
        return

    start_index = min(args.start_index, len(dataset) - 1)
    sample = dataset[start_index]

    print("Sample metadata:", json.dumps(sample["metadata"], indent=2))
    print("State shape:", sample["state"].shape)
    print("Actions shape:", sample["actions"].shape)
    print("Action chunk mask:", sample["action_chunk_mask"].tolist())

    print("KKT current stats:")
    for key, value in sample["kkt_current"].items():
        if isinstance(value, np.ndarray):
            print("  %s: %s" % (key, _stats(value)))
        else:
            print("  %s: %s" % (key, value))

    print("KKT chunk stats:")
    for key, value in sample["kkt_chunk"].items():
        print("  %s: %s" % (key, _stats(value)))

    batch_size = min(args.batch_size, len(dataset))
    batch_samples = [dataset[i] for i in range(batch_size)]
    batch = collate_kkt_openvla_chunk_samples(batch_samples)

    print("Batch stats:")
    numeric_fields = [
        "states",
        "actions",
        "action_nominal",
        "action_safe",
        "action_delta",
        "action_chunk_mask",
    ]
    for field in numeric_fields:
        stats = _stats(batch[field])
        print("  %s: %s" % (field, stats))

    print("  kkt_current:")
    for key, value in batch["kkt_current"].items():
        print("    %s: %s" % (key, _stats(value)))

    print("  kkt_chunk:")
    for key, value in batch["kkt_chunk"].items():
        print("    %s: %s" % (key, _stats(value)))

    if args.load_images:
        print("  agentview_images:", batch["agentview_images"].shape if batch["agentview_images"] is not None else None)
        print("  wrist_images:", batch["wrist_images"].shape if batch["wrist_images"] is not None else None)


if __name__ == "__main__":
    main()
