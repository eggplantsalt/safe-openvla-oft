"""
inspect_kkt_pretrain_batch.py

Inspect pretraining-style KKT batch built from chunked OpenVLA samples.
This script does not enter any training loop or download models/data.
"""

import argparse
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
from prismatic.vla.datasets.kkt_openvla_pretrain_batch import (
    KKTPretrainBatchConfig,
    build_kkt_pretrain_batch,
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
    parser = argparse.ArgumentParser(description="Inspect pretraining-style KKT batch.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--require-kkt", action="store_true", default=False)
    parser.add_argument("--action-target", default="safe", choices=["safe", "delta", "nominal"])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--load-images", action="store_true", default=False)
    parser.add_argument("--image-layout", default="chw", choices=["chw", "hwc"])
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--no-wrist-image", action="store_true", default=False)
    args = parser.parse_args()

    dataset = KKTChunkedOpenVLADataset(
        manifest_path=args.manifest_path,
        require_kkt=args.require_kkt,
        action_target=args.action_target,
        chunk_size=args.chunk_size,
        load_images=args.load_images,
        max_samples=None,
    )

    print("Dataset length:", len(dataset))
    if len(dataset) == 0:
        raise SystemExit("error: no records found. Check manifest path, require_kkt filtering, and exported steps.jsonl files.")

    start_index = min(args.start_index, len(dataset) - 1)
    batch_size = min(args.batch_size, len(dataset) - start_index)

    samples = [dataset[i] for i in range(start_index, start_index + batch_size)]
    chunk_batch = collate_kkt_openvla_chunk_samples(samples)

    config = KKTPretrainBatchConfig(
        image_layout=args.image_layout,
        include_wrist_image=not args.no_wrist_image,
        max_text_length=args.max_text_length,
    )
    pretrain_batch = build_kkt_pretrain_batch(chunk_batch, config)

    print("Prompt sample:", pretrain_batch["prompts"][0])
    print("input_ids shape:", pretrain_batch["input_ids"].shape)
    print("attention_mask shape:", pretrain_batch["attention_mask"].shape)
    print("labels shape:", pretrain_batch["labels"].shape)

    pixel_values = pretrain_batch["pixel_values"]
    agentview = pixel_values.get("agentview")
    wrist = pixel_values.get("wrist")
    if args.load_images:
        if agentview is None:
            raise SystemExit("error: pixel_values.agentview is None despite --load-images")
        if not args.no_wrist_image and wrist is None:
            raise SystemExit("error: pixel_values.wrist is None despite --load-images")

    print("pixel_values.agentview shape:", None if agentview is None else agentview.shape)
    print("pixel_values.wrist shape:", None if wrist is None else wrist.shape)

    print("actions shape:", pretrain_batch["actions"].shape)
    print("action_chunk_mask shape:", pretrain_batch["action_chunk_mask"].shape)
    print("action_chunk_mask mean:", float(np.mean(pretrain_batch["action_chunk_mask"])))
    print("proprio shape:", pretrain_batch["proprio"].shape)

    print("kkt_targets.current:")
    for key, value in pretrain_batch["kkt_targets"]["current"].items():
        print("  %s: %s" % (key, _stats(value)))

    print("kkt_targets.chunk:")
    for key, value in pretrain_batch["kkt_targets"]["chunk"].items():
        print("  %s: %s" % (key, _stats(value)))

    print("kkt_masks:")
    for key, value in pretrain_batch["kkt_masks"].items():
        print("  %s: %s" % (key, _stats(value)))

    print("metadata sample:", pretrain_batch["metadata"][0])


if __name__ == "__main__":
    main()
