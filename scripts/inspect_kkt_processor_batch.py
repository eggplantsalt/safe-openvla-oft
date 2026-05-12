"""
inspect_kkt_processor_batch.py

Inspect processor/tokenizer compatibility batch built from KKT chunk samples.
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
from prismatic.vla.datasets.kkt_openvla_processor_batch import (
    KKTProcessorBatchConfig,
    build_kkt_processor_batch,
)


def _stats(arr: Any) -> Dict[str, Any]:
    if hasattr(arr, "shape"):
        shape = list(arr.shape)
    else:
        shape = None
    dtype = str(arr.dtype) if hasattr(arr, "dtype") else type(arr).__name__
    if isinstance(arr, np.ndarray) and arr.size:
        return {
            "shape": shape,
            "dtype": dtype,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }
    return {
        "shape": shape,
        "dtype": dtype,
        "min": None,
        "max": None,
        "mean": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect processor/tokenizer compatibility batch.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--require-kkt", action="store_true", default=False)
    parser.add_argument("--action-target", default="safe", choices=["safe", "delta", "nominal"])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--load-images", action="store_true", default=False)
    parser.add_argument("--processor-mode", default="dummy", choices=["dummy", "real"])
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--image-layout", default="chw", choices=["chw", "hwc"])
    parser.add_argument("--no-wrist-image", action="store_true", default=False)
    parser.add_argument("--fallback-to-dummy-on-error", action="store_true", default=False)
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
        print("No records found.")
        return

    if args.processor_mode == "real" and not args.processor_path:
        raise SystemExit("processor_mode=real requires --processor-path")

    start_index = min(args.start_index, len(dataset) - 1)
    batch_size = min(args.batch_size, len(dataset) - start_index)

    samples = [dataset[i] for i in range(start_index, start_index + batch_size)]
    chunk_batch = collate_kkt_openvla_chunk_samples(samples)

    config = KKTProcessorBatchConfig(
        processor_mode=args.processor_mode,
        processor_path=args.processor_path,
        image_layout=args.image_layout,
        include_wrist_image=not args.no_wrist_image,
        max_text_length=args.max_text_length,
    )

    try:
        batch = build_kkt_processor_batch(chunk_batch, config)
    except Exception as exc:
        if args.processor_mode == "real" and args.fallback_to_dummy_on_error:
            print("Real processor failed, falling back to dummy:", str(exc))
            config.processor_mode = "dummy"
            batch = build_kkt_processor_batch(chunk_batch, config)
        else:
            raise

    print("Processor mode:", batch["processor_debug"]["processor_mode"])
    print("Processor path:", batch["processor_debug"]["processor_path"])
    print("Prompt sample:", batch["prompts"][0])
    print("Model input keys:", batch["processor_debug"]["model_input_keys"])

    model_inputs = batch["model_inputs"]
    for key, value in model_inputs.items():
        print("  %s: %s" % (key, _stats(value)))

    print("actions shape:", batch["actions"].shape)
    print("action_chunk_mask shape:", batch["action_chunk_mask"].shape)
    print("action_chunk_mask mean:", float(np.mean(batch["action_chunk_mask"])))
    print("proprio shape:", batch["proprio"].shape)

    print("kkt_targets.current:")
    for key, value in batch["kkt_targets"]["current"].items():
        print("  %s: %s" % (key, _stats(value)))

    print("kkt_targets.chunk:")
    for key, value in batch["kkt_targets"]["chunk"].items():
        print("  %s: %s" % (key, _stats(value)))

    print("kkt_masks:")
    for key, value in batch["kkt_masks"].items():
        print("  %s: %s" % (key, _stats(value)))

    print("metadata sample:", batch["metadata"][0])
    print("processor_debug:", batch["processor_debug"])


if __name__ == "__main__":
    main()
