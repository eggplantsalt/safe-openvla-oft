"""Inspect KKT finetune dataset and collator (local-only processor).

This script does not call model.forward or perform training.
"""

import argparse

import numpy as np
import torch
from transformers import AutoProcessor

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.kkt_finetune_dataset import KKTFinetuneCollator, KKTFinetuneDataset


def as_mean(value):
    if isinstance(value, torch.Tensor):
        return float(value.float().mean())
    if isinstance(value, np.ndarray):
        return float(value.mean())
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect KKT finetune dataset batches")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--vla-path", required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--action-target", choices=["safe", "delta", "nominal"], default="safe")
    parser.add_argument("--require-kkt", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use-wrist-image", action="store_true")
    parser.add_argument("--use-proprio", action="store_true", default=True)
    parser.add_argument("--no-proprio", action="store_false", dest="use_proprio")
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.vla_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    dataset = KKTFinetuneDataset(
        manifest_path=args.manifest_path,
        action_tokenizer=action_tokenizer,
        tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        require_kkt=args.require_kkt,
        action_target=args.action_target,
        chunk_size=args.chunk_size,
        load_images=True,
        use_wrist_image=args.use_wrist_image,
        use_proprio=args.use_proprio,
        max_samples=args.max_samples,
        dataset_name="kkt_openvla",
    )

    end_index = min(len(dataset), args.start_index + args.batch_size)
    if end_index - args.start_index < args.batch_size:
        raise ValueError("Not enough samples to build batch")

    instances = [dataset[idx] for idx in range(args.start_index, end_index)]

    base_collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    collator = KKTFinetuneCollator(base_collator)
    batch = collator(instances)

    print(f"dataset length: {len(dataset)}")
    print(f"batch keys: {list(batch.keys())}")
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"labels shape: {tuple(batch['labels'].shape)}")
    print(f"attention_mask shape: {tuple(batch['attention_mask'].shape)}")
    print(f"pixel_values shape: {tuple(batch['pixel_values'].shape)}")
    print(f"actions shape: {tuple(batch['actions'].shape)}")

    proprio = batch.get("proprio")
    if proprio is not None:
        print(f"proprio shape: {tuple(proprio.shape)}")
    print(f"action_chunk_mask shape: {tuple(batch['action_chunk_mask'].shape)}")
    print(f"action_chunk_mask mean: {as_mean(batch['action_chunk_mask'])}")

    kkt_targets = batch.get("kkt_targets", {})
    kkt_masks = batch.get("kkt_masks", {})

    for scope in ["current", "chunk"]:
        if scope in kkt_targets:
            for key, value in kkt_targets[scope].items():
                print(f"kkt_targets.{scope}.{key} shape: {tuple(value.shape)}")

    for key, value in kkt_masks.items():
        if isinstance(value, torch.Tensor):
            print(f"kkt_masks.{key} shape: {tuple(value.shape)} mean: {as_mean(value)}")

    ground_truth_token_ids = batch["labels"][:, 1:]
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    print(f"current_action_mask shape: {tuple(current_action_mask.shape)} sum: {int(current_action_mask.sum())}")
    print(f"next_actions_mask shape: {tuple(next_actions_mask.shape)} sum: {int(next_actions_mask.sum())}")


if __name__ == "__main__":
    main()
