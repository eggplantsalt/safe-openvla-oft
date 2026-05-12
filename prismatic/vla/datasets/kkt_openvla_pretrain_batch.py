"""
kkt_openvla_pretrain_batch.py

Pretraining-style batch adapter for OpenVLA-style KKT chunk samples.
This module builds an inspection batch that resembles finetune.py inputs,
without using real tokenizers or processors.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class KKTPretrainBatchConfig:
    image_layout: str = "chw"  # "chw" or "hwc"
    image_dtype: str = "float32"  # only float32 supported
    image_scale: str = "zero_one"  # scale uint8 to [0,1]
    include_wrist_image: bool = True
    prompt_template: str = "What action should the robot take to {instruction}?"
    dummy_text_tokenization: bool = True
    max_text_length: int = 128


def build_prompt(instruction: str, template: str) -> str:
    return template.format(instruction=instruction)


def dummy_tokenize_texts(texts: List[str], max_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    input_ids = np.zeros((len(texts), max_length), dtype=np.int64)
    attention_mask = np.zeros((len(texts), max_length), dtype=np.int64)
    labels = np.full((len(texts), max_length), -100, dtype=np.int64)

    for i, text in enumerate(texts):
        token_ids = []
        for ch in text:
            token_ids.append((ord(ch) % 997) + 1)
        token_ids = token_ids[:max_length]
        length = len(token_ids)
        if length == 0:
            continue
        input_ids[i, :length] = token_ids
        attention_mask[i, :length] = 1
        labels[i, :length] = input_ids[i, :length]

    return input_ids, attention_mask, labels


def image_to_pixel_values(images: np.ndarray, layout: str = "chw") -> np.ndarray:
    if layout not in ["chw", "hwc"]:
        raise ValueError("Invalid image layout: %s" % layout)
    if images is None:
        return None

    arr = images
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    if layout == "chw":
        arr = np.transpose(arr, (0, 3, 1, 2))
    return arr


def build_kkt_pretrain_batch(chunk_batch: Dict[str, Any], config: KKTPretrainBatchConfig) -> Dict[str, Any]:
    required_keys = [
        "instructions",
        "states",
        "actions",
        "action_chunk_mask",
        "kkt_current",
        "kkt_chunk",
        "metadata",
        "raw",
    ]
    for key in required_keys:
        if key not in chunk_batch:
            raise ValueError("Missing key in chunk_batch: %s" % key)

    instructions = chunk_batch["instructions"]
    prompts = [build_prompt(instr, config.prompt_template) for instr in instructions]

    if not config.dummy_text_tokenization:
        raise ValueError("Only dummy_text_tokenization is supported in Phase 8D.")

    input_ids, attention_mask, labels = dummy_tokenize_texts(prompts, config.max_text_length)

    agentview_images = chunk_batch.get("agentview_images")
    wrist_images = chunk_batch.get("wrist_images")

    pixel_values = {
        "agentview": image_to_pixel_values(agentview_images, layout=config.image_layout)
        if agentview_images is not None
        else None,
        "wrist": image_to_pixel_values(wrist_images, layout=config.image_layout)
        if (wrist_images is not None and config.include_wrist_image)
        else None,
    }

    actions = chunk_batch["actions"]
    action_chunk_mask = chunk_batch["action_chunk_mask"]
    proprio = chunk_batch["states"]

    _validate_array_shape(actions, 3, "actions")
    _validate_array_shape(action_chunk_mask, 2, "action_chunk_mask")
    _validate_array_shape(proprio, 2, "proprio")

    kkt_current = chunk_batch["kkt_current"]
    kkt_chunk = chunk_batch["kkt_chunk"]

    kkt_targets = {
        "current": {
            "dual_cbf_main": kkt_current["dual_cbf_main"],
            "active_cbf_main": kkt_current["active_cbf_main"],
            "h": kkt_current["h"],
            "linear_cbf_lhs": kkt_current["linear_cbf_lhs"],
            "constraint_direction": kkt_current["constraint_direction"],
        },
        "chunk": {
            "dual_cbf_main": kkt_chunk["dual_cbf_main"],
            "active_cbf_main": kkt_chunk["active_cbf_main"],
            "h": kkt_chunk["h"],
            "linear_cbf_lhs": kkt_chunk["linear_cbf_lhs"],
            "constraint_direction": kkt_chunk["constraint_direction"],
        },
    }

    kkt_masks = {
        "current_has_kkt": kkt_current["has_kkt"],
        "current_qp_valid": kkt_current["qp_valid"],
        "chunk_has_kkt": kkt_chunk["has_kkt"],
        "chunk_qp_valid": kkt_chunk["qp_valid"],
        "action_chunk_mask": action_chunk_mask,
    }

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "actions": actions,
        "action_chunk_mask": action_chunk_mask,
        "proprio": proprio,
        "kkt_targets": kkt_targets,
        "kkt_masks": kkt_masks,
        "instructions": instructions,
        "prompts": prompts,
        "metadata": chunk_batch["metadata"],
        "raw": chunk_batch["raw"],
    }


def _validate_array_shape(arr: np.ndarray, ndim: int, name: str) -> None:
    if not isinstance(arr, np.ndarray):
        raise ValueError("%s must be a numpy array" % name)
    if arr.ndim != ndim:
        raise ValueError("%s must be %dD, got %dD" % (name, ndim, arr.ndim))
