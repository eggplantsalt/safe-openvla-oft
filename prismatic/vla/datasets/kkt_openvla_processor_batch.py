"""
kkt_openvla_processor_batch.py

Processor/tokenizer compatibility adapter for OpenVLA-style KKT chunk samples.
This module builds an inspection batch and does not call model forward.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from prismatic.vla.datasets.kkt_openvla_pretrain_batch import (
    KKTPretrainBatchConfig,
    build_kkt_pretrain_batch,
)


@dataclass
class KKTProcessorBatchConfig:
    processor_mode: str = "dummy"  # "dummy" or "real"
    processor_path: Optional[str] = None
    local_files_only: bool = True
    trust_remote_code: bool = True
    image_layout: str = "chw"
    include_wrist_image: bool = True
    max_text_length: int = 128
    prompt_template: str = "What action should the robot take to {instruction}?"
    fail_on_real_processor_error: bool = True


def build_prompt(instruction: str, template: str) -> str:
    return template.format(instruction=instruction)


def load_real_processor(config: KKTProcessorBatchConfig):
    if config.processor_mode != "real":
        raise ValueError("load_real_processor called when processor_mode is not real")
    if not config.processor_path:
        raise ValueError("processor_path is required for real processor mode")

    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise ImportError("transformers is required for real processor mode") from exc

    return AutoProcessor.from_pretrained(
        config.processor_path,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )


def build_kkt_processor_batch(chunk_batch: Dict[str, Any], config: KKTProcessorBatchConfig) -> Dict[str, Any]:
    if config.processor_mode not in ["dummy", "real"]:
        raise ValueError("Invalid processor_mode: %s" % config.processor_mode)

    instructions = chunk_batch.get("instructions")
    if instructions is None:
        raise ValueError("chunk_batch missing instructions")

    prompts = [build_prompt(instr, config.prompt_template) for instr in instructions]

    if config.processor_mode == "dummy":
        pretrain_cfg = KKTPretrainBatchConfig(
            image_layout=config.image_layout,
            include_wrist_image=config.include_wrist_image,
            max_text_length=config.max_text_length,
            prompt_template=config.prompt_template,
        )
        pretrain_batch = build_kkt_pretrain_batch(chunk_batch, pretrain_cfg)
        model_inputs = {
            "pixel_values": pretrain_batch["pixel_values"]["agentview"],
            "input_ids": pretrain_batch["input_ids"],
            "attention_mask": pretrain_batch["attention_mask"],
            "labels": pretrain_batch["labels"],
        }
        if pretrain_batch["pixel_values"].get("wrist") is not None and config.include_wrist_image:
            model_inputs["wrist_pixel_values"] = pretrain_batch["pixel_values"]["wrist"]

        processor_debug = {
            "processor_mode": "dummy",
            "processor_path": None,
            "model_input_keys": list(model_inputs.keys()),
            "note": "Dummy tokenization and simple [0,1] image scaling for inspection only.",
        }
        return _assemble_batch(chunk_batch, prompts, model_inputs, processor_debug)

    processor = load_real_processor(config)
    agentview = chunk_batch.get("agentview_images")
    if agentview is None:
        raise ValueError("agentview_images is required for real processor mode")

    model_inputs = processor(
        prompts,
        list(agentview),
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=config.max_text_length,
    )

    if config.include_wrist_image:
        wrist = chunk_batch.get("wrist_images")
        if wrist is not None:
            model_inputs["wrist_pixel_values"] = wrist

    model_input_keys = list(model_inputs.keys()) if hasattr(model_inputs, "keys") else []
    processor_debug = {
        "processor_mode": "real",
        "processor_path": config.processor_path,
        "model_input_keys": model_input_keys,
        "note": "Real processor used with local_files_only; no model forward executed.",
    }

    return _assemble_batch(chunk_batch, prompts, model_inputs, processor_debug)


def _assemble_batch(
    chunk_batch: Dict[str, Any],
    prompts: List[str],
    model_inputs: Dict[str, Any],
    processor_debug: Dict[str, Any],
) -> Dict[str, Any]:
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
        "action_chunk_mask": chunk_batch["action_chunk_mask"],
    }

    return {
        "mode": processor_debug["processor_mode"],
        "model_inputs": model_inputs,
        "actions": chunk_batch["actions"],
        "action_chunk_mask": chunk_batch["action_chunk_mask"],
        "proprio": chunk_batch["states"],
        "kkt_targets": kkt_targets,
        "kkt_masks": kkt_masks,
        "instructions": chunk_batch["instructions"],
        "prompts": prompts,
        "metadata": chunk_batch["metadata"],
        "raw": chunk_batch["raw"],
        "processor_debug": processor_debug,
    }
