"""
kkt_finetune_hooks.py

Helper utilities for opt-in KKT-SenseVLA hooks in finetune.py.
These helpers do not call model forward and can run on CPU.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from prismatic.models.kkt_heads import KKTHeadConfig, KKTMultiHead
from prismatic.models.kkt_hidden_extract import extract_kkt_head_inputs, validate_action_token_mask
from prismatic.training.kkt_losses import (
    KKTTrainingLossConfig,
    compute_kkt_chunk_losses,
    compute_kkt_current_losses,
    compute_total_kkt_sense_loss,
)


@dataclass
class KKTFinetuneHookConfig:
    enable_kkt_sense_training: bool = False
    kkt_loss_weight: float = 1.0
    kkt_action_loss_weight: float = 1.0
    kkt_dual_loss_weight: float = 0.1
    kkt_active_loss_weight: float = 0.1
    kkt_h_loss_weight: float = 0.05
    kkt_direction_loss_weight: float = 0.05
    use_current_kkt_loss: bool = True
    use_chunk_kkt_loss: bool = True
    direction_loss_type: str = "cosine"
    kkt_hidden_dim: Optional[int] = None
    kkt_chunk_size: Optional[int] = None
    kkt_direction_dim: int = 6


def build_kkt_loss_config(hook_config: KKTFinetuneHookConfig) -> KKTTrainingLossConfig:
    return KKTTrainingLossConfig(
        action_loss_weight=hook_config.kkt_action_loss_weight,
        dual_loss_weight=hook_config.kkt_dual_loss_weight,
        active_loss_weight=hook_config.kkt_active_loss_weight,
        h_loss_weight=hook_config.kkt_h_loss_weight,
        direction_loss_weight=hook_config.kkt_direction_loss_weight,
        use_chunk_kkt_loss=hook_config.use_chunk_kkt_loss,
        use_current_kkt_loss=hook_config.use_current_kkt_loss,
        direction_loss_type=hook_config.direction_loss_type,
    )


def build_action_token_mask_from_finetune_masks(
    current_action_mask: torch.Tensor,
    next_actions_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(current_action_mask, torch.Tensor) or not isinstance(next_actions_mask, torch.Tensor):
        raise TypeError("current_action_mask and next_actions_mask must be torch.Tensor")
    if current_action_mask.dim() != 2 or next_actions_mask.dim() != 2:
        raise ValueError("action masks must have shape (B, L)")
    if current_action_mask.shape != next_actions_mask.shape:
        raise ValueError("current_action_mask and next_actions_mask must have same shape")

    action_token_mask = current_action_mask | next_actions_mask
    return validate_action_token_mask(action_token_mask)


def validate_kkt_batch_for_finetune(
    batch: Dict[str, Any],
    expected_chunk_size: Optional[int] = None,
    expected_action_dim: int = 7,
) -> None:
    required_keys = ["actions", "action_chunk_mask", "kkt_targets", "kkt_masks"]
    for key in required_keys:
        if key not in batch:
            raise ValueError("batch missing required key: %s" % key)

    actions = batch["actions"]
    if isinstance(actions, np.ndarray):
        actions_shape = actions.shape
    elif isinstance(actions, torch.Tensor):
        actions_shape = tuple(actions.shape)
    else:
        raise ValueError("batch.actions must be a numpy array or torch.Tensor")

    if len(actions_shape) != 3 or actions_shape[2] != expected_action_dim:
        raise ValueError("actions must have shape (B, T, %d)" % expected_action_dim)

    if expected_chunk_size is not None and actions_shape[1] != expected_chunk_size:
        raise ValueError("actions chunk size mismatch: %s != %s" % (actions_shape[1], expected_chunk_size))

    kkt_targets = batch["kkt_targets"]
    kkt_masks = batch["kkt_masks"]

    for section in ["current", "chunk"]:
        if section not in kkt_targets:
            raise ValueError("kkt_targets missing section: %s" % section)
        for key in ["dual_cbf_main", "active_cbf_main", "h", "constraint_direction"]:
            if key not in kkt_targets[section]:
                raise ValueError("kkt_targets.%s missing key: %s" % (section, key))

    for key in ["current_has_kkt", "current_qp_valid", "chunk_has_kkt", "chunk_qp_valid", "action_chunk_mask"]:
        if key not in kkt_masks:
            raise ValueError("kkt_masks missing key: %s" % key)


def init_kkt_heads_for_model(
    hidden_dim: int,
    chunk_size: int,
    direction_dim: int = 6,
    use_current: bool = True,
    use_chunk: bool = True,
) -> KKTMultiHead:
    config = KKTHeadConfig(
        hidden_dim=hidden_dim,
        chunk_size=chunk_size,
        direction_dim=direction_dim,
        predict_current=use_current,
        predict_chunk=use_chunk,
    )
    return KKTMultiHead(config)


def _to_tensor(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device)
    if isinstance(value, dict):
        return {k: _to_tensor(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_tensor(v, device) for v in value]
    return value


def compute_kkt_auxiliary_loss_from_finetune_outputs(
    kkt_head: KKTMultiHead,
    output_hidden_states_last: torch.Tensor,
    current_action_mask: torch.Tensor,
    next_actions_mask: torch.Tensor,
    batch: Dict[str, Any],
    hook_config: KKTFinetuneHookConfig,
    predicted_actions: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, Any]]:
    if not hook_config.enable_kkt_sense_training:
        return None
    if kkt_head is None:
        raise ValueError("kkt_head is required when enable_kkt_sense_training is True")
    if hook_config.kkt_chunk_size is None:
        raise ValueError("kkt_chunk_size must be set when enable_kkt_sense_training is True")
    if hook_config.kkt_hidden_dim is None:
        raise ValueError("kkt_hidden_dim must be set when enable_kkt_sense_training is True")

    head_param = next(kkt_head.parameters(), None)
    if head_param is None:
        raise ValueError("kkt_head has no parameters to infer device or dtype")
    target_device = head_param.device
    target_dtype = head_param.dtype

    output_hidden_states_last = output_hidden_states_last.to(device=target_device, dtype=target_dtype)
    current_action_mask = current_action_mask.to(device=target_device)
    next_actions_mask = next_actions_mask.to(device=target_device)

    actions_value = batch.get("actions")
    if isinstance(actions_value, torch.Tensor):
        expected_action_dim = int(actions_value.shape[-1])
    elif isinstance(actions_value, np.ndarray):
        expected_action_dim = int(actions_value.shape[-1])
    else:
        expected_action_dim = 7

    validate_kkt_batch_for_finetune(
        batch,
        expected_chunk_size=hook_config.kkt_chunk_size,
        expected_action_dim=expected_action_dim,
    )

    action_token_mask = build_action_token_mask_from_finetune_masks(current_action_mask, next_actions_mask)
    hidden_inputs = extract_kkt_head_inputs(
        output_hidden_states_last,
        action_token_mask,
        chunk_size=hook_config.kkt_chunk_size,
        current_index=0,
    )

    kkt_outputs = kkt_head(
        current_hidden=hidden_inputs["current_hidden"] if hook_config.use_current_kkt_loss else None,
        chunk_hidden=hidden_inputs["chunk_hidden"] if hook_config.use_chunk_kkt_loss else None,
    )

    loss_config = build_kkt_loss_config(hook_config)
    batch_torch = _to_tensor(batch, target_device)

    if predicted_actions is None:
        current_loss = hidden_inputs["current_hidden"].new_tensor(0.0)
        chunk_loss = hidden_inputs["current_hidden"].new_tensor(0.0)
        current_components = {}
        chunk_components = {}

        if hook_config.use_current_kkt_loss and kkt_outputs.get("kkt_current") is not None:
            current_components = compute_kkt_current_losses(
                kkt_outputs["kkt_current"],
                batch_torch["kkt_targets"]["current"],
                batch_torch["kkt_masks"],
                loss_config,
            )
            current_loss = current_components["total_current_kkt_loss"]

        if hook_config.use_chunk_kkt_loss and kkt_outputs.get("kkt_chunk") is not None:
            chunk_components = compute_kkt_chunk_losses(
                kkt_outputs["kkt_chunk"],
                batch_torch["kkt_targets"]["chunk"],
                batch_torch["kkt_masks"],
                loss_config,
            )
            chunk_loss = chunk_components["total_chunk_kkt_loss"]

        total_kkt_loss = current_loss + chunk_loss
    else:
        predictions = {
            "actions": predicted_actions,
            "kkt_current": kkt_outputs.get("kkt_current"),
            "kkt_chunk": kkt_outputs.get("kkt_chunk"),
        }
        total_components = compute_total_kkt_sense_loss(predictions, batch_torch, loss_config)
        total_kkt_loss = total_components["total_loss"]
        current_loss = total_components.get("current_total_current_kkt_loss", total_kkt_loss.new_tensor(0.0))
        chunk_loss = total_components.get("chunk_total_chunk_kkt_loss", total_kkt_loss.new_tensor(0.0))
        current_components = {}
        chunk_components = {}

    return {
        "kkt_loss": total_kkt_loss,
        "kkt_current_loss": current_loss,
        "kkt_chunk_loss": chunk_loss,
        "kkt_loss_components": {
            "current": current_components,
            "chunk": chunk_components,
        },
        "kkt_predictions": kkt_outputs,
        "kkt_debug": {
            "num_action_tokens": hidden_inputs["num_action_tokens"],
            "chunk_size": hook_config.kkt_chunk_size,
            "hidden_dim": hook_config.kkt_hidden_dim,
        },
    }
