"""
kkt_losses.py

Loss functions for KKT-SenseVLA dry-run inspections.
These losses are CPU-safe and do not call model forward.
"""

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F


@dataclass
class KKTTrainingLossConfig:
    action_loss_weight: float = 1.0
    dual_loss_weight: float = 0.1
    active_loss_weight: float = 0.1
    h_loss_weight: float = 0.05
    direction_loss_weight: float = 0.05
    use_chunk_kkt_loss: bool = True
    use_current_kkt_loss: bool = True
    active_loss_type: str = "bce"
    direction_loss_type: str = "cosine"
    eps: float = 1e-6


def masked_mean(loss: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    masked = loss * mask
    return masked.sum() / (mask.sum() + eps)


def compute_action_chunk_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    action_chunk_mask: torch.Tensor,
) -> torch.Tensor:
    if pred_actions.shape != target_actions.shape:
        raise ValueError("pred_actions and target_actions must have same shape")
    if action_chunk_mask.dim() != 2:
        raise ValueError("action_chunk_mask must have shape (B, T)")

    mask = action_chunk_mask.unsqueeze(-1)
    l1 = torch.abs(pred_actions - target_actions)
    return masked_mean(l1, mask)


def compute_kkt_current_losses(
    pred_current: Dict[str, torch.Tensor],
    target_current: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    config: KKTTrainingLossConfig,
) -> Dict[str, torch.Tensor]:
    mask = masks["current_has_kkt"] * masks["current_qp_valid"]

    dual_loss = torch.tensor(0.0)
    active_loss = torch.tensor(0.0)
    h_loss = torch.tensor(0.0)
    direction_loss = torch.tensor(0.0)

    if pred_current.get("dual_cbf_main") is not None:
        dual_l1 = torch.abs(pred_current["dual_cbf_main"] - target_current["dual_cbf_main"])
        dual_loss = masked_mean(dual_l1, mask, config.eps)

    if pred_current.get("active_cbf_main_logit") is not None:
        if config.active_loss_type != "bce":
            raise ValueError("active_loss_type only supports bce")
        bce = F.binary_cross_entropy_with_logits(
            pred_current["active_cbf_main_logit"],
            target_current["active_cbf_main"],
            reduction="none",
        )
        active_loss = masked_mean(bce, mask, config.eps)

    if pred_current.get("h") is not None:
        h_l1 = torch.abs(pred_current["h"] - target_current["h"])
        h_loss = masked_mean(h_l1, mask, config.eps)

    if pred_current.get("constraint_direction") is not None:
        if config.direction_loss_type == "cosine":
            cos = F.cosine_similarity(
                pred_current["constraint_direction"],
                target_current["constraint_direction"],
                dim=-1,
            )
            dir_loss = 1.0 - cos
        elif config.direction_loss_type == "mse":
            diff = pred_current["constraint_direction"] - target_current["constraint_direction"]
            dir_loss = (diff * diff).mean(dim=-1)
        else:
            raise ValueError("direction_loss_type must be cosine or mse")
        direction_loss = masked_mean(dir_loss.unsqueeze(-1), mask, config.eps)

    total = (
        dual_loss * config.dual_loss_weight
        + active_loss * config.active_loss_weight
        + h_loss * config.h_loss_weight
        + direction_loss * config.direction_loss_weight
    )

    return {
        "dual_loss": dual_loss,
        "active_loss": active_loss,
        "h_loss": h_loss,
        "direction_loss": direction_loss,
        "total_current_kkt_loss": total,
    }


def compute_kkt_chunk_losses(
    pred_chunk: Dict[str, torch.Tensor],
    target_chunk: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    config: KKTTrainingLossConfig,
) -> Dict[str, torch.Tensor]:
    chunk_mask = masks["chunk_has_kkt"] * masks["chunk_qp_valid"] * masks["action_chunk_mask"].unsqueeze(-1)

    dual_loss = torch.tensor(0.0)
    active_loss = torch.tensor(0.0)
    h_loss = torch.tensor(0.0)
    direction_loss = torch.tensor(0.0)

    if pred_chunk.get("dual_cbf_main") is not None:
        dual_l1 = torch.abs(pred_chunk["dual_cbf_main"] - target_chunk["dual_cbf_main"])
        dual_loss = masked_mean(dual_l1, chunk_mask, config.eps)

    if pred_chunk.get("active_cbf_main_logit") is not None:
        if config.active_loss_type != "bce":
            raise ValueError("active_loss_type only supports bce")
        bce = F.binary_cross_entropy_with_logits(
            pred_chunk["active_cbf_main_logit"],
            target_chunk["active_cbf_main"],
            reduction="none",
        )
        active_loss = masked_mean(bce, chunk_mask, config.eps)

    if pred_chunk.get("h") is not None:
        h_l1 = torch.abs(pred_chunk["h"] - target_chunk["h"])
        h_loss = masked_mean(h_l1, chunk_mask, config.eps)

    if pred_chunk.get("constraint_direction") is not None:
        if config.direction_loss_type == "cosine":
            cos = F.cosine_similarity(
                pred_chunk["constraint_direction"],
                target_chunk["constraint_direction"],
                dim=-1,
            )
            dir_loss = 1.0 - cos
        elif config.direction_loss_type == "mse":
            diff = pred_chunk["constraint_direction"] - target_chunk["constraint_direction"]
            dir_loss = (diff * diff).mean(dim=-1)
        else:
            raise ValueError("direction_loss_type must be cosine or mse")
        direction_loss = masked_mean(dir_loss.unsqueeze(-1), chunk_mask, config.eps)

    total = (
        dual_loss * config.dual_loss_weight
        + active_loss * config.active_loss_weight
        + h_loss * config.h_loss_weight
        + direction_loss * config.direction_loss_weight
    )

    return {
        "dual_loss": dual_loss,
        "active_loss": active_loss,
        "h_loss": h_loss,
        "direction_loss": direction_loss,
        "total_chunk_kkt_loss": total,
    }


def compute_total_kkt_sense_loss(
    predictions: Dict[str, Any],
    batch: Dict[str, Any],
    config: KKTTrainingLossConfig,
) -> Dict[str, torch.Tensor]:
    if "actions" not in predictions:
        raise ValueError("predictions must include actions")
    if "actions" not in batch:
        raise ValueError("batch must include actions")
    if "action_chunk_mask" not in batch:
        raise ValueError("batch must include action_chunk_mask")

    pred_actions = predictions["actions"]
    target_actions = batch["actions"]
    action_chunk_mask = batch["action_chunk_mask"]

    action_loss = compute_action_chunk_loss(pred_actions, target_actions, action_chunk_mask)

    current_kkt_loss = torch.tensor(0.0)
    chunk_kkt_loss = torch.tensor(0.0)
    current_components = {}
    chunk_components = {}

    if config.use_current_kkt_loss and predictions.get("kkt_current") is not None:
        current_components = compute_kkt_current_losses(
            predictions["kkt_current"],
            batch["kkt_targets"]["current"],
            batch["kkt_masks"],
            config,
        )
        current_kkt_loss = current_components["total_current_kkt_loss"]

    if config.use_chunk_kkt_loss and predictions.get("kkt_chunk") is not None:
        chunk_components = compute_kkt_chunk_losses(
            predictions["kkt_chunk"],
            batch["kkt_targets"]["chunk"],
            batch["kkt_masks"],
            config,
        )
        chunk_kkt_loss = chunk_components["total_chunk_kkt_loss"]

    total_loss = (
        action_loss * config.action_loss_weight
        + current_kkt_loss
        + chunk_kkt_loss
    )

    output = {
        "action_loss": action_loss,
        "current_kkt_loss": current_kkt_loss,
        "chunk_kkt_loss": chunk_kkt_loss,
        "total_loss": total_loss,
    }
    output.update({"current_" + k: v for k, v in current_components.items()})
    output.update({"chunk_" + k: v for k, v in chunk_components.items()})
    return output
