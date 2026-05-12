"""
kkt_forward_bridge.py

Standalone forward bridge for KKT-SenseVLA dry-run.
This module does not call model forward or perform training.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from prismatic.models.kkt_heads import KKTHeadConfig, KKTMultiHead
from prismatic.models.kkt_hidden_extract import extract_kkt_head_inputs
from prismatic.training.kkt_losses import KKTTrainingLossConfig, compute_total_kkt_sense_loss


@dataclass
class KKTForwardBridgeConfig:
    hidden_dim: int
    action_dim: int = 7
    chunk_size: int = 8
    direction_dim: int = 6
    dropout: float = 0.0
    use_layernorm: bool = True
    use_current_kkt: bool = True
    use_chunk_kkt: bool = True
    action_head_type: str = "linear_mlp"


class LightweightActionChunkHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        dropout: float,
        use_layernorm: bool,
    ) -> None:
        super().__init__()
        self.use_layernorm = use_layernorm
        self.layernorm = nn.LayerNorm(hidden_dim) if use_layernorm else None
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, chunk_hidden: torch.Tensor) -> torch.Tensor:
        if self.use_layernorm:
            chunk_hidden = self.layernorm(chunk_hidden)
        return self.net(chunk_hidden)


class KKTForwardBridge(nn.Module):
    def __init__(self, config: KKTForwardBridgeConfig) -> None:
        super().__init__()
        if config.action_head_type != "linear_mlp":
            raise ValueError("action_head_type only supports linear_mlp")
        self.config = config
        self.action_head = LightweightActionChunkHead(
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            dropout=config.dropout,
            use_layernorm=config.use_layernorm,
        )
        self.kkt_head = KKTMultiHead(
            KKTHeadConfig(
                hidden_dim=config.hidden_dim,
                chunk_size=config.chunk_size,
                direction_dim=config.direction_dim,
                dropout=config.dropout,
                use_layernorm=config.use_layernorm,
                predict_current=config.use_current_kkt,
                predict_chunk=config.use_chunk_kkt,
            )
        )

    def forward(
        self,
        last_hidden_states: torch.Tensor,
        action_token_mask: torch.Tensor,
    ) -> Dict[str, Dict[str, Any]]:
        hidden_inputs = extract_kkt_head_inputs(
            last_hidden_states,
            action_token_mask,
            chunk_size=self.config.chunk_size,
            current_index=0,
        )
        current_hidden = hidden_inputs["current_hidden"]
        chunk_hidden = hidden_inputs["chunk_hidden"]

        pred_actions = self.action_head(chunk_hidden)
        kkt_outputs = self.kkt_head(
            current_hidden=current_hidden if self.config.use_current_kkt else None,
            chunk_hidden=chunk_hidden if self.config.use_chunk_kkt else None,
        )

        return {
            "predictions": {
                "actions": pred_actions,
                "kkt_current": kkt_outputs.get("kkt_current"),
                "kkt_chunk": kkt_outputs.get("kkt_chunk"),
            },
            "hidden": {
                "action_hidden": hidden_inputs["action_hidden"],
                "current_hidden": current_hidden,
                "chunk_hidden": chunk_hidden,
            },
            "debug": {
                "num_action_tokens": hidden_inputs["num_action_tokens"],
                "chunk_size": self.config.chunk_size,
                "hidden_dim": self.config.hidden_dim,
            },
        }


def _to_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        return {k: _to_tensor(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_tensor(v) for v in value]
    return value


def _ensure_tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    raise TypeError("%s must be a torch.Tensor or numpy.ndarray" % name)


def compute_kkt_forward_bridge_loss(
    bridge: KKTForwardBridge,
    last_hidden_states: Any,
    action_token_mask: Any,
    batch: Dict[str, Any],
    loss_config: KKTTrainingLossConfig,
) -> Dict[str, Any]:
    last_hidden_states = _ensure_tensor(last_hidden_states, "last_hidden_states")
    action_token_mask = _ensure_tensor(action_token_mask, "action_token_mask")

    bridge_output = bridge(last_hidden_states, action_token_mask)
    batch_torch = _to_tensor(batch)

    losses = compute_total_kkt_sense_loss(
        bridge_output["predictions"],
        batch_torch,
        loss_config,
    )

    return {
        "bridge_output": bridge_output,
        "losses": losses,
    }
