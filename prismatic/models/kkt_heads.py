"""
kkt_heads.py

Auxiliary heads for KKT-SenseVLA predictions.
These heads are independent of OpenVLA forward and support CPU dry-run.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class KKTHeadConfig:
    hidden_dim: int
    chunk_size: int = 8
    direction_dim: int = 6
    dropout: float = 0.0
    use_layernorm: bool = True
    predict_current: bool = True
    predict_chunk: bool = True


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, dropout: float, use_layernorm: bool) -> None:
        super().__init__()
        self.use_layernorm = use_layernorm
        self.layernorm = nn.LayerNorm(hidden_dim) if use_layernorm else None
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layernorm:
            x = self.layernorm(x)
        return self.net(x)


class KKTCurrentHead(nn.Module):
    def __init__(self, hidden_dim: int, direction_dim: int, dropout: float, use_layernorm: bool) -> None:
        super().__init__()
        self.dual = MLP(hidden_dim, 1, dropout, use_layernorm)
        self.active = MLP(hidden_dim, 1, dropout, use_layernorm)
        self.h = MLP(hidden_dim, 1, dropout, use_layernorm)
        self.direction = MLP(hidden_dim, direction_dim, dropout, use_layernorm)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "dual_cbf_main": self.dual(hidden),
            "active_cbf_main_logit": self.active(hidden),
            "h": self.h(hidden),
            "constraint_direction": self.direction(hidden),
        }


class KKTChunkHead(nn.Module):
    def __init__(self, hidden_dim: int, direction_dim: int, dropout: float, use_layernorm: bool) -> None:
        super().__init__()
        self.dual = MLP(hidden_dim, 1, dropout, use_layernorm)
        self.active = MLP(hidden_dim, 1, dropout, use_layernorm)
        self.h = MLP(hidden_dim, 1, dropout, use_layernorm)
        self.direction = MLP(hidden_dim, direction_dim, dropout, use_layernorm)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "dual_cbf_main": self.dual(hidden),
            "active_cbf_main_logit": self.active(hidden),
            "h": self.h(hidden),
            "constraint_direction": self.direction(hidden),
        }


class KKTMultiHead(nn.Module):
    def __init__(self, config: KKTHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.current_head: Optional[KKTCurrentHead] = None
        self.chunk_head: Optional[KKTChunkHead] = None

        if config.predict_current:
            self.current_head = KKTCurrentHead(
                hidden_dim=config.hidden_dim,
                direction_dim=config.direction_dim,
                dropout=config.dropout,
                use_layernorm=config.use_layernorm,
            )
        if config.predict_chunk:
            self.chunk_head = KKTChunkHead(
                hidden_dim=config.hidden_dim,
                direction_dim=config.direction_dim,
                dropout=config.dropout,
                use_layernorm=config.use_layernorm,
            )

    def forward(
        self,
        current_hidden: Optional[torch.Tensor] = None,
        chunk_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[Dict[str, torch.Tensor]]]:
        outputs = {
            "kkt_current": None,
            "kkt_chunk": None,
        }

        if self.current_head is not None and current_hidden is not None:
            outputs["kkt_current"] = self.current_head(current_hidden)

        if self.chunk_head is not None and chunk_hidden is not None:
            outputs["kkt_chunk"] = self.chunk_head(chunk_hidden)

        return outputs
