"""
kkt_finetune_dataset.py

KKT finetune dataset and collator for OpenVLA-style KKT samples.
This module is opt-in and does not modify default RLDS training behavior.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import RLDSBatchTransform
from prismatic.vla.datasets.kkt_openvla_chunk_dataset import KKTChunkedOpenVLADataset


class KKTFinetuneDataset:
    def __init__(
        self,
        manifest_path: str,
        action_tokenizer: ActionTokenizer,
        tokenizer,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        resize_resolution: Optional[Tuple[int, int]] = None,
        require_kkt: bool = True,
        action_target: str = "safe",
        chunk_size: int = 8,
        load_images: bool = True,
        use_wrist_image: bool = False,
        use_proprio: bool = True,
        max_samples: Optional[int] = None,
        dataset_name: str = "kkt_openvla",
    ) -> None:
        self.dataset_name = dataset_name
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio
        self.load_images = load_images
        self.chunk_size = chunk_size

        self._chunked_dataset = KKTChunkedOpenVLADataset(
            manifest_path=manifest_path,
            require_kkt=require_kkt,
            action_target=action_target,
            chunk_size=chunk_size,
            load_images=load_images,
            max_samples=max_samples,
        )
        self._stats_dataset = KKTChunkedOpenVLADataset(
            manifest_path=manifest_path,
            require_kkt=require_kkt,
            action_target=action_target,
            chunk_size=chunk_size,
            load_images=False,
            max_samples=max_samples,
        )

        self._batch_transform = RLDSBatchTransform(
            action_tokenizer=action_tokenizer,
            base_tokenizer=tokenizer,
            image_transform=image_transform,
            prompt_builder_fn=prompt_builder_fn,
            use_wrist_image=use_wrist_image,
            use_proprio=use_proprio,
        )

        self.dataset_statistics = self._compute_dataset_statistics()
        self.resize_resolution = resize_resolution

    def __len__(self) -> int:
        return len(self._chunked_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._chunked_dataset[idx]

        agentview_image = sample.get("agentview_image")
        if agentview_image is None:
            raise ValueError("KKTFinetuneDataset requires agentview_image to be loaded")

        observation = {
            "image_primary": np.expand_dims(agentview_image, axis=0),
        }
        if self.use_wrist_image:
            wrist_image = sample.get("wrist_image")
            if wrist_image is None:
                raise ValueError("use_wrist_image=True but wrist_image is missing")
            observation["wrist"] = np.expand_dims(wrist_image, axis=0)

        if self.use_proprio:
            observation["proprio"] = sample.get("state")

        rlds_batch = {
            "dataset_name": self.dataset_name,
            "action": sample["actions"],
            "observation": observation,
            "task": {"language_instruction": sample.get("instruction", "").encode("utf-8")},
        }

        base_sample = self._batch_transform(rlds_batch)

        base_sample["actions"] = np.asarray(sample["actions"], dtype=np.float32)
        base_sample["action_chunk_mask"] = np.asarray(sample["action_chunk_mask"], dtype=np.float32)
        base_sample["kkt_targets"] = {
            "current": {
                "dual_cbf_main": np.asarray(sample["kkt_current"]["dual_cbf_main"], dtype=np.float32),
                "active_cbf_main": np.asarray(sample["kkt_current"]["active_cbf_main"], dtype=np.float32),
                "h": np.asarray(sample["kkt_current"]["h"], dtype=np.float32),
                "linear_cbf_lhs": np.asarray(sample["kkt_current"]["linear_cbf_lhs"], dtype=np.float32),
                "constraint_direction": np.asarray(sample["kkt_current"]["constraint_direction"], dtype=np.float32),
            },
            "chunk": {
                "dual_cbf_main": np.asarray(sample["kkt_chunk"]["dual_cbf_main"], dtype=np.float32),
                "active_cbf_main": np.asarray(sample["kkt_chunk"]["active_cbf_main"], dtype=np.float32),
                "h": np.asarray(sample["kkt_chunk"]["h"], dtype=np.float32),
                "linear_cbf_lhs": np.asarray(sample["kkt_chunk"]["linear_cbf_lhs"], dtype=np.float32),
                "constraint_direction": np.asarray(sample["kkt_chunk"]["constraint_direction"], dtype=np.float32),
            },
        }
        base_sample["kkt_masks"] = {
            "current_has_kkt": np.asarray(sample["kkt_current"]["has_kkt"], dtype=np.float32),
            "current_qp_valid": np.asarray(sample["kkt_current"]["qp_valid"], dtype=np.float32),
            "chunk_has_kkt": np.asarray(sample["kkt_chunk"]["has_kkt"], dtype=np.float32),
            "chunk_qp_valid": np.asarray(sample["kkt_chunk"]["qp_valid"], dtype=np.float32),
            "action_chunk_mask": np.asarray(sample["action_chunk_mask"], dtype=np.float32),
        }
        base_sample["metadata"] = sample.get("metadata", {})
        base_sample["raw"] = sample.get("raw", {})

        return base_sample

    def _compute_dataset_statistics(self) -> Dict[str, Any]:
        actions_list: List[np.ndarray] = []
        proprio_list: List[np.ndarray] = []
        for idx in range(len(self._stats_dataset)):
            sample = self._stats_dataset[idx]
            actions_list.append(np.asarray(sample["actions"], dtype=np.float32))
            if self.use_proprio:
                state = sample.get("state")
                if isinstance(state, np.ndarray) and state.size > 0:
                    proprio_list.append(state.astype(np.float32))

        if not actions_list:
            action_stats = {
                "mean": np.zeros((7,), dtype=np.float32),
                "std": np.ones((7,), dtype=np.float32),
                "max": np.zeros((7,), dtype=np.float32),
                "min": np.zeros((7,), dtype=np.float32),
                "q01": np.zeros((7,), dtype=np.float32),
                "q99": np.ones((7,), dtype=np.float32),
            }
            stats = {"action": action_stats, "num_transitions": 0, "num_trajectories": 0}
            return {self.dataset_name: stats}

        actions = np.concatenate(actions_list, axis=0)
        action_stats = {
            "mean": actions.mean(0),
            "std": actions.std(0),
            "max": actions.max(0),
            "min": actions.min(0),
            "q01": np.quantile(actions, 0.01, axis=0),
            "q99": np.quantile(actions, 0.99, axis=0),
        }

        stats = {
            "action": action_stats,
            "num_transitions": int(actions.shape[0]),
            "num_trajectories": int(len(self._stats_dataset)),
        }

        if self.use_proprio and proprio_list:
            proprio = np.stack(proprio_list, axis=0)
            stats["proprio"] = {
                "mean": proprio.mean(0),
                "std": proprio.std(0),
                "max": proprio.max(0),
                "min": proprio.min(0),
                "q01": np.quantile(proprio, 0.01, axis=0),
                "q99": np.quantile(proprio, 0.99, axis=0),
            }

        return {self.dataset_name: stats}


@dataclass
class KKTFinetuneCollator:
    base_collator: PaddedCollatorForActionPrediction

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        base_batch = self.base_collator(instances)

        action_chunk_mask = self._stack_field(instances, "action_chunk_mask")

        kkt_targets = {
            "current": self._stack_nested(instances, "kkt_targets", "current"),
            "chunk": self._stack_nested(instances, "kkt_targets", "chunk"),
        }
        kkt_masks = {
            "current_has_kkt": self._stack_nested(instances, "kkt_masks", "current_has_kkt"),
            "current_qp_valid": self._stack_nested(instances, "kkt_masks", "current_qp_valid"),
            "chunk_has_kkt": self._stack_nested(instances, "kkt_masks", "chunk_has_kkt"),
            "chunk_qp_valid": self._stack_nested(instances, "kkt_masks", "chunk_qp_valid"),
            "action_chunk_mask": action_chunk_mask,
        }

        base_batch["action_chunk_mask"] = action_chunk_mask
        base_batch["kkt_targets"] = kkt_targets
        base_batch["kkt_masks"] = kkt_masks
        base_batch["metadata"] = [instance.get("metadata", {}) for instance in instances]

        return base_batch

    def _stack_field(self, instances: List[Dict[str, Any]], key: str):
        arrays = [np.asarray(instance[key], dtype=np.float32) for instance in instances]
        first_shape = arrays[0].shape
        for arr in arrays:
            if arr.shape != first_shape:
                raise ValueError("Inconsistent shape for %s: %s != %s" % (key, arr.shape, first_shape))
        return self._to_tensor(np.stack(arrays, axis=0))

    def _stack_nested(self, instances, key, subkey):
        """Stack nested KKT fields.

        Supports both:
        - instance[key][subkey] = np.ndarray
        - instance[key][subkey] = dict[str, np.ndarray]

        The dict case is used by kkt_targets["current"] and
        kkt_targets["chunk"], which contain multiple target fields.
        """
        values = [instance[key][subkey] for instance in instances]
        if not values:
            raise ValueError("Cannot stack empty nested values for %s.%s" % (key, subkey))

        first = values[0]

        if isinstance(first, dict):
            output = {}
            for field_name in first.keys():
                arrays = []
                for idx, value in enumerate(values):
                    if not isinstance(value, dict):
                        raise ValueError(
                            "Mixed nested value types for %s.%s at index %d" % (key, subkey, idx)
                        )
                    if field_name not in value:
                        raise ValueError(
                            "Missing nested field %s.%s.%s at index %d"
                            % (key, subkey, field_name, idx)
                        )
                    arrays.append(np.asarray(value[field_name], dtype=np.float32))

                shapes = [arr.shape for arr in arrays]
                if len(set(shapes)) != 1:
                    raise ValueError(
                        "Inconsistent shapes for %s.%s.%s: %s"
                        % (key, subkey, field_name, shapes)
                    )

                output[field_name] = torch.from_numpy(np.stack(arrays, axis=0))
            return output

        arrays = [np.asarray(value, dtype=np.float32) for value in values]
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) != 1:
            raise ValueError("Inconsistent shapes for %s.%s: %s" % (key, subkey, shapes))

        return torch.from_numpy(np.stack(arrays, axis=0))
    def _to_tensor(self, array: np.ndarray):
        return torch.from_numpy(np.asarray(array, dtype=np.float32))
