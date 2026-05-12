"""
kkt_openvla_dataset.py

Adapter for OpenVLA-style KKT samples exported by safetydistill.
This dataset is read-only and does not depend on torch or training code.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _safe_float_array(value: Any, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    if value is None:
        if shape is None:
            return np.array([], dtype=np.float32)
        return np.zeros(shape, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if shape is not None:
        try:
            arr = arr.reshape(shape)
        except Exception:
            raise ValueError("Cannot reshape array to expected shape: %s" % (shape,))
    return arr


def _resolve_path(base_dir: str, maybe_path: Optional[str]) -> Optional[str]:
    if not maybe_path:
        return None
    if os.path.isabs(maybe_path):
        return maybe_path
    return os.path.join(base_dir, maybe_path)


def _load_image(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    if Image is None:
        raise ImportError("PIL is required to load images; install pillow or set load_images=False.")
    if not os.path.isfile(path):
        return None
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.asarray(img)


class KKTOpenVLASampleDataset:
    def __init__(
        self,
        manifest_path: str,
        require_kkt: bool = True,
        action_target: str = "safe",
        load_images: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        if action_target not in ["safe", "delta", "nominal"]:
            raise ValueError("Invalid action_target: %s" % action_target)

        self.manifest_path = manifest_path
        self.require_kkt = require_kkt
        self.action_target = action_target
        self.load_images = load_images
        self.max_samples = max_samples

        self._records: List[Dict[str, Any]] = []
        self._load_manifest()

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self._records[idx]

        episode_dir = record.get("_episode_dir", "")
        agentview_path = _resolve_path(episode_dir, record.get("agentview_image") or record.get("agentview_image_path"))
        wrist_path = _resolve_path(episode_dir, record.get("wrist_image") or record.get("wrist_image_path"))

        agentview_image = _load_image(agentview_path) if self.load_images else None
        wrist_image = _load_image(wrist_path) if self.load_images else None

        action_nominal = _safe_float_array(record.get("action_nominal"), shape=(7,))
        action_safe = _safe_float_array(record.get("action_safe"), shape=(7,))
        action_delta = _safe_float_array(record.get("action_delta"), shape=(7,))

        if self.action_target == "safe":
            action = action_safe
        elif self.action_target == "delta":
            action = action_delta
        else:
            action = action_nominal

        dual_vars = record.get("dual_variables") or {}
        active_set = record.get("active_set") or {}
        constraint_values = record.get("constraint_values") or {}
        constraint_gradients = record.get("constraint_gradients") or {}

        dual_cbf_main = _safe_float_array(dual_vars.get("cbf_main"), shape=(1,))
        active_cbf_main = _safe_float_array(
            1.0 if active_set.get("cbf_main") else 0.0,
            shape=(1,),
        )
        h = _safe_float_array(constraint_values.get("h"), shape=(1,))
        linear_cbf_lhs = _safe_float_array(constraint_values.get("linear_cbf_lhs"), shape=(1,))

        a_u_v = _safe_float_array(constraint_gradients.get("a_u_v"), shape=(3,))
        a_uz = _safe_float_array(constraint_gradients.get("a_uz"), shape=(3,))
        constraint_direction = np.concatenate([a_u_v, a_uz]).astype(np.float32)

        qp_status = record.get("qp_status")
        qp_valid = 1.0 if (qp_status is not None and qp_status != "no_safety_control") else 0.0
        has_kkt = 1.0 if self._has_kkt_fields(record) else 0.0

        state_value = record.get("state")
        state_len = len(state_value) if isinstance(state_value, (list, tuple)) else 0
        state = _safe_float_array(state_value, shape=(state_len,))

        return {
            "instruction": record.get("instruction", ""),
            "agentview_image": agentview_image,
            "wrist_image": wrist_image,
            "agentview_image_path": agentview_path or "",
            "wrist_image_path": wrist_path or "",
            "state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "action_nominal": action_nominal,
            "action_safe": action_safe,
            "action_delta": action_delta,
            "dual_cbf_main": dual_cbf_main,
            "active_cbf_main": active_cbf_main,
            "h": h,
            "linear_cbf_lhs": linear_cbf_lhs,
            "constraint_direction": constraint_direction,
            "masks": {
                "has_kkt": _safe_float_array(has_kkt, shape=(1,)),
                "qp_valid": _safe_float_array(qp_valid, shape=(1,)),
            },
            "metadata": {
                "task_suite_name": record.get("task_suite_name", ""),
                "safety_level": record.get("safety_level", ""),
                "task_index": int(record.get("task_index", -1)) if record.get("task_index") is not None else -1,
                "episode_index": int(record.get("episode_index", -1)) if record.get("episode_index") is not None else -1,
                "step_index": int(record.get("step_index", -1)) if record.get("step_index") is not None else -1,
                "qp_status": qp_status,
            },
            "raw": record,
        }

    def _load_manifest(self) -> None:
        if not os.path.isfile(self.manifest_path):
            return

        with open(self.manifest_path, "r") as f:
            try:
                manifest = json.load(f)
            except json.JSONDecodeError:
                return

        episodes = manifest.get("episodes", [])
        if not episodes:
            return

        manifest_dir = os.path.dirname(os.path.abspath(self.manifest_path))

        for episode_entry in episodes:
            episode_dir, steps_path = self._resolve_episode(episode_entry, manifest_dir)
            if not steps_path or not os.path.isfile(steps_path):
                continue

            self._index_steps(episode_dir, steps_path)
            if self.max_samples is not None and len(self._records) >= self.max_samples:
                break

    def _resolve_episode(self, episode_entry: Any, manifest_dir: str) -> Tuple[str, Optional[str]]:
        if isinstance(episode_entry, str):
            episode_dir = _resolve_path(manifest_dir, episode_entry)
            steps_path = _resolve_path(episode_dir, "steps.jsonl") if episode_dir else None
            return episode_dir or "", steps_path

        if isinstance(episode_entry, dict):
            episode_dir = (
                episode_entry.get("episode_dir")
                or episode_entry.get("path")
                or episode_entry.get("dir")
            )
            episode_dir = _resolve_path(manifest_dir, episode_dir) if episode_dir else None
            steps_path = episode_entry.get("steps_path") or episode_entry.get("steps_jsonl")
            if not steps_path and episode_dir:
                steps_path = os.path.join(episode_dir, "steps.jsonl")
            if steps_path:
                steps_path = _resolve_path(manifest_dir, steps_path)
            return episode_dir or "", steps_path

        return "", None

    def _index_steps(self, episode_dir: str, steps_path: str) -> None:
        try:
            with open(steps_path, "r") as f:
                for line in f:
                    if self.max_samples is not None and len(self._records) >= self.max_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    record["_episode_dir"] = episode_dir

                    if self.require_kkt and not self._has_kkt_fields(record):
                        continue

                    self._records.append(record)
        except OSError:
            return

    def _has_kkt_fields(self, record: Dict[str, Any]) -> bool:
        required = [
            "dual_variables",
            "active_set",
            "constraint_values",
            "constraint_gradients",
        ]
        for field in required:
            value = record.get(field)
            if value is None or value == {}:
                return False
        return True


def collate_kkt_openvla_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate empty samples list.")

    instructions = [s["instruction"] for s in samples]
    metadata = [s["metadata"] for s in samples]
    raw = [s["raw"] for s in samples]

    def stack_array(field: str, expected_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        arrays = []
        for s in samples:
            arr = s[field]
            if expected_shape is not None and arr.shape != expected_shape:
                raise ValueError("Inconsistent shape for %s: %s != %s" % (field, arr.shape, expected_shape))
            arrays.append(arr)
        return np.stack(arrays, axis=0)

    def stack_images(field: str) -> Optional[np.ndarray]:
        images = [s[field] for s in samples]
        if all(img is None for img in images):
            return None
        if any(img is None for img in images):
            raise ValueError("Mixed None and image arrays for %s." % field)
        shapes = [img.shape for img in images]
        if len(set(shapes)) != 1:
            raise ValueError("Inconsistent image shapes for %s: %s" % (field, shapes))
        return np.stack(images, axis=0)

    agentview_images = stack_images("agentview_image")
    wrist_images = stack_images("wrist_image")

    state_shape = samples[0]["state"].shape
    states = stack_array("state", expected_shape=state_shape)
    actions = stack_array("action", expected_shape=(7,))
    action_nominal = stack_array("action_nominal", expected_shape=(7,))
    action_safe = stack_array("action_safe", expected_shape=(7,))
    action_delta = stack_array("action_delta", expected_shape=(7,))

    dual_cbf_main = stack_array("dual_cbf_main", expected_shape=(1,))
    active_cbf_main = stack_array("active_cbf_main", expected_shape=(1,))
    h = stack_array("h", expected_shape=(1,))
    linear_cbf_lhs = stack_array("linear_cbf_lhs", expected_shape=(1,))
    constraint_direction = stack_array("constraint_direction", expected_shape=(6,))

    masks = {
        "has_kkt": np.stack([s["masks"]["has_kkt"] for s in samples], axis=0),
        "qp_valid": np.stack([s["masks"]["qp_valid"] for s in samples], axis=0),
    }

    return {
        "instructions": instructions,
        "agentview_images": agentview_images,
        "wrist_images": wrist_images,
        "states": states,
        "actions": actions,
        "action_nominal": action_nominal,
        "action_safe": action_safe,
        "action_delta": action_delta,
        "dual_cbf_main": dual_cbf_main,
        "active_cbf_main": active_cbf_main,
        "h": h,
        "linear_cbf_lhs": linear_cbf_lhs,
        "constraint_direction": constraint_direction,
        "masks": masks,
        "metadata": metadata,
        "raw": raw,
    }
