"""
kkt_openvla_chunk_dataset.py

Chunked adapter for OpenVLA-style KKT samples exported by safetydistill.
This dataset is read-only and does not depend on torch or training code.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prismatic.vla.datasets.kkt_openvla_dataset import _load_image, _resolve_path, _safe_float_array


def _has_kkt_fields(record: Dict[str, Any]) -> bool:
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


def _extract_kkt_fields(record: Dict[str, Any]) -> Dict[str, np.ndarray]:
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
    has_kkt = 1.0 if _has_kkt_fields(record) else 0.0

    return {
        "dual_cbf_main": dual_cbf_main,
        "active_cbf_main": active_cbf_main,
        "h": h,
        "linear_cbf_lhs": linear_cbf_lhs,
        "constraint_direction": constraint_direction,
        "has_kkt": _safe_float_array(has_kkt, shape=(1,)),
        "qp_valid": _safe_float_array(qp_valid, shape=(1,)),
        "qp_status": qp_status,
    }


class KKTChunkedOpenVLADataset:
    def __init__(
        self,
        manifest_path: str,
        require_kkt: bool = True,
        action_target: str = "safe",
        chunk_size: int = 8,
        load_images: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        if action_target not in ["safe", "delta", "nominal"]:
            raise ValueError("Invalid action_target: %s" % action_target)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.manifest_path = manifest_path
        self.require_kkt = require_kkt
        self.action_target = action_target
        self.chunk_size = chunk_size
        self.load_images = load_images
        self.max_samples = max_samples

        self._episodes: List[Dict[str, Any]] = []
        self._index: List[Tuple[int, int]] = []
        self._load_manifest()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode_idx, step_idx = self._index[idx]
        episode = self._episodes[episode_idx]
        steps = episode["steps"]
        current = steps[step_idx]

        agentview_path = _resolve_path(episode["episode_dir"], current.get("agentview_image") or current.get("agentview_image_path"))
        wrist_path = _resolve_path(episode["episode_dir"], current.get("wrist_image") or current.get("wrist_image_path"))

        agentview_image = _load_image(agentview_path) if self.load_images else None
        wrist_image = _load_image(wrist_path) if self.load_images else None

        state_value = current.get("state")
        state_len = len(state_value) if isinstance(state_value, (list, tuple)) else 0
        state = _safe_float_array(state_value, shape=(state_len,)).astype(np.float32)

        action_nominal_chunk, action_safe_chunk, action_delta_chunk, action_mask = self._build_action_chunk(steps, step_idx)
        if self.action_target == "safe":
            actions = action_safe_chunk
        elif self.action_target == "delta":
            actions = action_delta_chunk
        else:
            actions = action_nominal_chunk

        kkt_current = self._build_kkt_current(current)
        kkt_chunk = self._build_kkt_chunk(steps, step_idx, action_mask)

        metadata = {
            "task_suite_name": current.get("task_suite_name", ""),
            "safety_level": current.get("safety_level", ""),
            "task_index": int(current.get("task_index", -1)) if current.get("task_index") is not None else -1,
            "episode_index": int(current.get("episode_index", -1)) if current.get("episode_index") is not None else -1,
            "step_index": int(current.get("step_index", -1)) if current.get("step_index") is not None else -1,
            "chunk_size": self.chunk_size,
            "num_valid_actions": int(action_mask.sum()),
            "qp_status": current.get("qp_status"),
        }

        return {
            "instruction": current.get("instruction", ""),
            "agentview_image": agentview_image,
            "wrist_image": wrist_image,
            "agentview_image_path": agentview_path or "",
            "wrist_image_path": wrist_path or "",
            "state": state,
            "actions": actions,
            "action_nominal": action_nominal_chunk,
            "action_safe": action_safe_chunk,
            "action_delta": action_delta_chunk,
            "action_chunk_mask": action_mask,
            "kkt_current": kkt_current,
            "kkt_chunk": kkt_chunk,
            "metadata": metadata,
            "raw": current,
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

            steps = self._load_steps(steps_path)
            if not steps:
                continue

            steps = self._sort_steps(steps)
            episode_idx = len(self._episodes)
            self._episodes.append({"episode_dir": episode_dir, "steps": steps})

            for i, step in enumerate(steps):
                if self.require_kkt and not _has_kkt_fields(step):
                    continue
                self._index.append((episode_idx, i))
                if self.max_samples is not None and len(self._index) >= self.max_samples:
                    return

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

    def _load_steps(self, steps_path: str) -> List[Dict[str, Any]]:
        steps = []
        try:
            with open(steps_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    steps.append(record)
        except OSError:
            return []
        return steps

    def _sort_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with_index = []
        for idx, step in enumerate(steps):
            step_index = step.get("step_index")
            sort_key = step_index if isinstance(step_index, int) else idx
            with_index.append((sort_key, idx, step))
        with_index.sort(key=lambda x: (x[0], x[1]))
        return [item[2] for item in with_index]

    def _build_action_chunk(
        self, steps: List[Dict[str, Any]], start_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        action_nominal = np.zeros((self.chunk_size, 7), dtype=np.float32)
        action_safe = np.zeros((self.chunk_size, 7), dtype=np.float32)
        action_delta = np.zeros((self.chunk_size, 7), dtype=np.float32)
        action_mask = np.zeros((self.chunk_size,), dtype=np.float32)

        for offset in range(self.chunk_size):
            idx = start_idx + offset
            if idx >= len(steps):
                break
            step = steps[idx]
            action_nominal[offset] = _safe_float_array(step.get("action_nominal"), shape=(7,))
            action_safe[offset] = _safe_float_array(step.get("action_safe"), shape=(7,))
            action_delta[offset] = _safe_float_array(step.get("action_delta"), shape=(7,))
            action_mask[offset] = 1.0

        return action_nominal, action_safe, action_delta, action_mask

    def _build_kkt_current(self, current: Dict[str, Any]) -> Dict[str, np.ndarray]:
        kkt = _extract_kkt_fields(current)
        return {
            "dual_cbf_main": kkt["dual_cbf_main"],
            "active_cbf_main": kkt["active_cbf_main"],
            "h": kkt["h"],
            "linear_cbf_lhs": kkt["linear_cbf_lhs"],
            "constraint_direction": kkt["constraint_direction"],
            "has_kkt": kkt["has_kkt"],
            "qp_valid": kkt["qp_valid"],
        }

    def _build_kkt_chunk(self, steps: List[Dict[str, Any]], start_idx: int, action_mask: np.ndarray) -> Dict[str, np.ndarray]:
        dual_cbf_main = np.zeros((self.chunk_size, 1), dtype=np.float32)
        active_cbf_main = np.zeros((self.chunk_size, 1), dtype=np.float32)
        h = np.zeros((self.chunk_size, 1), dtype=np.float32)
        linear_cbf_lhs = np.zeros((self.chunk_size, 1), dtype=np.float32)
        constraint_direction = np.zeros((self.chunk_size, 6), dtype=np.float32)
        has_kkt = np.zeros((self.chunk_size, 1), dtype=np.float32)
        qp_valid = np.zeros((self.chunk_size, 1), dtype=np.float32)

        for offset in range(self.chunk_size):
            idx = start_idx + offset
            if idx >= len(steps):
                break
            if action_mask[offset] == 0.0:
                continue

            step = steps[idx]
            kkt = _extract_kkt_fields(step)
            dual_cbf_main[offset] = kkt["dual_cbf_main"]
            active_cbf_main[offset] = kkt["active_cbf_main"]
            h[offset] = kkt["h"]
            linear_cbf_lhs[offset] = kkt["linear_cbf_lhs"]
            constraint_direction[offset] = kkt["constraint_direction"]
            has_kkt[offset] = kkt["has_kkt"]
            qp_valid[offset] = kkt["qp_valid"]

            if action_mask[offset] == 0.0:
                has_kkt[offset] = 0.0
                qp_valid[offset] = 0.0

        return {
            "dual_cbf_main": dual_cbf_main,
            "active_cbf_main": active_cbf_main,
            "h": h,
            "linear_cbf_lhs": linear_cbf_lhs,
            "constraint_direction": constraint_direction,
            "has_kkt": has_kkt,
            "qp_valid": qp_valid,
        }


def collate_kkt_openvla_chunk_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    actions = stack_array("actions", expected_shape=samples[0]["actions"].shape)
    action_nominal = stack_array("action_nominal", expected_shape=samples[0]["action_nominal"].shape)
    action_safe = stack_array("action_safe", expected_shape=samples[0]["action_safe"].shape)
    action_delta = stack_array("action_delta", expected_shape=samples[0]["action_delta"].shape)
    action_chunk_mask = stack_array("action_chunk_mask", expected_shape=samples[0]["action_chunk_mask"].shape)

    def stack_kkt(field: str) -> Dict[str, np.ndarray]:
        first = samples[0][field]
        return {
            "dual_cbf_main": stack_array(field + ".dual_cbf_main", expected_shape=first["dual_cbf_main"].shape),
        }

    kkt_current = {
        "dual_cbf_main": np.stack([s["kkt_current"]["dual_cbf_main"] for s in samples], axis=0),
        "active_cbf_main": np.stack([s["kkt_current"]["active_cbf_main"] for s in samples], axis=0),
        "h": np.stack([s["kkt_current"]["h"] for s in samples], axis=0),
        "linear_cbf_lhs": np.stack([s["kkt_current"]["linear_cbf_lhs"] for s in samples], axis=0),
        "constraint_direction": np.stack([s["kkt_current"]["constraint_direction"] for s in samples], axis=0),
        "has_kkt": np.stack([s["kkt_current"]["has_kkt"] for s in samples], axis=0),
        "qp_valid": np.stack([s["kkt_current"]["qp_valid"] for s in samples], axis=0),
    }

    kkt_chunk = {
        "dual_cbf_main": np.stack([s["kkt_chunk"]["dual_cbf_main"] for s in samples], axis=0),
        "active_cbf_main": np.stack([s["kkt_chunk"]["active_cbf_main"] for s in samples], axis=0),
        "h": np.stack([s["kkt_chunk"]["h"] for s in samples], axis=0),
        "linear_cbf_lhs": np.stack([s["kkt_chunk"]["linear_cbf_lhs"] for s in samples], axis=0),
        "constraint_direction": np.stack([s["kkt_chunk"]["constraint_direction"] for s in samples], axis=0),
        "has_kkt": np.stack([s["kkt_chunk"]["has_kkt"] for s in samples], axis=0),
        "qp_valid": np.stack([s["kkt_chunk"]["qp_valid"] for s in samples], axis=0),
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
        "action_chunk_mask": action_chunk_mask,
        "kkt_current": kkt_current,
        "kkt_chunk": kkt_chunk,
        "metadata": metadata,
        "raw": raw,
    }
