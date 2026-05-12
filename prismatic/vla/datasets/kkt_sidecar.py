"""
kkt_sidecar.py

Lightweight adapter for reading KKT sidecar JSONL files listed in a manifest.
This module is intentionally standalone and does not depend on torch or safetydistill.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class KKTAlignmentKey:
    task_suite_name: str
    safety_level: str
    task_index: int
    episode_index: int
    step_index: int

    def as_tuple(self):
        return (
            self.task_suite_name,
            self.safety_level,
            self.task_index,
            self.episode_index,
            self.step_index,
        )

    def __str__(self):
        return "|".join(
            [
                str(self.task_suite_name),
                str(self.safety_level),
                str(self.task_index),
                str(self.episode_index),
                str(self.step_index),
            ]
        )


class KKTSidecarIndex:
    def __init__(self, manifest_path: str, require_kkt: bool = True) -> None:
        self.manifest_path = manifest_path
        self.require_kkt = require_kkt

        self.num_files = 0
        self.num_records = 0
        self.num_indexed_records = 0
        self.duplicate_keys = []
        self.missing_key_records = []
        self.skipped_no_kkt_records = []

        self._index = {}

        self._load_manifest_and_index()

    def __len__(self) -> int:
        return self.num_indexed_records

    def has_key(self, key: KKTAlignmentKey) -> bool:
        return key in self._index

    def get(self, key: KKTAlignmentKey) -> Optional[Dict[str, Any]]:
        return self._index.get(key)

    def iter_keys(self, limit: Optional[int] = None) -> Iterable[KKTAlignmentKey]:
        count = 0
        for key in self._index.keys():
            if limit is not None and count >= limit:
                break
            yield key
            count += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "require_kkt": self.require_kkt,
            "num_files": self.num_files,
            "num_records": self.num_records,
            "num_indexed_records": self.num_indexed_records,
            "duplicate_keys": len(self.duplicate_keys),
            "missing_key_records": len(self.missing_key_records),
            "skipped_no_kkt_records": len(self.skipped_no_kkt_records),
        }

    def _load_manifest_and_index(self) -> None:
        if not os.path.isfile(self.manifest_path):
            return

        with open(self.manifest_path, "r") as f:
            try:
                manifest = json.load(f)
            except json.JSONDecodeError:
                return

        file_entries = manifest.get("files", [])
        if not file_entries:
            return

        self.num_files = len(file_entries)
        manifest_dir = os.path.dirname(os.path.abspath(self.manifest_path))

        for file_entry in file_entries:
            jsonl_path = self._resolve_jsonl_path(file_entry, manifest_dir)
            if not jsonl_path or not os.path.isfile(jsonl_path):
                continue

            self._index_jsonl(jsonl_path)

    def _resolve_jsonl_path(self, file_entry: Any, manifest_dir: str) -> Optional[str]:
        if isinstance(file_entry, str):
            path = file_entry
        elif isinstance(file_entry, dict):
            path = (
                file_entry.get("path")
                or file_entry.get("file_path")
                or file_entry.get("filepath")
                or file_entry.get("jsonl_path")
            )
        else:
            path = None

        if not path:
            return None

        if not os.path.isabs(path):
            path = os.path.join(manifest_dir, path)
        return path

    def _index_jsonl(self, jsonl_path: str) -> None:
        try:
            with open(jsonl_path, "r") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    self.num_records += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    key = self._build_key(record)
                    if key is None:
                        self.missing_key_records.append({"file": jsonl_path, "line": line_idx})
                        continue

                    if self.require_kkt and not self._has_kkt_fields(record):
                        self.skipped_no_kkt_records.append({"file": jsonl_path, "line": line_idx})
                        continue

                    if key in self._index:
                        self.duplicate_keys.append({"key": str(key), "file": jsonl_path, "line": line_idx})
                        continue

                    self._index[key] = record
                    self.num_indexed_records += 1
        except OSError:
            return

    def _build_key(self, record: Dict[str, Any]) -> Optional[KKTAlignmentKey]:
        required_fields = [
            "task_suite_name",
            "safety_level",
            "task_index",
            "episode_index",
            "step_index",
        ]
        for field in required_fields:
            if field not in record:
                return None

        try:
            return KKTAlignmentKey(
                task_suite_name=str(record["task_suite_name"]),
                safety_level=str(record["safety_level"]),
                task_index=int(record["task_index"]),
                episode_index=int(record["episode_index"]),
                step_index=int(record["step_index"]),
            )
        except (TypeError, ValueError):
            return None

    def _has_kkt_fields(self, record: Dict[str, Any]) -> bool:
        required = [
            "dual_variables",
            "active_set",
            "constraint_values",
            "constraint_gradients",
        ]
        for field in required:
            if field not in record or record.get(field) is None:
                return False
        return True
