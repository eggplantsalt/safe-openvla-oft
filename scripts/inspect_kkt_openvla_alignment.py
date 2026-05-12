"""
inspect_kkt_openvla_alignment.py

KKT sidecar inspection and optional OpenVLA dataset metadata inspection.
This script does not enter any training loop and does not download models or datasets.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

# Allow running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prismatic.vla.datasets.kkt_sidecar import KKTAlignmentKey, KKTSidecarIndex


def _has_fields(record: Dict[str, Any], fields: Iterable[str]) -> Dict[str, bool]:
    return {field: field in record for field in fields}


def _collect_keys(obj: Any, prefix: str = "") -> List[str]:
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            keys.append(full_key)
            keys.extend(_collect_keys(v, full_key))
    return keys


def _print_kkt_sample(index: KKTSidecarIndex, max_kkt_print: int) -> List[str]:
    printed_keys = []
    for key in index.iter_keys(limit=max_kkt_print):
        record = index.get(key)
        if record is None:
            continue

        printed_keys.append(str(key))
        print("- key:", str(key))
        print("  instruction:", record.get("instruction", "[missing]"))

        action_fields = ["action_nominal", "action_safe", "action_delta"]
        kkt_fields = [
            "dual_variables",
            "active_set",
            "constraint_values",
            "constraint_gradients",
        ]
        action_presence = _has_fields(record, action_fields)
        kkt_presence = _has_fields(record, kkt_fields)

        print("  action_fields:", action_presence)
        print("  kkt_fields:", kkt_presence)
        print("  qp_status:", record.get("qp_status", "[missing]"))
    return printed_keys


def _inspect_openvla_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.try_openvla_dataset:
        print("OpenVLA dataset inspection skipped.")
        return {
            "sample_keys_observed": [],
            "candidate_alignment_fields": [],
            "error": None,
        }

    if not args.data_root_dir or not args.dataset_name:
        raise ValueError("--data-root-dir and --dataset-name are required when --try-openvla-dataset is set.")

    try:
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK
        from prismatic.vla.datasets.rlds.dataset import make_interleaved_dataset
        from prismatic.vla.datasets.rlds.oxe.materialize import get_oxe_dataset_kwargs_and_weights
        from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES

        if args.dataset_name in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[args.dataset_name]
        else:
            mixture_spec = [(args.dataset_name, 1.0)]

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            args.data_root_dir,
            mixture_spec,
            load_camera_views=("primary", "wrist"),
            load_depth=False,
            load_proprio=True,
            load_language=True,
        )

        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,
                future_action_window_size=NUM_ACTIONS_CHUNK - 1,
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=dict(
                resize_size=(1, 1),
                depth_resize_size=(1, 1),
                image_augment_kwargs={},
                num_parallel_calls=1,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=1,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=True,
        )

        dataset, _, _ = make_interleaved_dataset(**rlds_config)

        keys = []
        if hasattr(dataset, "element_spec"):
            keys = _collect_keys(dataset.element_spec)
        else:
            sample = next(dataset.as_numpy_iterator())
            keys = _collect_keys(sample)

        keys = sorted(set(keys))

        candidate_fields = [
            k
            for k in keys
            if k.endswith("task_index")
            or k.endswith("episode_index")
            or k.endswith("step_index")
            or k.endswith("trajectory_id")
            or k.endswith("episode_id")
            or k.endswith("timestep")
            or k.endswith("language_instruction")
            or k.endswith("dataset_name")
        ]

        return {
            "sample_keys_observed": keys,
            "candidate_alignment_fields": candidate_fields,
            "error": None,
        }
    except Exception as exc:
        return {
            "sample_keys_observed": [],
            "candidate_alignment_fields": [],
            "error": str(exc),
        }


def _alignment_conclusion(sample_keys: List[str]) -> str:
    if not sample_keys:
        return "cannot_verify_alignment_without_openvla_metadata"

    key_fields = ["task_index", "episode_index", "step_index"]
    if any(any(k.endswith(f) for f in key_fields) for k in sample_keys):
        return "explicit_alignment_possible"

    if any(k.endswith("language_instruction") or k.endswith("dataset_name") for k in sample_keys):
        return "explicit_alignment_not_found"

    return "cannot_verify_alignment_without_openvla_metadata"


def _recommendation_for_conclusion(conclusion: str) -> str:
    if conclusion == "explicit_alignment_possible":
        return "Proceed to explicit alignment using task/episode/step metadata."
    if conclusion == "explicit_alignment_not_found":
        return "Add explicit metadata or export full OpenVLA training samples from safetydistill."
    return "Run OpenVLA dataset inspection with valid data and retry alignment verification."


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect KKT sidecar and OpenVLA dataset alignment metadata.")
    parser.add_argument("--kkt-manifest-path", required=True)
    parser.add_argument("--require-kkt", action="store_true", default=False)
    parser.add_argument("--max-kkt-print", type=int, default=3)
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--data-root-dir", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--try-openvla-dataset", action="store_true", default=False)
    args = parser.parse_args()

    index = KKTSidecarIndex(args.kkt_manifest_path, require_kkt=args.require_kkt)
    summary = index.summary()

    print("KKT summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("KKT sample keys:")
    printed_keys = _print_kkt_sample(index, args.max_kkt_print)
    if not printed_keys:
        print("  [no indexed records]")

    openvla_result = _inspect_openvla_dataset(args)

    if openvla_result.get("error"):
        print("OpenVLA dataset inspection failed:")
        print(openvla_result["error"])

    conclusion = _alignment_conclusion(openvla_result.get("sample_keys_observed", []))
    recommendation = _recommendation_for_conclusion(conclusion)

    print("Alignment conclusion:", conclusion)
    print("Recommendation:", recommendation)

    if args.output_report:
        report = {
            "kkt_summary": summary,
            "sample_keys_observed": openvla_result.get("sample_keys_observed", []),
            "candidate_alignment_fields": openvla_result.get("candidate_alignment_fields", []),
            "conclusion": conclusion,
            "recommendation": recommendation,
        }
        with open(args.output_report, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
