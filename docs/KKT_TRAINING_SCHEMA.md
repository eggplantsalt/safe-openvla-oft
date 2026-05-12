# KKT Training Schema and Alignment Notes

This document describes the KKT sidecar schema, alignment keys, and Phase 7B inspection flow.

## KKT Sidecar Record Schema

Minimum expected fields per record (from safetydistill JSONL):

- task_suite_name
- safety_level
- task_index
- episode_index
- step_index
- instruction
- action_nominal
- action_safe
- action_delta
- dual_variables
- active_set
- constraint_values
- constraint_gradients
- qp_status

Notes:
- If any of the key fields are missing, the record cannot be aligned by explicit key.
- If any of the KKT fields are missing and require_kkt is enabled, the record is skipped in the index.

## Alignment Key Design

Preferred explicit alignment key:

- task_suite_name
- safety_level
- task_index
- episode_index
- step_index

Why explicit keys are required:
- Order-based alignment can silently drift if any sample is filtered, shuffled, or re-chunked.
- RLDS pipelines often apply filtering, sampling, or transformation that breaks raw ordering.

## Why Order Alignment Is Not Recommended

- RLDS uses shuffle buffers and interleaving, which changes sample order.
- Filtering (e.g., skip_unlabeled) changes sample cardinality.
- Any future dataset mix or augmentation can change ordering.

## Phase 7B Adapter and Inspection Usage

Adapter:
- prismatic/vla/datasets/kkt_sidecar.py
- Provides KKTSidecarIndex for loading manifest.json and indexing JSONL records.

Inspection script:
- scripts/inspect_kkt_openvla_alignment.py

Example usage:

```bash
python scripts/inspect_kkt_openvla_alignment.py \
  --kkt-manifest-path /PATH/TO/manifest.json \
  --max-kkt-print 3
```

Optional OpenVLA dataset inspection (may fail if dataset is missing or dependencies are not installed):

```bash
python scripts/inspect_kkt_openvla_alignment.py \
  --kkt-manifest-path /PATH/TO/manifest.json \
  --try-openvla-dataset \
  --data-root-dir /PATH/TO/RLDS \
  --dataset-name libero_spatial_no_noops
```

## If Alignment Is Not Possible

If explicit alignment keys are not present in OpenVLA samples:

- Prefer updating the data export step in safetydistill to include full OpenVLA training samples
  (image, proprio, language, action, and KKT fields), or
- Add explicit metadata fields in the RLDS pipeline at preprocessing time.

## OpenVLA-Style Full Sample Schema (Phase 8B)

When explicit alignment keys are not available in RLDS, safetydistill exports complete OpenVLA-style samples.
Each step should include images, state, instruction, actions, and KKT fields in a single JSONL record.

Expected step fields (minimum):

- instruction
- agentview_image or agentview_image_path
- wrist_image or wrist_image_path
- state
- action_nominal
- action_safe
- action_delta
- dual_variables
- active_set
- constraint_values
- constraint_gradients
- qp_status
- task_suite_name
- safety_level
- task_index
- episode_index
- step_index

This format avoids relying on RLDS ordering or sidecar alignment keys.

## Phase 8B Inspection Usage

Dataset adapter:
- prismatic/vla/datasets/kkt_openvla_dataset.py

Inspection script:
- scripts/inspect_kkt_openvla_samples.py

Example usage:

```bash
python scripts/inspect_kkt_openvla_samples.py \
  --manifest-path /PATH/TO/manifest.json \
  --batch-size 8
```

Optional image loading:

```bash
python scripts/inspect_kkt_openvla_samples.py \
  --manifest-path /PATH/TO/manifest.json \
  --load-images
```
