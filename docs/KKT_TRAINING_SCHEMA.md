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

## Phase 8C Chunked Schema (Pretraining-Style)

To align with OpenVLA-OFT training expectations (NUM_ACTIONS_CHUNK), each sample is expanded into
an action chunk with padding and masks.

Current step inputs:
- instruction
- agentview_image / wrist_image (optional)
- state (current step only)

Action chunk targets:
- actions: shape (chunk_size, 7)
- action_chunk_mask: shape (chunk_size,)

KKT targets:
- kkt_current: current step KKT signals
- kkt_chunk: per-step KKT signals for the action chunk

Padding behavior:
- If future steps are unavailable, actions and KKT fields are padded with zeros.
- action_chunk_mask marks valid steps with 1.0 and padding with 0.0.
- kkt_chunk.has_kkt and kkt_chunk.qp_valid are forced to 0.0 on padded positions.

## Phase 8D Pretraining-Style Batch Schema

Phase 8D builds an inspection batch that resembles finetune.py inputs without using the
real OpenVLA processor or tokenizer.

Key points:
- Dummy tokenization is used only for interface inspection; it is not the final training tokenizer.
- pixel_values are simple float32 [0,1] conversions; real training must use the OpenVLA image processor.

Batch fields (summary):
- pixel_values.agentview / pixel_values.wrist
- input_ids / attention_mask / labels
- actions / action_chunk_mask / proprio
- kkt_targets.current / kkt_targets.chunk
- kkt_masks.current_* / kkt_masks.chunk_* / action_chunk_mask

## Phase 8E Processor Compatibility (Inspection)

Phase 8E adds a processor/tokenizer compatibility adapter that can run in:

- dummy mode: uses Phase 8D dummy tokenization and simple image scaling
- real mode: attempts to load a local processor with local_files_only=True

Notes:
- real mode must not download any files; it only uses local paths.
- dummy mode is not a substitute for the real tokenizer or processor.
- no model forward or loss computation happens in Phase 8E.

## Phase 9A KKT Loss Schema (Dry-Run)

Loss components:
- action loss: L1 on action chunks with action_chunk_mask
- dual loss: L1 on dual_cbf_main
- active loss: BCEWithLogits on active_cbf_main
- h loss: L1 on h
- direction loss: cosine (1 - cos) or MSE on constraint_direction

Masking:
- current mask = current_has_kkt * current_qp_valid
- chunk mask = chunk_has_kkt * chunk_qp_valid * action_chunk_mask

Phase 9A only runs CPU dry-run with fake predictions; no model forward or training.

## Phase 9B KKT Heads (Dry-Run)

Heads:
- current head: takes (B, hidden_dim) and outputs dual/active/h/direction
- chunk head: takes (B, T, hidden_dim) and outputs per-step dual/active/h/direction

Notes:
- active output is a logit (for BCEWithLogitsLoss)
- heads are not yet wired to OpenVLA hidden states in Phase 9B
