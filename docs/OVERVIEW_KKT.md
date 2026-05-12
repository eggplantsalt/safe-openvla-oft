# OVERVIEW_KKT：OpenVLA-OFT KKT 接入点分析

> 本文档仅基于静态代码阅读，不训练/不下载/不改模型与训练逻辑；不确定处标注 [需确认]。

## 1. 结论摘要

- **最推荐的 KKT 接入方案**：Phase 7B 采用“方案 C：独立 batch inspection”作为最小可行验证；中期接入建议优先沿用 safetydistill 的 KKTJsonlDataset（方案 A）以减少重复实现。
- **最大不确定点**：RLDS/LIBERO 训练样本中是否包含 `task_index / episode_index / step_index` 等对齐字段；目前数据流仅看到 `dataset_name` 与 `language_instruction`，对齐依据不足。[需确认]
- **Phase 7B 下一步**：仅新增对齐/抽样检查脚本与 sidecar adapter（不入训练 loop），验证 manifest/JSONL 与 OpenVLA-OFT RLDS 样本是否可稳定对齐。

## 2. 已阅读文件列表

- [docs/KKT_GLOBAL_CONTEXT.md](docs/KKT_GLOBAL_CONTEXT.md)：KKT-SenseVLA 全局背景与约束、对齐原则。
- [vla-scripts/finetune.py](vla-scripts/finetune.py)：OpenVLA-OFT 训练入口、forward 与 loss 位置。
- [prismatic/vla/datasets/datasets.py](prismatic/vla/datasets/datasets.py)：RLDSBatchTransform、RLDSDataset 与样本字段。
- [prismatic/vla/datasets/rlds/dataset.py](prismatic/vla/datasets/rlds/dataset.py)：RLDS 读入、轨迹/帧转换、key 结构。
- [prismatic/vla/datasets/rlds/obs_transforms.py](prismatic/vla/datasets/rlds/obs_transforms.py)：图像解码/resize/增强。
- [prismatic/vla/datasets/rlds/traj_transforms.py](prismatic/vla/datasets/rlds/traj_transforms.py)：action/observation chunking 与 pad mask。
- [prismatic/vla/datasets/rlds/utils/data_utils.py](prismatic/vla/datasets/rlds/utils/data_utils.py)：action/proprio 归一化与统计缓存。
- [prismatic/vla/datasets/rlds/oxe/materialize.py](prismatic/vla/datasets/rlds/oxe/materialize.py)：Open-X 数据集 kwargs 组装入口。
- [prismatic/vla/materialize.py](prismatic/vla/materialize.py)：Dataset + collator 工厂函数。
- [prismatic/util/data_utils.py](prismatic/util/data_utils.py)：PaddedCollatorForActionPrediction 与 batch keys。
- [prismatic/models/action_heads.py](prismatic/models/action_heads.py)：L1/扩散动作头输入输出。
- [prismatic/training/metrics.py](prismatic/training/metrics.py)：通用 logging 容器（本训练脚本未使用）。
- [experiments/robot/openvla_utils.py](experiments/robot/openvla_utils.py)：推理端 action 反归一化与 predict_action 路径。
- [experiments/robot/libero/run_libero_eval.py](experiments/robot/libero/run_libero_eval.py)：LIBERO 评估入口，确认 student checkpoint 评测路径。

## 3. LIBERO / RLDS training data flow

**数据集创建路径**

- 训练脚本中通过 [vla-scripts/finetune.py](vla-scripts/finetune.py) 创建 `RLDSBatchTransform` 与 `RLDSDataset`。
- `RLDSDataset` 内部调用 [prismatic/vla/datasets/rlds/dataset.py](prismatic/vla/datasets/rlds/dataset.py) 的 `make_interleaved_dataset()`，最终通过 TFDS/RLDS pipeline 输出样本。

**单条 sample keys（RLDSBatchTransform 输入）**

- 来自 `make_dataset_from_rlds()` 的结构：
  - `observation`: `image_*`, `depth_*`, `proprio`（若启用）, `timestep`
  - `task`: `language_instruction`
  - `action`
  - `dataset_name`
  - `absolute_action_mask`（若配置）
- 在 `traj_transforms.chunk_act_obs()` 后，`action` 与 `observation` 带有 chunk 维度。
- **是否包含 task/episode/step metadata**：当前路径未发现 `task_index / episode_index / step_index`。仅看到 `dataset_name` 与 `language_instruction`。[需确认]

**Batch keys（进入训练 loop）**

由 [prismatic/util/data_utils.py](prismatic/util/data_utils.py) 的 `PaddedCollatorForActionPrediction` 生成：
- `pixel_values`
- `input_ids`
- `attention_mask`
- `labels`
- `actions`
- `proprio`（若存在）
- `dataset_names`（若存在）

**image / language / proprio / action 的流动路径**

- image：RLDS `observation.image_*` -> `RLDSBatchTransform` -> `pixel_values` -> model forward
- language：`task.language_instruction` -> prompt -> `input_ids/labels`
- proprio：`observation.proprio` -> batch `proprio` -> model forward
- action：`action` -> `actions`（监督目标）与 prompt action token（labels）

**Sidecar alignment 风险**

- 现有 pipeline 中缺少 `task_index / episode_index / step_index`，sidecar 对齐只能依赖顺序或自定义注入，风险高。[需确认]

## 4. Action target 和 loss flow

**Action label 来源**

- `RLDSBatchTransform` 直接从 `rlds_batch["action"]` 读取当前动作与未来动作 chunk。
- 进入训练时以 batch key `actions` 形式提供给 loss（连续动作场景）。

**Action normalization**

- 归一化发生在 [prismatic/vla/datasets/rlds/utils/data_utils.py](prismatic/vla/datasets/rlds/utils/data_utils.py) 的 `normalize_action_and_proprio()`。
- 统计量在 [vla-scripts/finetune.py](vla-scripts/finetune.py) 训练前保存为 `dataset_statistics.json`。
- 推理端通过 [experiments/robot/openvla_utils.py](experiments/robot/openvla_utils.py) 的 `_load_dataset_stats()` 读取并用于 `vla.predict_action(..., unnorm_key=...)`。

**Action head 位置**

- 连续动作时，`actions_hidden_states` 从 `output.hidden_states[-1]` 切片后送入 action head。
- 入口位于 [vla-scripts/finetune.py](vla-scripts/finetune.py) 的 `run_forward_pass()`。

**L1 / diffusion loss 位置**

- L1 回归 loss：`run_forward_pass()` 内对 `predicted_actions` 与 `ground_truth_actions` 计算。
- Diffusion loss：`run_forward_pass()` 内对 `noise_pred` 与 `noise` 计算。
- Loss 不在 head 内部完成，而在训练脚本内完成。

**最适合加入 action_safe / action_delta supervision 的位置**

- 在 `run_forward_pass()` 读取 `batch["actions"]` 之处替换或扩展 supervision（例如替换 `ground_truth_actions` 为 `action_safe`）。
- 若 KKT sidecar 不进入训练，Phase 7B 仅做 batch inspection，不改此路径。

## 5. Model forward / hidden state 接入点

**可用 hidden state**

- `output.hidden_states[-1]`：最后一层 hidden state。
- `text_hidden_states = last_hidden_states[:, num_patches:-1]`：去掉视觉 patch 与终止 token。
- `actions_hidden_states`：根据 `current_action_mask | next_actions_mask` 取出的 action token hidden states。

**KKT auxiliary heads 的接入建议**

- 最小改动：复用 `actions_hidden_states` 作为 KKT heads 输入，避免修改 VLA 主体。
- 备选：使用 `text_hidden_states` 的 pooled 表达作为 KKT 辅助特征。[需确认]

**应避免修改的底层文件**

- [prismatic/extern/hf/modeling_prismatic.py](prismatic/extern/hf/modeling_prismatic.py)（核心模型结构）
- [prismatic/models/backbones](prismatic/models/backbones) 下的 LLM/视觉骨干
- [prismatic/vla/action_tokenizer.py](prismatic/vla/action_tokenizer.py)（动作离散化策略）

## 6. KKT sidecar label 接入方案

**方案 A：OpenVLA-OFT 直接 import safetydistill 的 KKTJsonlDataset**
- 优点：复用已验证 schema、collate 逻辑，减少重复实现。
- 缺点：需要解决与 RLDS 样本对齐问题；依赖外部仓库路径/PYTHONPATH。

**方案 B：OpenVLA-OFT 自己实现 KKT sidecar index（manifest/JSONL）**
- 优点：对齐逻辑完全可控；可与 RLDS 数据结构深度绑定。
- 缺点：重复实现、易产生 schema 漂移。

**方案 C：先做独立 batch inspection，不进入训练 loop**
- 优点：最小侵入、零训练修改；可快速验证对齐策略是否可行。
- 缺点：不验证训练阶段的 loss 与梯度路径。

**推荐方案**
- Phase 7B 推荐方案 C；若对齐可行，Phase 7C 再逐步引入方案 A。

## 7. Alignment key 设计

**理想 key**

- `task_suite_name / safety_level / task_index / episode_index / step_index`

**当前 OpenVLA-OFT 是否有这些字段**

- 在 RLDS pipeline 与 `RLDSBatchTransform` 中未看到上述字段，仅见 `dataset_name` 和 `language_instruction`。[需确认]

**如果缺失，需要在哪里补**

- 选项 1：在 [prismatic/vla/datasets/rlds/dataset.py](prismatic/vla/datasets/rlds/dataset.py) 的 `make_dataset_from_rlds()` 里注入 metadata（高风险，Phase 7B 不做）。
- 选项 2：在 `RLDSBatchTransform` 后插入 sidecar lookup（中等风险，需对齐策略明确）。

**临时 synthetic dataset 如何对齐**

- 若 safetydistill 生成样本顺序与 RLDS dataset 顺序一致，可临时做顺序对齐，但风险高。[需确认]

**正式数据如何对齐**

- 推荐通过显式 key（task/episode/step）对齐；需要确认 RLDS 原始样本是否提供这些字段，或在数据预处理阶段引入。[需确认]

## 8. 推荐新增文件（Phase 7B）

- `prismatic/vla/datasets/kkt_sidecar.py`：KKT sidecar 对齐与读取的最小 adapter（不进训练 loop）。
- `scripts/inspect_kkt_openvla_alignment.py`：离线验证 RLDS sample 与 KKT JSONL 是否能对齐。
- `docs/KKT_TRAINING_SCHEMA.md`：记录最终对齐字段、目标 schema 与 masks 规则。

## 9. 高风险文件（不建议第一版大改）

- [vla-scripts/finetune.py](vla-scripts/finetune.py)
- [prismatic/extern/hf/modeling_prismatic.py](prismatic/extern/hf/modeling_prismatic.py)
- [prismatic/vla/datasets/rlds/dataset.py](prismatic/vla/datasets/rlds/dataset.py)
- [prismatic/models/backbones](prismatic/models/backbones) 目录

## 10. Phase 7B 实施计划（最小可行）

1. 不训练、不改模型，仅新增 sidecar 对齐与 batch inspection 脚本。
2. 读取 RLDS sample（通过 `RLDSDataset` 迭代）并采样输出关键字段。
3. 读取 KKT manifest/JSONL（通过 safetydistill 的 loader 或本地简易 parser）。
4. 验证对齐键：若无 `task_index/episode_index/step_index`，输出无法对齐的证据。[需确认]
5. 形成对齐结论，作为 Phase 7C 是否进入训练 loop 的准入条件。

## 11. Phase 7B 交付内容

- `KKTSidecarIndex`：只读 sidecar 索引，不依赖 safetydistill import，不进入训练 loop。
- `inspect_kkt_openvla_alignment.py`：KKT-only inspection 必须可运行；OpenVLA dataset inspection 为可选分支。
- 仍不进入训练 loop；对齐验证是进入 KKT loss 的前置条件。

## 12. Phase 8B：OpenVLA-Style KKT Samples

- 使用 safetydistill 导出的完整 OpenVLA-style KKT 样本，避免 RLDS sidecar 对齐问题。
- 新增 `KKTOpenVLASampleDataset` 与 `collate_kkt_openvla_samples` 仅做样本与 batch inspection。
- 仍不进入训练 loop；KKT loss 接入仍需后续阶段评估。

## 13. Phase 8C：Chunked Pretraining Adapter

- 新增 `KKTChunkedOpenVLADataset` 与 `collate_kkt_openvla_chunk_samples` 生成 action chunk 与 KKT chunk。
- 该 adapter 是接入 finetune 前的中间层，用于验证 chunk 对齐与 padding 规则。
- 仍不进入 training loop。

## 14. Phase 8D：Pretraining-Style Batch Adapter

- 新增预训练风格 batch adapter，输出接近 finetune.py 期望的 batch keys（但使用 dummy tokenization）。
- 不修改 finetune.py，不进入 training loop。
- 为 Phase 8E 接入真实 processor/tokenizer 做准备。

---

**完成后报告**

- 新增/修改文件：本阶段仅新增 [docs/OVERVIEW_KKT.md](docs/OVERVIEW_KKT.md)。
- 关键发现：训练 batch keys 中未见 task/episode/step 级 metadata；action loss 计算位置在 `run_forward_pass()`。
- 最大不确定点：RLDS 原始样本是否包含可用于对齐的索引字段。[需确认]
- 推荐 Phase 7B：执行 sidecar 对齐验证脚本与 batch inspection，不入训练 loop。
