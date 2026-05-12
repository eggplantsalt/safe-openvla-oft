# KKT-SenseVLA for OpenVLA-OFT：全局上下文文档
> 本文档用于在 codeagent 上下文有限的情况下，持续提供 KKT-SenseVLA 项目背景、当前工程状态、OpenVLA-OFT 仓库职责、与 safetydistill/AEGIS teacher 仓库的连接方式，以及后续开发边界。  
> 每次向 codeagent 提需求时，都请附带本文档，避免上下文断裂。
---
## 1. 项目总目标
我们正在开发一个 VLA safety defense / safety-aware distillation 方法，暂定名：
**KKT-SenseVLA**
核心思想不是简单让 VLA 模仿安全动作，而是让 VLA 学会安全优化器背后的结构化安全判断：
- 哪条安全约束正在变紧；
- 约束压力有多大；
- 约束在动作空间中希望动作往哪个方向修正；
- 当前动作为什么不安全；
- 如何产生更符合 KKT 最优性结构的安全动作。
传统 AEGIS / CBF-QP safety layer 的形式是：
```text
VLA 输出原始动作 a0
        ↓
AEGIS / CBF-QP 安全优化器
        ↓
输出安全动作 a*
KKT-SenseVLA 的目标不是在推理时永远外挂 CBF-QP，而是：
teacher safety optimizer 产生安全动作和 KKT certificate
        ↓
离线生成 KKT-SafeLIBERO 标签
        ↓
student VLA 学习 action + KKT safety structure
        ↓
student 在内部形成 KKT safety consciousness
也就是说，我们希望 student 不只是学会“答案是什么”，还学会“为什么要这样修正”。
2. 两个仓库的职责分工
当前项目分成两个主要代码库。
2.1 safetydistill / VLSA-AEGIS 仓库
这个仓库是 teacher-side data generation repo。
它负责：
SafeLIBERO / AEGIS rollout；
CBF-QP safety optimizer；
synthetic / real obstacle safety control；
KKT certificate extraction；
JSONL label export；
manifest 构建；
label validator；
dataset loader；
training target schema；
smoke tests。
该仓库已经完成并验证：
VLSA/AEGIS rollout
→ action_nominal / action_safe / action_delta 捕获
→ CVXPY QP certificate 导出
→ dual_variables / active_set / constraint_values / constraint_gradients 写入 JSONL
→ manifest.json 构建
→ KKTJsonlDataset 读取
→ training target batch collate
当前已验证 prototype dataset：
manifest path example:
/storage/v-xiangxizheng/zy_workspace/safetydistill/data/kkt_safelibero_synthetic_debug/manifest.json
num_files: 3
num_records: 900
qp_status_counts: {"optimal": 900}
KKT fields:
  dual_variables
  active_set
  constraint_values
  constraint_gradients
该仓库中的关键模块：
kkt_sense/dataset.py
kkt_sense/training_targets.py
kkt_sense/scripts/validate_kkt_labels.py
kkt_sense/scripts/build_kkt_manifest.py
kkt_sense/scripts/inspect_kkt_dataset.py
kkt_sense/scripts/inspect_kkt_training_batch.py
OpenVLA-OFT 训练阶段可以通过以下方式使用这些工具：
export PYTHONPATH=$PYTHONPATH:/storage/v-xiangxizheng/zy_workspace/safetydistill
然后在 OpenVLA-OFT 代码中：
from kkt_sense.dataset import KKTJsonlDataset
from kkt_sense.training_targets import build_training_target, collate_training_targets
后续也可以把 safetydistill 做成 editable package：
pip install -e /storage/v-xiangxizheng/zy_workspace/safetydistill
但当前优先使用 PYTHONPATH，减少 packaging 工作量。
2.2 OpenVLA-OFT 仓库
这个仓库是 student training repo。
它负责：
OpenVLA-OFT 模型加载；
LIBERO / RLDS 数据读取；
图像、语言、proprio 输入处理；
action head；
LoRA / OFT fine-tuning；
loss 计算；
KKT auxiliary head / loss 的接入；
student checkpoint 训练和评估。
OpenVLA-OFT 不应该负责：
运行 AEGIS teacher；
运行 SafeLIBERO rollout 生成 KKT 标签；
在线调用 CBF-QP solver；
在线调用 GroundingDINO；
复刻 safetydistill 里的 teacher data generation 逻辑。
正确关系是：
safetydistill / VLSA-AEGIS
    负责离线生成 KKT JSONL + manifest
        ↓
OpenVLA-OFT
    负责读取 manifest / JSONL 作为 sidecar labels
        ↓
训练 student VLA
错误关系是：
OpenVLA-OFT training step
    每一步实时调用 AEGIS / CBF-QP / SafeLIBERO teacher
不要这样做。
3. OpenVLA-OFT 当前代码结构理解
根据当前仓库静态分析，OpenVLA-OFT 主要结构为：
openvla-oft/
├─ vla-scripts/
│  ├─ finetune.py
│  ├─ deploy.py
│  └─ merge_lora_weights_and_save.py
├─ prismatic/
│  ├─ models/
│  ├─ training/
│  ├─ vla/
│  ├─ preprocessing/
│  ├─ extern/
│  └─ util/
├─ experiments/
│  └─ robot/
│     ├─ libero/
│     ├─ aloha/
│     └─ openvla_utils.py
├─ docs/
├─ README.md
├─ SETUP.md
├─ LIBERO.md
└─ pyproject.toml
当前已知关键文件：
vla-scripts/finetune.py
    OpenVLA-OFT 训练入口，使用 draccus/dataclass 配置，负责 LoRA 微调。
prismatic/vla/datasets/rlds/dataset.py
    RLDS / LIBERO 数据加载、轨迹转换、batch 构建相关入口。
prismatic/models/action_heads.py
    连续动作头实现，包括 L1 regression action head 和 diffusion action head。
prismatic/vla/constants.py
    根据平台设置 ACTION_DIM、NUM_ACTIONS_CHUNK、PROPRIO_DIM 等常量。
experiments/robot/libero/run_libero_eval.py
    LIBERO 评估入口。
experiments/robot/openvla_utils.py
    评估和模型加载相关工具。
需要进一步确认的关键问题：
LIBERO training dataset 的单条 sample 具体包含哪些字段；
sample 是否包含 task index、episode index、step index；
如果没有这些字段，如何和 KKT JSONL sidecar labels 对齐；
action label 在哪里读取、归一化、反归一化；
model forward 返回哪些中间表示；
action head 的输入 hidden state 在哪里；
loss 在哪里计算；
trainer / collator 在哪里组装 batch；
加 KKT auxiliary head 的最小侵入点在哪里。
4. KKT sidecar label 数据格式
safetydistill 仓库已经生成的每条 KKT record 包含：
task_suite_name
safety_level
task_index
episode_index
step_index
instruction
action_nominal
action_safe
action_delta
dual_variables
active_set
constraint_values
constraint_gradients
qp_status
collision_info
extra_debug
经过 KKTJsonlDataset 和 training_targets.py 后，训练侧可获得标准化字段：
inputs:
  instruction
metadata:
  task_suite_name
  safety_level
  task_index
  episode_index
  step_index
  qp_status
  instruction
targets:
  action
  action_nominal
  action_safe
  action_delta
  dual_cbf_main
  active_cbf_main
  h
  linear_cbf_lhs
  constraint_direction
masks:
  has_action
  has_kkt
  active_cbf_main
  qp_valid
其中：
action:
  可选择 action_safe / action_delta / action_nominal 作为监督目标。
constraint_direction:
  concat(a_u_v, a_uz)，shape = (6,)
dual_cbf_main:
  当前主 CBF 约束的 dual variable，表示约束压力。
active_cbf_main:
  当前主 CBF 约束是否 active。
h:
  CBF / safety margin。
linear_cbf_lhs:
  当前 QP constraint 的线性左侧值。
5. OpenVLA-OFT 中 KKT 接入的推荐原则
5.1 不要先大改模型
第一步不要直接加复杂 KKT correction layer，也不要立刻改完整 OpenVLA 模型结构。
推荐顺序：
Phase 7A：代码库勘探，确认数据流和 loss 位置
Phase 7B：只接入 KKT sidecar dataset adapter / batch inspection
Phase 7C：加 KKT labels 到 training batch，但不改 loss
Phase 7D：加 auxiliary KKT heads 和 loss
Phase 7E：加入 action_distillation loss / delta loss
Phase 7F：再考虑 KKT correction layer
5.2 最小可行训练目标
最小 student 训练目标可以先是：
主 action loss:
  预测 action_safe 或 action_delta
辅助 KKT loss:
  预测 dual_cbf_main
  预测 active_cbf_main
  预测 h
  预测 constraint_direction
但第一版不一定全部做。最小建议：
action target:
  action_safe 或 action_delta
auxiliary:
  dual_cbf_main
  active_cbf_main
先让代码跑通，再逐步增加 h / constraint_direction。
5.3 loss mask 规则
KKT labels 不一定每条都有，因此必须使用 masks：
has_action:
  action supervision 是否有效
has_kkt:
  KKT supervision 是否有效
qp_valid:
  QP 是否有效；no_safety_control 样本应屏蔽 KKT loss
active_cbf_main:
  可用于 active constraint 分类，也可用于 hard-example weighting
6. 两个仓库如何连接
6.1 推荐连接方式
在 OpenVLA-OFT 环境中，把 safetydistill 仓库加入 PYTHONPATH：
export PYTHONPATH=$PYTHONPATH:/storage/v-xiangxizheng/zy_workspace/safetydistill
OpenVLA-OFT 代码中直接 import：
from kkt_sense.dataset import KKTJsonlDataset
from kkt_sense.training_targets import build_training_target, collate_training_targets
训练命令中传入：
--kkt_manifest_path /storage/v-xiangxizheng/zy_workspace/safetydistill/data/kkt_safelibero_synthetic_debug/manifest.json
6.2 KKT sidecar 对齐方式
理想 alignment key 是：
task_suite_name
safety_level
task_index
episode_index
step_index
如果 OpenVLA-OFT 原始 LIBERO sample 中没有这些字段，则需要进一步选择：
方案 A：修改 OpenVLA-OFT 数据预处理，让 sample 带上 task/episode/step metadata。
方案 B：用顺序对齐，即 dataset order 与 KKT JSONL 顺序一致。风险较高，不推荐作为正式方法。
方案 C：让 safetydistill 未来导出 OpenVLA-OFT 可直接读取的完整训练样本，包含 image / state / instruction / action / KKT labels。工程更重，但最自洽。
当前 Phase 7A 重点就是确认 OpenVLA-OFT 是否已有可用 alignment key。
7. 当前已经完成的阶段
Teacher side 已完成
Phase 1：kkt_sense 标签 schema / JSONL 导出骨架
Phase 2：接入 main_aegis_translational.py，捕获 action_nominal / action_safe
Phase 3A：从 CVXPY QP 导出 dual / active_set / constraint_values / gradients
Phase 3B：稳健性修补
Phase 3C/3D：debug smoke test，确认真实 SafeLIBERO episode 能导出 JSONL
Phase 4A：synthetic obstacle 触发 CBF-QP，导出非空 KKT certificate
Phase 4B：dummy server / validator / smoke test 文档正规化
Phase 5A：批量 synthetic KKT mini-dataset 生成
Phase 5B：manifest/index 构建
Phase 5C：KKTJsonlDataset loader
Phase 6A：training target builder / batch collate 检查
当前验证结果
已验证：
dataset_length: 900
batch_size: 8
targets.action: (8, 7)
targets.constraint_direction: (8, 6)
targets.dual_cbf_main: (8, 1)
masks.has_action sum: 8
masks.has_kkt sum: 8
masks.qp_valid sum: 8
说明 KKT label 已经能被整理成 student training target batch。
