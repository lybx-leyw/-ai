# TinySeek — 仓库结构与快速导航

本文件为仓库结构速览与快速上手参考，便于在本项目中定位主要实现、训练入口与常用工具。

概览（高层）

- 目标：实验性实现 MoE（Mixture-of-Experts）+ 样本级 Gate，支持 LoRA / Prefix 微调。
- 语言：Python（PyTorch），部分数据预处理工具使用 C（tools/utilities_c）。

顶层重要文件

- `run.py` — 交互式选择训练入口（LoRA/Pre/Prefix/SFT/Gate）。
- `evaluate.py` — 离线评估脚本，使用 `evaluate_agent.evaluate_function`。
- `agent_chat.py` — 交互式 chat 会话入口（调用 evaluate_agent.chat.start_chat_session）。
- `model_config.json` — 全局模型/训练超参。
- `requirements.txt` — Python 依赖（用于创建环境）。

目录说明（按职责）

- `model/`：模型与微调工具
    - [model/tiny_seek.py](model/tiny_seek.py#L1) — `TinySeek`, `gated_TinySeek`, `gate`。
    - [model/lora_model.py](model/lora_model.py#L1) — LoRA 插入/应用工具（`apply_lora`）。
    - [model/prefix_model.py](model/prefix_model.py#L1) — Prefix 插入工具（`apply_prefix`）。

- `modules/`：网络子模块与层
    - [modules/mla.py](modules/mla.py#L1) — MLA 注意力实现。
    - [modules/moe.py](modules/moe.py#L1) — MoE（专家路由）实现。
    - `modules/layers/` — `token_embedding.py`, `layer_norm.py`。

- `train_agent/`：训练 wrapper（用户入口）
    - wrapper 示例：`train_agent/sft_train.py`, `train_agent/lora_train.py`, `train_agent/pre_train.py`, `train_agent/prefix_train.py`, `train_agent/train_gate.py`。
    - `train_agent/trainers/` — 实际训练器实现（`pre_training.py`, `sft_training.py`, `lora_tuning.py`, `gated_training.py` 等）。

- `evaluate_agent/`：评估与交互实现
    - [evaluate_agent/evaluate_function.py](evaluate_agent/evaluate_function.py#L1) — 模型加载/评估/生成工具。
    - [evaluate_agent/chat.py](evaluate_agent/chat.py#L1) — 交互式会话逻辑。

- `tools/`：辅助工具
    - `tools/tokenizer.py`, `tools/vocab.py` — 分词与词表管理。
    - `tools/config_manager.py` — 读取 `model_config.json`。
    - `tools/mask.py` — 区间掩码（mask_from_id_to_id）。
    - `tools/plot.py`、`tools/fast_print.py` 等。
    - `tools/utilities_c/` — 可选的 C 源文件用于数据预处理（编译后为 exe）。

- `dataset/`：训练/评估数据（JSONL，含子目录与说明）。
- `docs/`：设计、训练与微调文档（`architecture.md`, `training.md`, 等）。
- `out/`, `final/`, `logs/`：运行时生成的 checkpoint、导出与日志目录（通常不纳入 Git 大文件）。

关键开发/运行约定（快速）

- 配置管理：wrapper 会通过 `tools.ConfigManager('model_config.json').config` 获取默认超参并可被覆盖。
- 数据格式：SFT/训练数据为 JSONL，SFT 的每条记录包含 `conversations`（role/content 列表），生成时使用 `<|im_end|>` 分割。
- 损失约定：训练器使用两路 cross-entropy（`fc1`/`fc2`）之和，并加上 MoE 的路由正则项（`alpha * Lexp`）。
- LoRA/Prefix：在加载/微调时先调用 `apply_lora`/`apply_prefix`，训练器通过模块名解冻对应参数。

常用命令示例

```powershell
# 环境创建
conda create -n tinyseek python=3.10 -y
conda activate tinyseek
python -m pip install -r requirements.txt

# 评估（交互式）
python evaluate.py

# 交互式训练选择
python run.py

# 启动交互式聊天（agent_chat）
python agent_chat.py
```

建议（维护与变更注意事项）

- 若修改 embedding 维度或输出头（`fc1`/`fc2`），请同步更新所有训练器中 logits 的形状处理与 checkpoint 兼容逻辑。
- `gated_TinySeek.load_part_model` 中有针对 LoRA/Prefix 的兼容加载分支，合并专家时会尝试平均 `tokenEmbedding` 保证编码一致性。
- 在推理/评估前，确认使用的 checkpoint 类型（完整模型 / state_dict / LoRA-only），并按 `evaluate_agent` 中的加载逻辑处理。