# LLM Evaluate Framework

一个面向工程落地的大模型评测框架，支持：

- 主流开源 **LLM / VLM** 模型调用（Transformers / vLLM / SGLang / OpenAI-compatible）。
- 一键接入开源数据集（Hugging Face / ModelScope / 本地数据）。
- 自定义数据评测模板（文本、多模态统一 schema）。
- 多数据集单次评测、结果独立保存与可视化对比。

## 1. 架构设计

```text
┌──────────────────────────────┐
│ CLI / Config (YAML + Typer)  │
└──────────────┬───────────────┘
               │
     ┌─────────▼─────────┐
     │  Evaluation Engine │
     │  - task executor   │
     │  - batch runner    │
     │  - metrics manager │
     └───────┬──────┬─────┘
             │      │
 ┌───────────▼───┐  ▼────────────────┐
 │ Model Layer   │  Dataset Layer    │
 │ - LLM adapter │  - HF loader      │
 │ - VLM adapter │  - ModelScope     │
 │ - OpenAI API  │  - Local JSONL/CSV│
 └───────────────┘  └────────────────┘
```

## 2. 快速开始

### 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,all]'
```

### 运行示例

```bash
llm-evaluate run --config configs/demo_hf_qna.yaml
llm-evaluate run --config configs/demo_multi_dataset.yaml

llm-evaluate run --config configs/demo_local_transformers_qwen3_squad.yaml

```

输出位于 `outputs/<run_id>/`：

- `<dataset_id>/predictions.jsonl`：逐样本预测
- `<dataset_id>/metrics.json`：数据集级指标
- `summary_metrics.json`：多数据集汇总
- `summary_metrics.html`：可视化对比页面
- `run_config.snapshot.yaml`：配置快照

## 3. 支持的数据源

| 数据源 | 用法 | 说明 |
|---|---|---|
| Hugging Face | `source: huggingface` | 使用 `datasets.load_dataset` |
| ModelScope | `source: modelscope` | 使用 `MsDataset.load` |
| Local | `source: local` | JSONL/CSV 本地文件 |

## 4. 支持的模型后端

| 后端 | 任务类型 | 说明 |
|---|---|---|
| `transformers` | llm/vlm | 本地推理，支持纯文本与图文 |
| `openai_compatible` | llm/vlm | 对接 vLLM/TGI/ollama/自建 OpenAI 协议服务 |
| `vllm` | llm/vlm | 对接 vLLM OpenAI 协议服务 |
| `sglang` | llm/vlm | 对接 SGLang OpenAI 协议服务 |

## 5. 自定义数据模板

见 `src/llm_evaluate/templates/custom_dataset_template.jsonl`、`examples/data/*.jsonl` 与 `docs/custom_dataset.md`。

统一字段：

- `id`: 样本 id
- `prompt`: 文本输入
- `image`: 可选，图像路径/URL（VLM）
- `answer`: 参考答案（可选，纯推理可为空）
- `metadata`: 可选附加信息

## 6. 目录结构

```text
src/llm_evaluate/
  cli/               # CLI
  data/              # dataset loader + schema mapping
  models/            # llm/vlm adapters
  metrics/           # metric implementations
  eval/              # evaluation engine
  templates/         # custom data templates
```

## 7. 配置文档

详见 `docs/configuration.md`（包含多 GPU、超大模型、多数据集等详细说明）。


## 8. 推理服务启动脚本

- vLLM（Qwen3-4B-Instruct）: `scripts/serve_vllm_qwen3_4b.sh`
- SGLang（Qwen3-4B-Instruct）: `scripts/serve_sglang_qwen3_4b.sh`

可直接执行脚本，或通过环境变量覆盖 host/port/并行参数。
