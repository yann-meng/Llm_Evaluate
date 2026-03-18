# 配置说明（`run_config.yaml`）

本文档覆盖本框架所有核心配置项，适用于 LLM / VLM、单数据集与多数据集评测。

## 1. 最小配置

```yaml
run_name: demo_local
output_dir: outputs

dataset:
  dataset_id: local_qa
  source: local
  path: examples/data/custom_eval.jsonl
  input_column: prompt
  answer_column: answer

model:
  backend: openai_compatible
  task_type: llm
  model_name_or_path: Qwen/Qwen2.5-7B-Instruct
  api_base: http://127.0.0.1:8000/v1

metrics:
  names: [exact_match, token_f1, rouge_l, char_f1]
```

## 2. 顶层参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `run_name` | `str` | `default_run` | 本次评测任务名 |
| `output_dir` | `str` | `outputs` | 输出目录根路径 |
| `batch_size` | `int` | `1` | 预留字段（当前按样本顺序推理） |
| `dataset` | `object` | `null` | 单数据集配置 |
| `datasets` | `list[object]` | `[]` | 多数据集配置（推荐） |
| `model` | `object` | 必填 | 模型与推理后端参数 |
| `metrics` | `object` | `exact_match` | 指标配置 |

> `dataset` 与 `datasets` 至少配置一个。若同时存在，优先使用 `datasets`。

## 3. 数据集配置（`dataset` / `datasets[*]`）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `dataset_id` | `str` | `default_dataset` | 数据集标识，用于分目录输出 |
| `source` | `huggingface/modelscope/local` | 必填 | 数据来源 |
| `name` | `str` | `null` | HF / ModelScope 数据集名 |
| `subset` | `str` | `null` | 子集名 |
| `split` | `str` | `test` | 评测 split |
| `path` | `str` | `null` | 本地路径（jsonl/csv/parquet 文件，或 parquet 分片目录） |
| `input_column` | `str` | `prompt` | 输入字段 |
| `answer_column` | `str` | `answer` | 标准答案字段 |
| `image_column` | `str` | `null` | 图像字段（VLM） |
| `limit` | `int` | `null` | 抽样条数 |

## 4. 模型配置（`model`）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `backend` | `transformers/openai_compatible/vllm/sglang` | 必填 | 推理后端 |
| `task_type` | `llm/vlm` | `llm` | 任务类型 |
| `model_source` | `local/huggingface/modelscope` | `huggingface` | 模型来源标记 |
| `model_name_or_path` | `str` | `""` | 模型名或本地路径 |
| `api_base` | `str` | `null` | OpenAI 协议地址（`openai_compatible/vllm/sglang`） |
| `api_key` | `str` | `null` | API Key（可选） |
| `trust_remote_code` | `bool` | `false` | Transformers 是否允许远程代码 |
| `torch_dtype` | `auto/float16/bfloat16/float32` | `auto` | Transformers dtype |
| `device` | `str/int` | `null` | 指定单设备，如 `0`、`cuda:0` |
| `device_map` | `str` | `null` | Transformers 多 GPU，如 `auto` |
| `num_gpus` | `int` | `1` | GPU 数量声明 |
| `tensor_parallel_size` | `int` | `1` | 张量并行大小（vLLM / SGLang 服务侧） |
| `gpu_memory_utilization` | `float` | `0.9` | 显存利用率（服务侧） |
| `max_model_len` | `int` | `null` | 最大上下文（服务侧） |
| `max_new_tokens` | `int` | `256` | 生成长度 |
| `temperature` | `float` | `0.0` | 温度 |

## 5. 指标配置（`metrics`）

| 指标名 | 说明 |
|---|---|
| `exact_match` | 归一化后完全匹配 |
| `token_f1` | token 粒度 F1 |
| `rouge_l` | ROUGE-L F1（基于 LCS） |
| `char_f1` | 字符粒度 F1 |

## 6. 多数据集并行评测示例

参见 `configs/demo_multi_dataset.yaml`。

运行后会生成：

- `outputs/<run_id>/<dataset_id>/predictions.jsonl`
- `outputs/<run_id>/<dataset_id>/metrics.json`
- `outputs/<run_id>/summary_metrics.json`
- `outputs/<run_id>/summary_metrics.html`（可视化对比）


## 7. 本地 Transformers 评测示例（Qwen3-4B-Instruct + SQuAD）

参见：`configs/demo_local_transformers_qwen3_squad.yaml`

该示例使用常用测试集 `squad` 的 `validation` split，模型路径指向本地目录（可替换为你机器上的真实路径）。

## 8. 多 GPU 与超大模型建议
=======
## 7. 多 GPU 与超大模型建议


### Transformers 本地推理（单模型切分到多卡）

```yaml
model:
  backend: transformers
  task_type: llm
  model_name_or_path: /models/Qwen2.5-72B-Instruct
  torch_dtype: bfloat16
  device_map: auto
  max_new_tokens: 256
```

### vLLM / SGLang 服务推理（推荐大模型）

在服务端启动时配置 `--tensor-parallel-size`、`--gpu-memory-utilization` 等参数；框架端通过 `api_base` 调用。


## 9. 一键启动脚本

- vLLM: `scripts/serve_vllm_qwen3_4b.sh`
- SGLang: `scripts/serve_sglang_qwen3_4b.sh`
=======
