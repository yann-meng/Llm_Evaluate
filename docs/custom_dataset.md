# 自定义数据适配模板

## 支持格式

- `jsonl`: 每行一个 JSON 对象（推荐）
- `csv`: 列名映射

## 统一字段

| 字段 | 必需 | 说明 |
|---|---|---|
| `id` | 否 | 样本唯一标识，缺失时自动使用行号 |
| `prompt` | 是 | 模型输入文本 |
| `answer` | 否 | 参考答案（用于自动打分） |
| `image` | 否 | 图像路径或 URL（VLM 任务） |
| `metadata` | 否 | 业务附加信息 |

## 配置映射示例

```yaml
dataset:
  source: local
  path: examples/data/custom_eval.jsonl
  input_column: prompt
  answer_column: answer
  image_column: image
```

若你的数据字段名不同（比如 `question` / `gold`），直接改映射：

```yaml
dataset:
  input_column: question
  answer_column: gold
```
