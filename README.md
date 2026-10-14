# COMP5423 Skill-Augmented Search Agent

这是课程项目的可运行代码实现，不是已完成的课程提交。真实课程语料、指定模型 API、报告、签名和视频需要小组补充。未运行官方数据时不能声明准确率或满分。

## 实现与要求对应

| 要求 | 实现 |
|---|---|
| 稀疏检索 | SQLite FTS5 的 BM25，英文 Porter 分词/词干处理 |
| 编码器稠密检索 | `intfloat/e5-small-v2`，区分 query/passage 前缀 |
| 指令感知稠密检索 | `Qwen/Qwen3-Embedding-0.6B`，查询使用检索指令 |
| 恰好一个检索工具 | `retrieve`，内部选择类别或 RRF 融合；按文档去重 |
| 辅助工具 | `open_document`、`find_in_document`、`load_skill` |
| 基础工具 Agent | 指定 API 模型通过 JSON action 自主生成工具名和参数，程序验证并执行；不是硬编码问答流程 |
| 多轮循环 | 根据先前证据、缺失条件及工具错误改写查询、检查候选、停止 |
| 上下文管理 | 独立 SearchState、证据去重、窗口选择、片段裁剪、重复调用防护 |
| 技能 | 两个原创技能，符合 Agent Skills 目录和 frontmatter 要求 |
| 自动技能选择 | 初始只暴露 name/description；模型自主调用 load_skill，也可以不加载 |
| 界面 | CLI 和 Streamlit：答案、证据、技能、动作、API tokens |
| 预测格式 | 每题 6 字段 JSONL，完整机器可读轨迹及真实 API usage |
| 评估 | normalized exact match、总工具数、总 API tokens；支持消融实验 |

默认 `config.json` 启用三类检索。`configs/sparse.json` 是明确的轻量配置，只使用一类，不声称获得三类检索分数。没有向量索引时完整配置会报错，不会偷偷退回 BM25。

## 环境

Python 3.10+；Windows PowerShell 示例，先进入本项目目录：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dense.txt
```

只有 BM25、CLI 和测试时不需要第三方包。只需要界面则安装 `requirements.txt`。
Linux/macOS 将 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python`。

本实现依赖 SQLite 的 FTS5（大多数官方 CPython 发行版已包含）。
E5 和 Qwen 的模型权重首次使用时下载到 Hugging Face 缓存，不能随提交打包。
默认 CPU 可运行但全量向量化可能较慢；有 GPU 时设置 `device` 为 `cuda`，并使用对应 CUDA 的 PyTorch 安装。
两套向量索引采用 float32，实际磁盘/内存随分块数增加。Qwen 最大长度和批大小可调，索引与查询配置必须一致。

## 模型配置

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```text
LLM_API_KEY=你的密钥
LLM_BASE_URL=课程认可服务商的 OpenAI-compatible v1 地址
LLM_MODEL=该服务商实际提供的 DeepSeek-V4-Flash 模型 ID
LLM_TIMEOUT=120
```

课程规定主模型是 DeepSeek-V4-Flash；不同服务商的模型 ID/别名可能不同。不要把其他模型填进去后声称符合要求。模型 ID 故意不预填，避免猜测错误。
接口使用 Chat Completions、`response_format=json_object`，并要求每次返回 `usage.total_tokens`。确认所选服务商支持这些功能。若需要接口适配，改 `agentsearch/llm.py`，必须保留实际用量统计。
运行轨迹中记录返回的 model、请求 messages 和 usage，不保存 Authorization/API Key。

## 数据格式

从教师提供的课程资源获取固定语料和验证/测试集。这里不猜测课程 100 题子集的 Hugging Face 地址。

`data/documents.jsonl`，一行一篇文档：

```json
{"doc_id":"123","text":"原文正文","source_url":"https://source.example/page"}
```

`data/validation.jsonl`：

```json
{"id":"q1","question":"题目原文","answer":"标准答案"}
```

`data/test.jsonl` 只有 `id` 和 `question`。程序在问答阶段主动丢弃 gold answer，只在独立 evaluate 阶段读取答案。

如果来源是 HF dataset/parquet：

```powershell
.\.venv\Scripts\python.exe -m pip install datasets
.\.venv\Scripts\python.exe scripts/export_dataset.py --dataset 教师提供的真实数据集ID --split train --kind documents --id-field doc_id --url-field url --output data/documents.jsonl
```

根据实际列名设置 `--id-field`、`--text-field`、`--url-field` 等参数。课程问题集单独导出，不能把整个原始题库当成教师指定的 100 题子集。

## 建立全量三类索引

```powershell
.\.venv\Scripts\python.exe main.py build --documents data/documents.jsonl --categories sparse dense instruction --expected-documents 100195
```

流程：建立文档库和 BM25 → 分批编码 E5 → 分批编码 Qwen → 保存 FAISS 和指纹清单。
所有索引都保留 chunk→doc→字符位置→source URL。向量构建批处理，不一次把全库文本送进模型。
已有语料库可追加建立向量索引；程序会检查输入语料哈希和配置。改变分块设置时应使用新的 index_dir。
Interrupted build 的 `corpus.building.sqlite` 会保留以供检查，不会自动覆盖。

## 交互和界面

```powershell
.\.venv\Scripts\python.exe main.py ask "输入你的问题" --output outputs/interactive.json
.\.venv\Scripts\python.exe -m streamlit run app.py
```

界面在问题完成后展示真实步骤和证据；不把模拟输出呈现成实时模型答案。

## 生成课程预测

```powershell
.\.venv\Scripts\python.exe main.py batch --questions data/test.jsonl --output outputs/test_prediction.jsonl
.\.venv\Scripts\python.exe main.py validate --predictions outputs/test_prediction.jsonl --questions data/test.jsonl --course
```

每条记录包含且仅包含：`id`、`question`、`answer`、`trajectory`、`tool_calls`、`total_tokens`。
`trajectory` 记录启动配置、API 请求/响应、决策、工具参数与完整返回、验证及最终证据。只记录模型实际返回的过程，不编造隐藏思维过程。
`tool_calls` 包含 Agent 发出的工具请求，包括失败或重复的请求；不统计检索器内部向量计算、数据库操作。
`total_tokens` 累加所有实际 API 调用（包括技能后的决策、格式修正和答案验证）的 `usage.total_tokens`，不是字符数估算。
最终 answer 仅给问题要求的姓名、日期、值等，证据放在轨迹中，避免额外解释影响 EM。

默认需要恰好 100 道题。完成前写 `.partial` checkpoint；全部成功后才发布正式 JSONL。
完整成功问题可以在题与题之间的干净中断后用同样命令加 `--resume` 继续，配置、语料和模型必须一致。若在一题执行中中断，inflight 标记会阻止继续，避免丢弃已产生但未保存的 API 用量。
遇到 API/预算失败会中止并保存 `.error.json`，不会为失败题填充假的成功记录。失败运行不能偷偷略去 API 用量后 resume；修复后使用新的输出路径，保留旧运行记录。
不自动重试可能已计费但未收到结果的请求。token 预算是逐次调用前检查的软上限，最后一次 API 调用可能超过剩余预算。
`--course` 校验检查真实 API 标识、100,195 文档、100 条完整记录、计数一致性；无法证明模型供应商身份或防止人为伪造，需要教师认可的服务商和真实可复现实验。

## 评估与消融

```powershell
.\.venv\Scripts\python.exe main.py batch --questions data/validation.jsonl --output outputs/validation_prediction.jsonl
.\.venv\Scripts\python.exe main.py evaluate --predictions outputs/validation_prediction.jsonl --gold data/validation.jsonl
.\.venv\Scripts\python.exe scripts/ablation.py --questions data/validation.jsonl --output-dir outputs/ablations
```

消融包括 single RAG、无技能 Agent、无上下文管理 Agent、完整 Agent、仅 sparse、sparse+dense。所有配置调用相同主模型；运行全部配置会产生真实 API 费用。
单轮基线明确标注 mode=single，不能当作课程要求的工具选择 Agent 提交。
评估使用小写、去 ASCII 标点、去英文冠词、归一化空格后的精确匹配；没有关键词命中率代替准确率。兼容 gold `answer` 或 `answers` 列。
排名得分需要全班结果，本程序不虚构排名或报告预计得分。

## 离线测试和示例

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe --version
```

测试用客户端只存在于 `tests/`，CLI/UI 不暴露 FakeLLM。测试验证算法、状态、错误处理和计数；不证明真实模型准确率。
`examples/` 是三个明确标注的虚构文档和一道虚构题，只供演示数据格式。
可以复制 sparse 配置并将 index_dir 改成 `indexes-demo`，然后：

```powershell
.\.venv\Scripts\python.exe main.py --config configs/demo.json build --documents examples/documents.jsonl --categories sparse --expected-documents 3
.\.venv\Scripts\python.exe main.py --config configs/demo.json ask "Who founded the Lantern Archive in 2007 and received the Silver Quill Award in 2018?"
```

这仍需要真实 API；三个示例文档绝不能冒充课程 100,195 篇文档。

## 代码结构

```text
agentsearch/
  config.py        配置与环境变量
  data.py          JSONL、字段转换、分块
  retrieval.py     BM25、E5、Qwen、FAISS、RRF、文档工具
  llm.py           真实 API 和 usage 检查
  state.py         结构化状态、去重、上下文裁剪
  skills.py        元信息发现与按需读取
  agent.py         主循环与候选验证
  evaluation.py    EM 和预测格式校验
  cli.py           建索引、问答、批量和评估入口
skills/            两个原创运行时技能
scripts/           数据导出、消融、打包
tests/             离线回归与真实 FAISS 集成检查（可选依赖）
app.py             Streamlit UI
main.py            CLI 入口
```

## 打包与仍需完成的交付物

```powershell
.\.venv\Scripts\python.exe scripts/package_code.py
```

此脚本只生成 `code/` 源码 ZIP，排除密钥、模型、索引、环境和运行结果，不是最终 Group-X.zip。
小组还必须自己运行官方测试集，加入 `test_prediction.jsonl`，按指定 Overleaf 模板撰写至少四页报告，补全所有成员签名和总计 100% 的贡献声明，并录制不超过五分钟的 MP4 视频。
最终包结构：`test_prediction.jsonl`、`code/`、`Group-X-Report.pdf`、`Group-X-Video.mp4`。
代码只是实现模板；理解、修改、原创技能认定及 AI 使用披露应遵循课程规则，不应直接声称已满足教师的所有验收。

## 设计参考（实现接口依据）

- DeepSeek Chat Completions: https://api-docs.deepseek.com/api/create-chat-completion/
- E5 model card: https://huggingface.co/intfloat/e5-small-v2
- Qwen embedding model card: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Agent Skills specification: https://agentskills.io/specification

两份技能正文为本项目新编写，没有复制现成技能。检索依赖使用其公开接口；课程要求的主 LLM 与本地检索编码模型是不同组件。
