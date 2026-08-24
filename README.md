# StyleRAG - 服装智能客服

基于 **高级 RAG 流水线** 的服装领域智能客服系统，具备多阶段检索、知识图谱增强和自动化评估能力。

## 核心功能

- **Query Rewriting**：将用户口语化问题改写为多个精确检索查询，提升召回率
- **Hybrid Search**：语义检索（ChromaDB）+ BM25 关键词检索，RRF 融合排序
- **Reranking**：交叉编码器（Cross-Encoder）对检索结果精细重排序
- **知识图谱**：基于 NetworkX 构建面料-季节-洗护关系图谱，补充上下文推理
- **流式输出**：支持逐 token 流式响应
- **多轮对话**：基于文件的聊天历史持久化
- **检索来源展示**：回答下方展示参考文档及来源
- **反馈机制**：用户可对回答点赞/点踩
- **评估体系**：15 条测试用例，基础 RAG vs 高级 RAG 对比评估

## 技术架构

```
用户问题
  │
  ▼
Query Rewriting（LLM 生成 3 个查询变体）
  │
  ▼
Hybrid Search（语义检索 + BM25，每路召回 top-5）
  │
  ▼
RRF 融合去重（合并为 top-10）
  │
  ▼
Reranking（交叉编码器精排，取 top-3）
  │
  ▼
Prompt Template + LLM 生成回答（流式输出）
```

## 技术栈

| 层级 | 技术 |
|---|---|
| 语言 | Python 3.10+ |
| LLM 框架 | LangChain |
| 大模型 | 通义千问 (qwen-plus) |
| Embedding | DashScope text-embedding-v1 |
| 向量数据库 | ChromaDB |
| 关键词检索 | BM25 (rank-bm25) |
| 重排序 | BAAI/bge-reranker-base |
| 知识图谱 | NetworkX |
| Web UI | Streamlit |
| 配置管理 | pydantic-settings |
| 包管理 | uv / pip |

## 项目结构

```
Style-RAG/
├── app.py                          # Streamlit 主入口
├── src/
│   ├── config.py                   # pydantic-settings 配置
│   ├── logging_config.py           # 日志系统
│   ├── rag/
│   │   ├── chain.py                # 基础 RAG 链
│   │   ├── advanced_chain.py       # 高级 RAG 流水线
│   │   ├── rewriting.py            # Query Rewriting
│   │   ├── hybrid.py               # Hybrid Search + RRF
│   │   └── reranker.py             # 交叉编码器重排序
│   ├── knowledge/
│   │   ├── base.py                 # 知识库管理
│   │   ├── graph.py                # 服装知识图谱
│   │   └── loader.py               # 文档加载器
│   ├── ui/
│   │   ├── app_qa.py               # 聊天界面
│   │   ├── app_uploader.py         # 知识库上传界面
│   │   └── history.py              # 聊天历史持久化
│   └── eval/
│       ├── dataset.py              # 测试数据集（15条）
│       └── run.py                  # 评估运行器
├── data/                           # 知识库源文件
│   ├── 尺码推荐.txt
│   ├── 洗涤养护.txt
│   └── 颜色选择.txt
├── tests/
│   └── test_basic.py               # 单元测试
├── pyproject.toml                  # 依赖配置
├── .env.example                    # 环境变量模板
└── .gitignore
```

## 快速开始

### 1. 环境准备

```bash
git clone git@github.com:xiaocaos/Style-RAG.git
cd Style-RAG
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e ".[all]"
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DashScope API Key
```

### 3. 启动应用

```bash
uv run python -m streamlit run app.py
```

访问 http://localhost:8501

## 评估

```bash
# 运行评估（基础 RAG vs 高级 RAG 对比）
python -m src.eval.run

# 运行单元测试
pytest tests/
```

## 与 SmartSweep-Agent 的区别

| 项目 | 定位 | 核心技术 |
|---|---|---|
| **SmartSweep-Agent** | Agent 智能客服 | ReAct Agent、Tool Calling、LangGraph、动态提示词、可观测性 |
| **StyleRAG** | RAG 深度优化 | Query Rewriting、Hybrid Search、Reranking、知识图谱、RAGAS 评估 |

两个项目互补：一个展示 Agent 开发能力，一个展示 RAG 工程深度。

## License

MIT
