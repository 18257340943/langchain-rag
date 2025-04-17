# LangChain RAG

基于 LangChain 实现的 RAG (检索增强生成) 项目。

## 安装说明

### 1. 克隆项目

```bash
git clone [你的仓库URL]
cd langchain-rag
```

111

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# 在 macOS/Linux 上:
source .venv/bin/activate
# 在 Windows 上:
# .venv\Scripts\activate
```

### 3. 安装依赖

在项目根目录下运行：

```bash
pip install -e .
```

这将以开发模式安装项目及其所有依赖。`-e` 参数启用"可编辑"模式，这意味着当你修改源代码时，不需要重新安装包。

### 4. 环境变量配置

创建 `.env` 文件并设置必要的环境变量：

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的配置，主要是硅基流动的 API Key
```

主要配置项包括：

- `SILICONFLOW_API_KEY`: 硅基流动的 API Key（必需）
- `EMBEDDING_MODEL`: 嵌入模型选择（可选，默认在运行时选择）
- `VECTOR_STORE_DIR`: 向量数据库存储目录（可选，默认为 vector_store）
- `SPLITS_DIR`: 文档分割信息存储目录（可选，默认为 splits_info）

## 使用方法

安装完成后，你可以通过以下命令运行：

```bash
rag
```

程序启动后会显示主菜单，提供以下功能：

1. 加载新文档 - 添加文档到知识库（支持 doc/docx/txt 格式）
2. 查看知识库内容 - 显示当前已加载的所有文档片段
3. 开始对话 - 进入交互式对话模式
4. 导出知识库内容 - 将文档片段导出到指定目录
5. 退出程序

### 嵌入模型选择

首次运行时，你需要：

1. 输入硅基流动的 API Key（如果未在环境变量中配置）
2. 选择要使用的嵌入模型：
   - BAAI/bge-large-zh-v1.5（推荐，效果最好）
   - BAAI/bge-m3（多语言支持）
   - moka-ai/m3e-base（轻量级）

## 依赖列表

项目主要依赖包括：

- langchain
- langchain-openai
- chromadb
- python-dotenv
- tiktoken
- requests
- docx2txt
- unstructured
- sentence-transformers

## 项目结构

项目使用 src 布局模式组织代码，主要模块包括：

- `chat/` - 对话相关功能
- `vectorstores/` - 向量存储实现
- `document_loaders/` - 文档加载和处理
- `embeddings/` - 文本嵌入模型
- `config/` - 配置管理

## 功能特点

- 支持多种文档格式的加载和向量化
- 基于向量相似度的智能文档检索
- 交互式对话界面
- 知识库内容管理（查看、导出）
- 支持多种嵌入模型选择
- 文档自动分段处理

## 数据存储

项目会自动创建以下目录来存储数据：

- `vector_store/`: 存储文档的向量表示和索引
- `splits_info/`: 存储文档的分割信息，包括每个片段的原始内容和位置

你可以在 `.env` 文件中自定义这些目录的位置。
