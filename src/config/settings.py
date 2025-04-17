import os
from dotenv import load_dotenv

# 获取项目根目录路径
ROOT_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量
load_dotenv()

# API 配置
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1"

# 向量存储配置
CHROMA_PERSIST_DIR = os.path.join(ROOT_DIR, "chroma_db")
LOADED_DOCUMENTS_FILE = os.path.join(
    CHROMA_PERSIST_DIR, "loaded_documents.json")

# 文本分割配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TEXT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# 模型配置
EMBEDDING_MODELS = {
    "1": "BAAI/bge-large-zh-v1.5",
    "2": "BAAI/bge-m3",
    "3": "moka-ai/m3e-base"
}

CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 检索配置
SIMILARITY_THRESHOLD = 0.7
TOP_K_RESULTS = 5

# 生成配置
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.7
TOP_K = 50
FREQUENCY_PENALTY = 0.5
