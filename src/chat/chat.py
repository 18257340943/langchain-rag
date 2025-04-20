import requests
from typing import List, Dict, Optional
from vectorstores.vector_store import VectorStore
from config.settings import (
    SILICONFLOW_API_URL,
    CHAT_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    FREQUENCY_PENALTY
)


class Chat:
    """聊天模块，处理对话和知识库检索"""

    def __init__(self, api_key: str, vector_store: VectorStore):
        """初始化聊天模块

        Args:
            api_key: API密钥
            vector_store: 向量存储实例
        """
        self.api_key = api_key
        self.api_url = f"{SILICONFLOW_API_URL}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.vector_store = vector_store
        self.conversation_history: List[Dict[str, str]] = []
        self.current_teacher: Optional[str] = None  # 当前选择的教师

    def set_teacher(self, teacher: str):
        """设置当前对话的教师

        Args:
            teacher: 教师名称
        """
        self.current_teacher = teacher
        print(f"\n已切换到 {teacher} 的知识库")

    def chat(self, message: str) -> str:
        """处理用户消息并返回回复

        Args:
            message: 用户消息

        Returns:
            助手回复
        """
        # 添加用户消息到历史记录
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # 从知识库检索相关内容
        context = self._get_context(message)

        # 准备对话请求
        data = {
            "model": CHAT_MODEL,
            "stream": False,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "frequency_penalty": FREQUENCY_PENALTY,
            "n": 1,
            "stop": [],
            "messages": self.conversation_history + [{"role": "user", "content": context or message}]
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data
            )

            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]

                # 添加助手回复到历史记录
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })

                return assistant_message
            else:
                return f"错误：{response.status_code} - {response.text}"

        except Exception as e:
            return f"发生错误：{str(e)}"

    def _get_context(self, query: str) -> Optional[str]:
        """从知识库获取相关上下文

        Args:
            query: 用户查询

        Returns:
            格式化的上下文字符串，如果没有相关内容则返回None
        """
        if not self.vector_store.vectorstore:
            print("\n提示：知识库为空，将使用模型的通用知识回答")
            return None

        # 从知识库检索相关内容
        docs = self.vector_store.search(query, teacher=self.current_teacher)

        # 如果没有找到相关内容，返回 None
        if not docs:
            print("\n提示：未找到相关内容，将使用模型的通用知识回答")
            return None

        # 直接使用检索到的文档内容
        context_parts = []
        for doc, score in docs:
            context_parts.append(doc.page_content.strip())

        context = "\n\n".join(context_parts)
        return f"参考信息：\n{context}\n\n请基于以上信息回答：{query}"
