import requests
from langchain.embeddings.base import Embeddings
from config.settings import SILICONFLOW_API_URL


class SiliconFlowEmbeddings(Embeddings):
    """SiliconFlow API 的嵌入模型实现"""

    def __init__(self, api_key: str, model_name: str = "BAAI/bge-large-zh-v1.5"):
        """初始化 SiliconFlow 嵌入模型

        Args:
            api_key: SiliconFlow API 密钥
            model_name: 要使用的模型名称
        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"{SILICONFLOW_API_URL}/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为文档列表生成嵌入向量

        Args:
            texts: 要生成嵌入的文本列表

        Returns:
            嵌入向量列表

        Raises:
            Exception: 当 API 调用失败时
        """
        embeddings = []
        for text in texts:
            data = {
                "input": text,
                "model": self.model_name,
                "encoding_format": "float"
            }
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data
            )
            if response.status_code == 200:
                result = response.json()
                embeddings.append(result["data"][0]["embedding"])
            else:
                raise Exception(
                    f"嵌入请求失败：{response.status_code} - {response.text}")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """为查询文本生成嵌入向量

        Args:
            text: 要生成嵌入的查询文本

        Returns:
            查询文本的嵌入向量

        Raises:
            Exception: 当 API 调用失败时
        """
        data = {
            "input": text,
            "model": self.model_name,
            "encoding_format": "float"
        }
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=data
        )
        if response.status_code == 200:
            result = response.json()
            return result["data"][0]["embedding"]
        else:
            raise Exception(f"嵌入请求失败：{response.status_code} - {response.text}")
