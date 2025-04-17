import os
import json
from typing import List, Tuple, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from config.settings import CHROMA_PERSIST_DIR, LOADED_DOCUMENTS_FILE, SIMILARITY_THRESHOLD, TOP_K_RESULTS


class VectorStore:
    """向量存储管理器，处理文档的存储和检索"""

    def __init__(self, embeddings: Embeddings):
        """初始化向量存储

        Args:
            embeddings: 嵌入模型实例
        """
        self.embeddings = embeddings
        self.vectorstore = None
        self.loaded_documents = []
        self._load_or_create_store()

    def _load_or_create_store(self):
        """加载现有的向量存储或创建新的"""
        print(f"\n检查知识库目录：{CHROMA_PERSIST_DIR}")
        if os.path.exists(CHROMA_PERSIST_DIR):
            print("找到已存在的知识库目录")
            try:
                from chromadb.config import Settings
                from chromadb import PersistentClient

                print("初始化 ChromaDB 客户端...")
                client = PersistentClient(
                    path=CHROMA_PERSIST_DIR,
                    settings=Settings(
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )

                print("创建 Chroma 向量存储...")
                self.vectorstore = Chroma(
                    client=client,
                    embedding_function=self.embeddings,
                )
                print("成功加载已存在的向量存储")

                # 检查集合是否存在
                collections = client.list_collections()
                print(f"发现的集合数量：{len(collections)}")
                for collection in collections:
                    print(f"集合名称：{collection.name}, 文档数量：{collection.count()}")

                self._load_document_list()
            except Exception as e:
                print(f"加载向量存储时出错：{str(e)}")
                import traceback
                print("错误详情：")
                print(traceback.format_exc())
                self.vectorstore = None
        else:
            print("知识库目录不存在，等待新文档加载...")

    def _load_document_list(self):
        """加载已处理的文档列表"""
        try:
            if os.path.exists(LOADED_DOCUMENTS_FILE):
                with open(LOADED_DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"加载文档列表时出错：{str(e)}")
            return []

    def _save_document_list(self):
        """保存已处理的文档列表"""
        try:
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            with open(LOADED_DOCUMENTS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.loaded_documents, f,
                          ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存文档列表时出错：{str(e)}")

    def add_documents(self, documents: List[str], source: str):
        """添加文档到向量存储"""
        try:
            # 添加到向量存储
            self.vectorstore.add_documents([
                Document(page_content=text, metadata={"source": source})
                for text in documents
            ])

            # 更新已加载文档列表
            if source not in self.loaded_documents:
                self.loaded_documents.append(source)
                self._save_document_list()

        except Exception as e:
            print(f"添加文档时出错：{str(e)}")
            print("错误详情：")
            import traceback
            print(traceback.format_exc())

    def search(self, query: str, k: int = TOP_K_RESULTS) -> List[Tuple[Document, float]]:
        """搜索相关文档

        Args:
            query: 搜索查询
            k: 返回结果数量

        Returns:
            文档和相似度分数的列表
        """
        if not self.vectorstore:
            print("知识库为空或未初始化")
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # 过滤低相似度结果
        filtered_results = [
            (doc, score) for doc, score in results
            if score < SIMILARITY_THRESHOLD
        ]

        if not filtered_results:
            print("\n未找到相关度足够高的内容。")
            return []

        # 去重处理
        unique_results = []
        seen_content = set()

        print("\n检索到的相关内容：")
        print("-" * 50)

        for i, (doc, score) in enumerate(filtered_results, 1):
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                unique_results.append((doc, score))
                print(f"\n片段 {i} (相似度: {score:.4f}):")
                print(content.strip())  # 使用 strip() 移除多余的空白字符
                print("-" * 50)

        return unique_results

    def print_all_segments(self):
        """打印知识库中的所有文档分段信息"""
        if not self.vectorstore:
            print("\n知识库为空或未初始化")
            return

        try:
            # 获取所有文档
            documents = self.vectorstore.get()
            if not documents['documents']:
                print("\n知识库中没有文档")
                return

            print("\n=== 知识库文档分段信息 ===")
            print(f"总文档数：{len(documents['documents'])}")
            print("=" * 50)

            for i, (doc, metadata) in enumerate(zip(documents['documents'], documents['metadatas']), 1):
                print(f"\n文档片段 {i}:")
                if metadata:
                    print(f"元数据：{metadata}")
                print(f"内容：\n{doc}")
                print("-" * 50)

        except Exception as e:
            print(f"获取知识库内容时出错：{str(e)}")

    def get_loaded_documents(self) -> List[str]:
        """获取已加载的文档列表

        Returns:
            文档路径列表
        """
        return self.loaded_documents

    def export_segments_by_source(self):
        """将知识库内容按照源文档导出到文件

        Returns:
            导出的文件夹路径，如果失败则返回 None
        """
        if not self.vectorstore:
            print("\n知识库为空或未初始化")
            return None

        try:
            # 获取所有文档
            collection = self.vectorstore._collection
            if not collection:
                print("\n知识库中没有文档")
                return None

            # 获取所有文档内容和元数据
            result = collection.get()
            if not result['documents']:
                print("\n知识库中没有文档")
                return None

            # 创建导出目录
            export_dir = os.path.join(os.path.dirname(
                CHROMA_PERSIST_DIR), "exported_segments")
            os.makedirs(export_dir, exist_ok=True)

            # 按源文档组织内容
            source_segments = {}
            for doc, metadata in zip(result['documents'], result['metadatas']):
                # 安全地处理元数据
                source = 'unknown_source'
                if metadata is not None:
                    source = metadata.get('source', 'unknown_source')

                if source not in source_segments:
                    source_segments[source] = []
                source_segments[source].append((doc, metadata))

            # 为每个源文档创建目录并导出内容
            for source, segments in source_segments.items():
                # 创建源文档对应的目录
                source_name = os.path.splitext(os.path.basename(source))[0]
                source_dir = os.path.join(export_dir, source_name)
                os.makedirs(source_dir, exist_ok=True)

                # 导出每个分段
                for i, (content, metadata) in enumerate(segments, 1):
                    segment_file = os.path.join(
                        source_dir, f"segment_{i:03d}.txt")
                    with open(segment_file, 'w', encoding='utf-8') as f:
                        f.write(f"=== 文档片段 {i} ===\n")
                        f.write(f"源文件：{source}\n")
                        if metadata is not None:  # 只在有元数据时输出
                            f.write(f"元数据：{metadata}\n")
                        f.write("\n内容：\n")
                        f.write(content)
                        f.write("\n" + "="*50 + "\n")

                print(f"\n已导出 {source_name} 的 {len(segments)} 个片段")

            print(f"\n所有文档片段已导出到：{export_dir}")
            return export_dir

        except Exception as e:
            print(f"导出文档片段时出错：{str(e)}")
            import traceback
            print("错误详情：")
            print(traceback.format_exc())
            return None
