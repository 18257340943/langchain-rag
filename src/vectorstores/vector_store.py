import os
import json
from typing import List, Tuple, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from config.settings import CHROMA_PERSIST_DIR, LOADED_DOCUMENTS_FILE, SIMILARITY_THRESHOLD, TOP_K_RESULTS
from chromadb import PersistentClient
from chromadb.config import Settings


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
                print("开始初始化 ChromaDB 客户端...")
                client = PersistentClient(
                    path=CHROMA_PERSIST_DIR,
                    settings=Settings(
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
                print("ChromaDB 客户端初始化完成")

                print("开始创建 Chroma 向量存储...")
                self.vectorstore = Chroma(
                    client=client,
                    embedding_function=self.embeddings,
                )
                print("Chroma 向量存储创建完成")

                # 获取所有集合
                print("开始获取集合信息...")
                collections = client.list_collections()
                print(f"发现的集合数量：{len(collections)}")

                if len(collections) > 0:
                    for collection in collections:
                        print(f"\n集合名称：{collection.name}")
                        print(f"文档数量：{collection.count()}")

                        # 获取集合中的所有数据
                        print(f"开始获取集合 {collection.name} 的数据...")
                        data = collection.get()
                        print(f"获取到 {len(data.get('documents', []))} 个文档")

                        for i, (doc, metadata) in enumerate(zip(data.get('documents', []), data.get('metadatas', [])), 1):
                            if i <= 3:  # 只显示前3个文档
                                print(f"\n文档 {i}:")
                                print(f"内容: {doc[:100]}...")  # 显示前100个字符
                                print(f"元数据: {metadata}")
                else:
                    print("没有找到任何集合")

                print("开始加载文档列表...")
                self._load_document_list()
                print("文档列表加载完成")

            except Exception as e:
                print(f"加载向量存储时出错：{str(e)}")
                import traceback
                print("错误详情：")
                print(traceback.format_exc())
                self.vectorstore = None
        else:
            print("知识库目录不存在，创建新的向量存储...")
            try:
                # 创建目录
                os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

                # 初始化新的客户端和向量存储
                client = PersistentClient(
                    path=CHROMA_PERSIST_DIR,
                    settings=Settings(
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
                self.vectorstore = Chroma(
                    client=client,
                    embedding_function=self.embeddings,
                )
                print("新的向量存储创建完成")

                # 初始化空的文档列表
                self.loaded_documents = []
                self._save_document_list()

            except Exception as e:
                print(f"创建新向量存储时出错：{str(e)}")
                import traceback
                print("错误详情：")
                print(traceback.format_exc())
                self.vectorstore = None

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

    def add_documents(self, documents: List[str], source: str, teacher: str):
        """添加文档到向量存储

        Args:
            documents: 文档内容列表
            source: 文档来源/路径
            teacher: 教师名称
        """
        try:
            # 检查文档是否已存在
            if source in self.loaded_documents:
                print(f"\n文档 {source} 已存在于知识库中，跳过加载...")
                return

            # 添加到向量存储
            docs_to_add = []
            for text in documents:
                # 如果输入已经是 Document 对象，直接使用其 page_content
                if isinstance(text, Document):
                    doc = Document(
                        page_content=text.page_content,
                        metadata={
                            "source": source,
                            "teacher": teacher
                        }
                    )
                else:
                    doc = Document(
                        page_content=str(text),
                        metadata={
                            "source": source,
                            "teacher": teacher
                        }
                    )
                docs_to_add.append(doc)

            self.vectorstore.add_documents(docs_to_add)

            # 更新已加载文档列表
            self.loaded_documents.append(source)
            self._save_document_list()

        except Exception as e:
            print(f"添加文档时出错：{str(e)}")
            print("错误详情：")
            import traceback
            print(traceback.format_exc())

    def search(self, query: str, teacher: Optional[str] = None, k: int = TOP_K_RESULTS) -> List[Tuple[Document, float]]:
        """搜索相关文档

        Args:
            query: 搜索查询
            teacher: 教师名称，如果指定则只搜索该教师的文档
            k: 返回结果数量

        Returns:
            文档和相似度分数的列表
        """
        if not self.vectorstore:
            print("知识库为空或未初始化")
            return []

        # 如果指定了教师，使用 where 过滤
        if teacher:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter={"teacher": teacher}
            )
        else:
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
                print(f"教师: {doc.metadata.get('teacher', '未知')}")
                print(content.strip())  # 使用 strip() 移除多余的空白字符
                print("-" * 50)

        return unique_results

    def print_all_segments(self, teacher: Optional[str] = None):
        """打印所有文档片段

        Args:
            teacher: 教师名称，如果指定则只显示该教师的文档
        """
        if not self.vectorstore:
            print("知识库为空或未初始化")
            return

        try:
            # 获取所有文档
            collection = self.vectorstore._collection
            if not collection:
                print("知识库中没有文档")
                return

            # 获取所有文档内容和元数据
            result = collection.get()
            if not result['documents']:
                print("知识库中没有文档")
                return

            # 过滤指定教师的文档
            filtered_docs = []
            for doc, metadata in zip(result['documents'], result['metadatas']):
                if teacher and metadata.get('teacher') != teacher:
                    continue
                filtered_docs.append((doc, metadata))

            if not filtered_docs:
                if teacher:
                    print(f"\n{teacher}的知识库中没有文档")
                else:
                    print("\n知识库中没有文档")
                return

            print("\n=== 知识库文档分段信息 ===")
            print(f"总文档数：{len(filtered_docs)}")
            print("=" * 50)

            for i, (content, metadata) in enumerate(filtered_docs, 1):
                print(f"\n文档片段 {i}:")
                print(f"元数据：{metadata}")
                print("内容：")
                print(content.strip())  # 使用 strip() 移除多余的空白字符
                print("=" * 50)

        except Exception as e:
            print(f"获取知识库内容时出错：{str(e)}")
            import traceback
            print("错误详情：")
            print(traceback.format_exc())

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
