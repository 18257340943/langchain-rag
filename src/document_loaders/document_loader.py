import os
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredFileLoader
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, TEXT_SEPARATORS


class DocumentLoader:
    """文档加载器类，支持不同格式的文档加载和分割"""

    @staticmethod
    def load_and_split(file_path: str) -> List[Document]:
        """加载文档并分割成片段

        Args:
            file_path: 文档路径

        Returns:
            文档片段列表

        Raises:
            Exception: 当文件不存在或格式不支持时
        """
        if not os.path.exists(file_path):
            raise Exception(f"文件不存在：{file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()

        # 根据文件扩展名选择合适的加载器
        try:
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                splits = DocumentLoader._split_text(text)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
                doc = loader.load()
                # 确保对 .docx 文件的内容也进行分割
                text = doc[0].page_content if doc else ""
                splits = DocumentLoader._split_text(text)
            else:
                raise Exception(f"不支持的文件格式：{file_extension}")

            return splits

        except Exception as e:
            raise Exception(f"加载文档时出错：{str(e)}")

    @staticmethod
    def _split_text(text: str) -> List[Document]:
        """将文本分割成片段

        Args:
            text: 要分割的文本

        Returns:
            文档片段列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=TEXT_SEPARATORS
        )
        return text_splitter.create_documents([text])

    @staticmethod
    def save_splits_info(splits: List[Document], original_file: str) -> str:
        """保存文档分割信息到文件

        Args:
            splits: 文档片段列表
            original_file: 原始文档路径

        Returns:
            保存的文件路径
        """
        base_name = os.path.splitext(os.path.basename(original_file))[0]
        splits_file = f"{base_name}_splits.txt"

        try:
            with open(splits_file, 'w', encoding='utf-8') as f:
                f.write(f"文档名：{original_file}\n")
                f.write(
                    f"总字符数：{sum(len(split.page_content) for split in splits)}\n")
                f.write(f"总切片数：{len(splits)}\n\n")

                for i, split in enumerate(splits, 1):
                    f.write(f"=== 切片 {i} ===\n")
                    f.write(f"长度：{len(split.page_content)} 字符\n")
                    f.write(f"内容：\n{split.page_content}\n")
                    f.write("="*50 + "\n\n")

            return splits_file

        except Exception as e:
            raise Exception(f"保存切片信息时出错：{str(e)}")
