from setuptools import setup, find_packages

setup(
    name="langchain-rag",
    version="0.1.0",
    package_dir={"": "src"},  # 告诉 setuptools src 目录包含包
    packages=find_packages(where="src"),  # 在 src 目录下查找包
    install_requires=[
        "langchain",
        "langchain-openai",
        "chromadb",
        "python-dotenv",
        "tiktoken",
        "requests",
        "docx2txt",
        "unstructured",
        "sentence-transformers"
    ],
    entry_points={
        'console_scripts': [
            'rag=main:main',  # 因为 src 是包的根目录，所以这里不需要 src. 前缀
        ],
    }
)
