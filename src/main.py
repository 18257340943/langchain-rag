import os
from config.settings import SILICONFLOW_API_KEY, EMBEDDING_MODELS
from embeddings.silicon_flow import SiliconFlowEmbeddings
from document_loaders.document_loader import DocumentLoader
from vectorstores.vector_store import VectorStore
from chat.chat import Chat


def get_api_key() -> str:
    """获取API密钥"""
    api_key = SILICONFLOW_API_KEY
    if not api_key:
        api_key = input("请输入硅基流动的API密钥：")
    return api_key


def select_model() -> str:
    """选择嵌入模型"""
    print("\n请选择嵌入模型：")
    print("1. BAAI/bge-large-zh-v1.5 (推荐，效果最好)")
    print("2. BAAI/bge-m3 (多语言支持)")
    print("3. moka-ai/m3e-base (轻量级)")

    choice = input("请输入选项（1-3）：")
    return EMBEDDING_MODELS.get(choice, EMBEDDING_MODELS["1"])


def load_documents(vector_store: VectorStore):
    """加载文档到知识库"""
    print("\n请输入要加载的文档路径（支持doc/docx/txt格式）")
    print("输入完成后，请按回车键，然后输入 'done' 结束输入")

    while True:
        file_path = input("\n请输入文档路径（或输入 'done' 结束）：").strip()
        if file_path.lower() == 'done':
            break

        if file_path in vector_store.get_loaded_documents():
            print(f"提示：文档 {file_path} 已经存在于知识库中，请重新输入其他文档路径")
            continue

        try:
            # 加载和分割文档
            splits = DocumentLoader.load_and_split(file_path)

            # 保存分割信息
            splits_file = DocumentLoader.save_splits_info(splits, file_path)
            print(f"切片信息已保存到：{splits_file}")

            # 添加到向量存储
            vector_store.add_documents(splits, file_path)

        except Exception as e:
            print(f"处理文档时出错：{str(e)}")


def chat_loop(chat: Chat):
    """交互式对话循环"""
    print("\n欢迎使用聊天系统！输入'退出'结束对话。")

    while True:
        message = input("\n请输入你的消息：")
        if message.lower() == '退出':
            break

        response = chat.chat(message)
        print(f"\n回答：{response}")


def main():
    """主程序入口"""
    # 获取API密钥
    api_key = get_api_key()

    # 选择嵌入模型
    model_name = select_model()

    # 初始化组件
    embeddings = SiliconFlowEmbeddings(api_key, model_name)
    vector_store = VectorStore(embeddings)
    chat = Chat(api_key, vector_store)

    while True:
        print("\n=== 主菜单 ===")
        print("1. 加载新文档")
        print("2. 查看知识库内容")
        print("3. 开始对话")
        print("4. 导出知识库内容")
        print("5. 退出程序")

        choice = input("\n请选择操作（1-5）：").strip()

        if choice == "1":
            load_documents(vector_store)
        elif choice == "2":
            vector_store.print_all_segments()
        elif choice == "3":
            chat_loop(chat)
        elif choice == "4":
            export_dir = vector_store.export_segments_by_source()
            if export_dir:
                print(f"\n文档片段已按源文件分类导出到各自的目录中")
                print(f"导出目录：{export_dir}")
        elif choice == "5":
            print("\n感谢使用！再见！")
            break
        else:
            print("\n无效的选择，请重试。")


if __name__ == "__main__":
    main()
