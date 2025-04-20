import os
from dotenv import load_dotenv
from config.settings import SILICONFLOW_API_KEY, EMBEDDING_MODELS
from embeddings.silicon_flow import SiliconFlowEmbeddings
from document_loaders.document_loader import DocumentLoader
from vectorstores.vector_store import VectorStore
from chat.chat import Chat
from teacher.teacher_manager import TeacherManager


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


def load_documents(vector_store: VectorStore, teacher: str):
    """加载文档到知识库

    Args:
        vector_store: 向量存储实例
        teacher: 教师名称
    """
    print(f"\n正在为 {teacher} 加载文档")
    print("请输入要加载的文档路径（支持doc/docx/txt格式）")
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

            # 添加到向量存储，指定教师
            vector_store.add_documents(splits, file_path, teacher)
            print(f"已将文档添加到 {teacher} 的知识库")

        except Exception as e:
            print(f"处理文档时出错：{str(e)}")


def chat_loop(chat: Chat):
    """交互式对话循环"""
    print("\n=== 对话模式 ===")
    print("输入问题开始对话")
    print("输入 'q'、'quit' 或 '退出' 可以返回主菜单")
    print("-" * 30)

    while True:
        message = input("\n请输入你的问题（输入 q 退出）：").strip()
        if message.lower() in ['q', 'quit', '退出']:
            print("\n正在返回主菜单...")
            break

        if not message:
            continue

        response = chat.chat(message)
        print(f"\n回答：{response}")


def main():
    # 加载环境变量
    load_dotenv()
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("错误：未设置 SILICONFLOW_API_KEY 环境变量")
        return

    # 初始化教师管理器
    teacher_manager = TeacherManager()

    # 如果没有教师，添加默认教师
    if not teacher_manager.teachers:
        teacher_manager.add_teacher("李老师")
        teacher_manager.add_teacher("王老师")

    # 选择教师
    selected_teacher = teacher_manager.select_teacher()
    if not selected_teacher:
        print("未选择教师，程序退出")
        return

    # 初始化向量存储
    embeddings = SiliconFlowEmbeddings(api_key, select_model())
    vector_store = VectorStore(embeddings)

    # 初始化聊天模块
    chat = Chat(api_key, vector_store)
    chat.set_teacher(selected_teacher)  # 设置当前教师

    while True:
        print("\n=== 主菜单 ===")
        print("1. 加载新文档")
        print("2. 查看知识库内容")
        print("3. 开始对话")
        print("4. 导出知识库内容")
        print("5. 切换教师")
        print("6. 退出程序")

        choice = input("\n请选择操作（1-6）：").strip()

        if choice == "1":
            load_documents(vector_store, selected_teacher)
        elif choice == "2":
            vector_store.print_all_segments(selected_teacher)
        elif choice == "3":
            chat_loop(chat)
        elif choice == "4":
            export_dir = vector_store.export_segments_by_source()
            if export_dir:
                print(f"\n文档片段已按源文件分类导出到各自的目录中")
                print(f"导出目录：{export_dir}")
        elif choice == "5":
            new_teacher = teacher_manager.select_teacher()
            if new_teacher:
                selected_teacher = new_teacher
                chat.set_teacher(selected_teacher)
        elif choice == "6":
            print("\n感谢使用！再见！")
            break
        else:
            print("\n无效的选择，请重试。")


if __name__ == "__main__":
    main()
