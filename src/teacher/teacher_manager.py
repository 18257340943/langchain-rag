from typing import List, Optional
import json
import os


class TeacherManager:
    """教师管理类，处理教师选择和验证"""

    def __init__(self, config_file: str = "config/teachers.json"):
        """初始化教师管理器

        Args:
            config_file: 教师配置文件路径
        """
        self.config_file = config_file
        self.teachers = self._load_teachers()
        self.current_teacher: Optional[str] = None

    def _load_teachers(self) -> List[str]:
        """从配置文件加载教师列表"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('teachers', [])
            return []
        except Exception as e:
            print(f"加载教师列表时出错：{str(e)}")
            return []

    def _save_teachers(self):
        """保存教师列表到配置文件"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({'teachers': self.teachers}, f,
                          ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存教师列表时出错：{str(e)}")

    def add_teacher(self, name: str):
        """添加教师

        Args:
            name: 教师名称
        """
        if name not in self.teachers:
            self.teachers.append(name)
            self._save_teachers()
            print(f"\n已添加教师：{name}")
        else:
            print(f"\n教师 {name} 已存在")

    def remove_teacher(self, name: str):
        """移除教师

        Args:
            name: 教师名称
        """
        if name in self.teachers:
            self.teachers.remove(name)
            self._save_teachers()
            print(f"\n已移除教师：{name}")
        else:
            print(f"\n教师 {name} 不存在")

    def list_teachers(self):
        """列出所有教师"""
        if not self.teachers:
            print("\n当前没有教师")
            return

        print("\n教师列表：")
        for i, teacher in enumerate(self.teachers, 1):
            prefix = "* " if teacher == self.current_teacher else "  "
            print(f"{prefix}{i}. {teacher}")

    def select_teacher(self) -> Optional[str]:
        """选择教师

        Returns:
            选中的教师名称，如果取消选择则返回 None
        """
        if not self.teachers:
            print("\n错误：当前没有可选择的教师")
            return None

        while True:
            print("\n请选择教师：")
            self.list_teachers()
            print("\n输入序号选择教师（输入 q 退出）：")

            choice = input().strip()
            if choice.lower() == 'q':
                return None

            try:
                index = int(choice) - 1
                if 0 <= index < len(self.teachers):
                    self.current_teacher = self.teachers[index]
                    print(f"\n已选择教师：{self.current_teacher}")
                    return self.current_teacher
                else:
                    print("\n无效的序号，请重新选择")
            except ValueError:
                print("\n无效的输入，请输入数字或 q")

    def get_current_teacher(self) -> Optional[str]:
        """获取当前选中的教师"""
        return self.current_teacher
