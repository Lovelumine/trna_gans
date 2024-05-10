import subprocess
import os

def export_conda_environment(env_name, save_path="environment.yml"):
    """
    导出Conda环境配置到一个文件

    参数:
    - env_name: str, 虚拟环境的名称
    - save_path: str, 保存环境配置的文件路径
    """
    # 使用conda env export命令导出环境配置
    command = f"conda env export --name {env_name} > {save_path}"
    process = subprocess.run(command, shell=True, check=True)

    if process.returncode == 0:
        print(f"环境 {env_name} 已成功导出到 {save_path}")
    else:
        print(f"导出环境 {env_name} 失败")

if __name__ == "__main__":
    # 替换 YOUR_ENV_NAME 为你的Conda环境名
    env_name = "tRNA"
    save_path = "environment.yml"
    export_conda_environment(env_name, save_path)
