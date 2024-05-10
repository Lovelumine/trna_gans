import pandas as pd
import os

# CSV文件路径
csv_file_path = 'trna_sequences.csv'
# PDB文件所在文件夹路径，确保这是一个相对路径，指向同级目录下的pdb文件夹
pdb_folder_path = './pdb'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 检查PDB文件是否存在的函数
def check_pdb_exists(row, pdb_folder_path):
    # 直接使用CSV文件中的Name列值作为PDB文件的名称
    pdb_file_name = f"{row['Name']}.pdb"
    # 检查文件是否存在于指定文件夹中
    pdb_file_path = os.path.join(pdb_folder_path, pdb_file_name)
    exists = "yes" if os.path.exists(pdb_file_path) else "no"
    # 打印出正在检查的PDB文件路径和存在状态
    print(f"Checking: {pdb_file_path}, Exists: {exists}")
    return exists

# 应用函数，增加新列
df['Has PDB'] = df.apply(check_pdb_exists, pdb_folder_path=pdb_folder_path, axis=1)

# 保存修改后的DataFrame回CSV文件
df.to_csv('data_with_pdb_status.csv', index=False)

print("处理完成，结果已保存到 'data_with_pdb_status.csv'。")
