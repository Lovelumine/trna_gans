import random
import os


def modify_pdb_file(input_file, output_file):
    # 定义可能的替代元素
    replacement_elements = ['C', 'N', 'O', 'P']

    # 读取原始PDB文件
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 修改数据
    modified_lines = []
    for line in lines:
        if line.startswith("ATOM") and len(line) > 54:  # 确保行足够长，包含需要的数据
            # 读取并保留原有的数据
            atom_number = line[6:11]
            original_element = line[12:16]
            residue_name = line[17:20]
            chain_id = line[21]
            residue_number = line[22:26]
            x_coord = line[30:38]
            y_coord = line[38:46]
            z_coord = line[46:54]

            # 随机选择一个新元素
            new_element = random.choice(replacement_elements)
             # 构造新的一行，保持固定宽度格式
            modified_line = f"ATOM  {atom_number} {new_element:<3}{residue_name} {chain_id}{residue_number}    {x_coord}{y_coord}{z_coord}\n"
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    # 写入修改后的文件
    with open(output_file, 'w') as file:
        file.writelines(modified_lines)


# 文件路径（假设脚本和文件在同一目录下）
input_file = 'generated_structure1.pdb'
output_file = 'modified_generated_structure1.pdb'

# 检查文件是否存在
if os.path.exists(input_file):
    modify_pdb_file(input_file, output_file)
    print("文件已成功修改并保存为:", output_file)
else:
    print("文件不存在，请检查路径和文件名")
