# utils/threeD_test.py
import pandas as pd
from threeD_data import prepare  # 导入修改后的prepare函数

def read_pdb(file_path):
    """
    从PDB文件中读取原子坐标和元素类型，返回一个DataFrame。
    """
    columns = ['element', 'x', 'y', 'z']
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                element = line[76:78].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                data.append([element, x, y, z])
    return pd.DataFrame(data, columns=columns)

def test_prepare_from_pdb(file_path):
    """
    从PDB文件中读取原子数据并应用prepare函数处理，无需关心标签。
    """
    atoms_df = read_pdb(file_path)

    # 直接调用prepare函数，无需使用create_transform
    prepared_data = prepare({'atoms': atoms_df, 'id': 'test_id', 'file_path': file_path})

    print("处理完成。")
    print(f"原子数量: {len(atoms_df)}")
    print(f"处理后的数据: {prepared_data}")

if __name__ == "__main__":
    test_prepare_from_pdb("../数据/实验数据/pdb/tdbPDB000018.pdb")  # 实际路径可能需要根据你的文件位置进行调整
