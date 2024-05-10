# data/data_loader.py
import pandas as pd
import os
import concurrent.futures
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from utils.line_character import tRNATransformerFeatureExtractor
from utils.threeD_data import prepare
from sklearn.model_selection import train_test_split


def read_pdb_to_df(pdb_path):
    """从PDB文件读取原子坐标和类型，并返回pandas.DataFrame。"""
    print(f"正在从PDB文件读取数据: {pdb_path}")
    data = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                element = line[76:78].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                data.append({'element': element, 'x': x, 'y': y, 'z': z})
    print(f"读取结构完成: {pdb_path}")
    return pd.DataFrame(data)


def process_pdb_file(pdb_file, pdb_folder_path, trna_df, base_to_idx):
    """处理单个PDB文件的函数，适用于线程池调用。"""
    pdb_id = pdb_file[:-4]
    print(f"处理PDB文件: {pdb_id}")
    row = trna_df.loc[trna_df['Name'].str.contains(pdb_id, na=False)]
    if not row.empty:
        sequence = row.iloc[0]['Sequence']
        print(f"找到序列: {sequence[:10]}... 对应于 {pdb_id}")
        line_feature_extractor = tRNATransformerFeatureExtractor(device='cuda')
        line_feature_vector = line_feature_extractor(sequence)
        pdb_path = os.path.join(pdb_folder_path, pdb_file)
        atoms_df = read_pdb_to_df(pdb_path)
        structure_data = prepare({'atoms': atoms_df, 'id': pdb_id, 'file_path': pdb_path})

        # 将tRNA名称、一级序列特征和三维结构特征保存到Data对象中
        data_object = Data(pdb_id=pdb_id,x=structure_data.x, pos=structure_data.pos,edge_index=structure_data.edge_index, edge_attr=structure_data.edge_attr,
                            sequence=sequence,sequence_character=line_feature_vector)
        return data_object
    else:
        print(f"CSV中未找到PDB ID {pdb_id} 对应的序列。")
        return None


def load_and_process_data(trna_csv_path, pdb_folder_path, base_to_idx, batch_size, max_workers=4):
    print(f"开始加载并处理数据")
    trna_df = pd.read_csv(trna_csv_path)
    print(f"已加载tRNA序列数据: {len(trna_df)} 条记录")
    structure_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pdb_file, pdb_file, pdb_folder_path, trna_df, base_to_idx) for pdb_file in
                   os.listdir(pdb_folder_path) if pdb_file.endswith('.pdb')]
        for future in concurrent.futures.as_completed(futures):
            structure_data = future.result()
            if structure_data is not None:
                structure_list.append(structure_data)
    print(f"所有文件处理完毕，总共处理的结构数量：{len(structure_list)}")

    # 分割数据集并创建DataLoader
    structures_train_val, structures_test = train_test_split(structure_list, test_size=0.2, random_state=42)
    structures_train, structures_val = train_test_split(structures_train_val, test_size=0.25, random_state=42)

    train_loader = DataLoader(structures_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(structures_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(structures_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_base_to_idx(all_bases):
    """创建碱基到索引的映射。"""
    print("创建碱基到索引的映射")
    return {base: idx for idx, base in enumerate(all_bases)}
