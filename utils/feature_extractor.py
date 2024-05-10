# utils/feature_extractor.py
import pandas as pd
import torch
from utils.threeD_data import prepare_data

import pandas as pd
import torch
from torch_geometric.data import Data  # 确保导入Data类
from utils.threeD_data import prepare_data


def extract_features(sequence_character, atom_type_probs, atom_coords, element_mapping):
    """
    将生成器输出的原子类型概率和坐标转换为图数据对象，并添加序列特征。

    参数:
    - sequence_character (torch.Tensor): tRNA序列的特征向量，形状为(batch_size, sequence_feature_size)。
    - atom_type_probs (torch.Tensor): 生成器输出的原子类型概率，形状为(batch_size, sequence_length, num_atom_types)。
    - atom_coords (torch.Tensor): 生成器输出的原子坐标，形状为(batch_size, sequence_length, 3)。
    - element_mapping (dict): 元素到索引的映射字典。

    返回:
    - data_list: 包含Data对象的列表，每个对象对应于一组原子的图表示及其序列特征。
    """
    batch_size = atom_type_probs.size(0)
    # print('模拟数据维度大小',batch_size)
    data_list = []

    for i in range(batch_size):
        types_indices = torch.argmax(atom_type_probs[i], dim=-1)
        elements = [list(element_mapping.keys())[index.item()] for index in types_indices]
        coords = atom_coords[i].detach().cpu().numpy()

        atoms_df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'z': coords[:, 2],
            'element': elements
        })

        item = {'atoms': atoms_df}
        # 调用 prepare_data 生成图结构数据
        data = prepare_data([item])[0]  #  prepare_data 返回一个包含Data对象的列表

        # 将sequence_character添加到每个图结构数据对象中
        data.sequence_character = sequence_character[i]

        # print(data.sequence_character)

        data_list.append(data)

    return data_list


# 测试代码
if __name__ == "__main__":
    batch_size = 2
    sequence_length = 5
    num_atom_types = 4
    sequence_feature_size = 256  # 假设序列特征向量的维度是256

    # 创建模拟的atom_type_probs和atom_coords
    atom_type_probs = torch.softmax(torch.randn(batch_size, sequence_length, num_atom_types), dim=-1)
    atom_coords = torch.randn(batch_size, sequence_length, 3)

    # 创建模拟的sequence_character张量
    sequence_character = torch.randn(batch_size, sequence_feature_size)

    # 元素到索引的映射字典
    element_mapping = {'C': 0, 'O': 1, 'N': 2, 'S': 3}

    # 调用extract_features函数
    data_list = extract_features(sequence_character, atom_type_probs, atom_coords, element_mapping)

    # 打印生成的Data对象的相关信息
    for data in data_list:
        print(f"Type of data: {type(data)}")
        print(f"Node features (x): {data.x}")
        print(f"Node positions (pos): {data.pos}")
        print(f"Edge index: {data.edge_index}")
        print(f"Edge attributes (edge_attr): {data.edge_attr}")
        print(f"Sequence character features: {data.sequence_character}\n")

