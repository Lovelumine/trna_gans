# utils/threeD_data.py
import pandas as pd
import numpy as np
import torch
from utils.threeD_character import DataNeighbors


def prepare(item, k=50):
    """
    准备数据的函数，对原子的3D坐标和元素类型进行处理，并构建图数据结构。
    """
    element_mapping = {'C': 0, 'O': 1, 'N': 2, 'P': 3}
    if isinstance(item['atoms'], pd.DataFrame):
        coords = item['atoms'][['x', 'y', 'z']].values
        elements = item['atoms']['element'].values

        # 使用布尔索引筛选出已知元素
        mask = np.isin(elements, list(element_mapping.keys()))
        filtered_coords = coords[mask]
        filtered_elements = elements[mask]

        # 直接构建特征矩阵而无需循环
        features = np.zeros((filtered_coords.shape[0], len(element_mapping)))
        for element, idx in element_mapping.items():
            features[filtered_elements == element, idx] = 1

        geometry = torch.tensor(filtered_coords, dtype=torch.float32)
        features = torch.tensor(features, dtype=torch.float32)
    else:
        # 如果没有原子信息，返回空的DataNeighbors对象
        return DataNeighbors(x=torch.empty((0, len(element_mapping))), pos=torch.empty((0, 3)), r_max=10.0)

    # 使用DataNeighbors类创建图数据结构
    data = DataNeighbors(x=features, pos=geometry, r_max=10.0)
    return data


def prepare_data(structures):
    """
    准备所有结构数据的函数，将原子的3D坐标和元素类型处理后构建图数据结构的列表。

    参数:
    - structures: 结构数据列表，每个元素是一个包含原子信息的字典。

    返回:
    - 数据列表，每个元素是处理后的DataNeighbors对象。
    """
    prepared_data = []
    for item in structures:
        data = prepare(item)
        prepared_data.append(data)

    return prepared_data