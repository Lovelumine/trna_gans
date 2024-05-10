# utils/threeD_character.py
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
import torch_geometric.data as tg_data
import numpy as np  # 引入numpy


def neighbor_list_and_relative_vec(pos, r_max, self_interaction=True):
    """
    根据给定的原子位置和最大半径（r_max），计算原子之间的邻居列表和相对位置向量。
    如果启用自相互作用，原子将被视为自己的邻居。

    参数:
    - pos (torch.Tensor): 原子的3D坐标，形状为 (N, 3)。
    - r_max (float): 原子间邻居的最大距离。
    - self_interaction (bool): 是否将原子视为自己的邻居。

    返回:
    - edge_index (torch.Tensor): 邻居关系的索引，形状为 (2, num_edges)。
    - edge_attr (torch.Tensor): 相对位置向量，形状为 (num_edges, 3)。
    """
    # 检查 pos 是否为 None 或空 tensor
    if pos is None or pos.nelement() == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.float32)

    N, _ = pos.shape
    atoms = Atoms(symbols=['H'] * N, positions=pos.numpy())

    i_list, j_list, S = neighbor_list('ijS', atoms, r_max, self_interaction=self_interaction)
    edge_index = torch.tensor(np.vstack([i_list, j_list]), dtype=torch.long)
    offsets = torch.tensor(S @ atoms.get_cell(), dtype=torch.float32)
    edge_attr = pos[j_list] + offsets - pos[i_list]

    return edge_index, edge_attr


class DataNeighbors(tg_data.Data):
    def __init__(self, x, pos, r_max, self_interaction=True, **kwargs):
        # 在调用 neighbor_list_and_relative_vec 之前检查 pos 是否为 None 或空 tensor
        if pos is None or pos.nelement() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float32)
        else:
            edge_index, edge_attr = neighbor_list_and_relative_vec(pos, r_max, self_interaction)

        super(DataNeighbors, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                            pos=pos if pos is not None else torch.empty((0, 3)), **kwargs)


# 测试代码保持不变
if __name__ == "__main__":
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    x = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    r_max = 5
    self_interaction = False

    data_neighbors = DataNeighbors(x=x, pos=pos, r_max=r_max, self_interaction=self_interaction)
    print("边的索引:", data_neighbors.edge_index)
    print("边的属性:", data_neighbors.edge_attr)
    print("原子的位置:", data_neighbors.pos)
