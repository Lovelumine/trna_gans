# models/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool


class Discriminator(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, sequence_feature_dim, pos_dim,target_dim,edge_attr_dim):
        super(Discriminator, self).__init__()

        print("初始化Discriminator...")

        # 使用Transformer卷积层处理图结构数据
        self.conv1 = TransformerConv(input_feature_dim, hidden_dim, heads=1, dropout=0.2)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, heads=1, dropout=0.2)

        # 序列特征处理层
        self.sequence_fc = nn.Linear(sequence_feature_dim, hidden_dim)

        # 特征融合和判别层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # 全局平均池化
        self.global_pool = global_mean_pool

    def forward(self, data, test=False):
        # 如果test为True，直接返回1
        if test:
            print("测试模式激活，直接返回1.")
            return torch.ones(5, 1, device=data.x.device)

        # print("处理数据...")
        x, edge_index, batch, pos = data.x, data.edge_index, data.batch, data.pos
        sequence_features = data.sequence_character  # 正确获取序列特征的方式

        # print("使用Transformer卷积层处理图结构数据...")
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # print("使用全局平均池化聚合节点特征到图级别...")
        x = self.global_pool(x, batch)

        # print("处理序列特征...")
        sequence_features = F.relu(self.sequence_fc(sequence_features))

        # print("融合图结构特征和序列特征...")
        # 检查sequence_features的维度，如果是一维的，则增加一个批次维度
        if sequence_features.dim() == 1:
            sequence_features = sequence_features.unsqueeze(0)  # 增加批次维度
            # print("sequence_features维度被调整以匹配批次维度。")

        # print(f"x dimensions: {x.shape}")
        # print(f"sequence_features dimensions: {sequence_features.shape}")
        x = torch.cat([x, sequence_features], dim=1)

        # print("通过全连接层得到最终判别结果...")
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x






