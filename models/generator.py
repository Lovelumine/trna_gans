# models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_feature_size):
        """
        生成器模型的初始化。

        参数:
        - input_size (int): 噪声向量的维度。
        - hidden_size (int): 内部隐藏层的维度。
        - output_size (int): 输出维度，包括4维原子类型和3维坐标，共7维。
        - sequence_feature_size (int): tRNA序列特征向量的维度。
        """
        super(Generator, self).__init__()

        self.hidden_size = hidden_size  # 将hidden_size保存为类的属性

        # 特征融合层，将噪声向量和序列特征向量融合
        self.feature_fusion = nn.Linear(input_size + sequence_feature_size, hidden_size)

        # 特征转换层，增加模型的非线性表达能力
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.2)
        )

        # 结构生成层，使用GRU生成序列
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        # 原子信息生成层，用于生成原子类型和坐标
        self.atom_generator = nn.Linear(hidden_size, output_size)

        # 坐标缩放参数，用于调整坐标的范围
        self.coord_scale = nn.Parameter(torch.tensor([300.0]))

    def forward(self, noise, sequence_features_or_batch, sequence_length=1800):
        """
        前向传播函数。

        参数:
        - noise (torch.Tensor): 输入的噪声向量，维度为(batch_size, input_size)。
        - sequence_features_or_batch (torch.Tensor or dict): tRNA序列的特征向量，维度为(batch_size, sequence_feature_size)，或包含序列特征的批处理字典。
        - sequence_length (int): 生成序列的长度，定义了预期生成的原子数量。
        """
        # 检查sequence_features_or_batch的类型，相应地获取sequence_features
        if isinstance(sequence_features_or_batch, torch.Tensor):
            # 如果传入的是Tensor，直接使用
            sequence_features = sequence_features_or_batch
        else:
            # 否则，尝试从批处理对象中提取sequence_features
            sequence_features = sequence_features_or_batch['sequence_character']
        # 融合噪声向量和序列特征向量
        fused_features = torch.cat([noise, sequence_features], dim=1)
        fused_features = self.feature_fusion(fused_features)

        # 通过特征转换层处理融合后的特征
        x = self.feature_transform(fused_features)

        # 准备GRU的输入，重复融合后的特征以匹配序列长度
        gru_input = x.unsqueeze(1).repeat(1, sequence_length, 1)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # 通过GRU生成序列
        output, _ = self.gru(gru_input, h0)
        atom_info = self.atom_generator(output)

        # 分离出原子类型和坐标信息
        atom_type_logits = atom_info[:, :, :4]
        atom_type_probs = F.softmax(atom_type_logits, dim=-1)  # 获取原子类型的概率分布

        # 调整坐标信息的范围
        atom_coords_raw = atom_info[:, :, 4:]
        atom_coords = torch.tanh(atom_coords_raw) * self.coord_scale
        atom_coords = torch.round(atom_coords * 1000) / 1000  # 保留小数点后三位精度

        return atom_type_probs, atom_coords
