import torch
import torch.nn as nn
import pandas as pd


class tRNATransformerFeatureExtractor(nn.Module):
    def __init__(self, embedding_size=128, nhead=8, num_encoder_layers=6, dim_feedforward=2048,
                 post_feat_dim=256, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 定义标准和非标准的碱基
        non_standard_bases = 'P6N;]KBLQ!$R<*+D?7TZ#=5I}ÿ[4OJ,\'W3.X8Ê>M°∃9Y&H2)\E/V%{⊄S(_t1`:^~F’¿«υc≠ÐguaeŁ¹'
        all_bases = 'AUGC' + non_standard_bases

        self.num_bases = len(all_bases)
        self.base_to_idx = {base: idx for idx, base in enumerate(all_bases)}
        self.embedding = nn.Embedding(num_embeddings=self.num_bases, embedding_dim=embedding_size).to(self.device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   batch_first=True).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers).to(self.device)

        # 设置注意力机制层
        self.attention = nn.Sequential(
            nn.Linear(embedding_size, 1),
            nn.Tanh(),
            nn.Softmax(dim=1)
        ).to(self.device)

        # 设置后处理全连接层
        self.post_process = nn.Linear(embedding_size, post_feat_dim).to(self.device)

    def forward(self, sequence):
        # 将序列中的碱基转换为对应的索引
        sequence_idx = [self.base_to_idx[base] for base in sequence if base in self.base_to_idx]
        sequence_tensor = torch.tensor(sequence_idx, device=self.device).unsqueeze(0)
        embedded_sequence = self.embedding(sequence_tensor)
        features = self.transformer_encoder(embedded_sequence)

        # 应用注意力机制
        attention_weights = self.attention(features)
        features_weighted = torch.mul(features, attention_weights.expand_as(features))
        features_pooled = torch.sum(features_weighted, dim=1)

        # 后处理增加特征维度
        features_post = self.post_process(features_pooled)

        return features_post


def read_trna_sequence_from_csv(pdb_name, file_path='../数据/实验数据/trna_sequences.csv'):
    df = pd.read_csv(file_path)
    sequence_row = df[df['Name'] == pdb_name]
    if not sequence_row.empty:
        modified_sequence = sequence_row.iloc[0]['Sequence']
        return modified_sequence
    else:
        print("未找到对应的tRNA名称。")
        return None


if __name__ == "__main__":
    line_model = tRNATransformerFeatureExtractor(device='cuda')

    pdb_name = 'tdbR00000398'
    sequence = read_trna_sequence_from_csv(pdb_name)

    if sequence:
        feature_vector = line_model(sequence)
        print("特征向量维度:", feature_vector.shape)
        print(feature_vector)
