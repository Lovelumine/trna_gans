import torch
from trainers.data_preparation import load_data_loaders
from models.generator import Generator
from utils.feature_extractor import extract_features
from utils.threeD_data import prepare_data

# 配置参数
generator_config = {
    'input_size': 100,
    'hidden_size': 256,
    'output_size': 7,  # 4维原子类型 + 3维坐标
    'sequence_feature_size': 128
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型（假设已经定义了Generator）
generator = Generator(
    input_size=generator_config['input_size'],
    hidden_size=generator_config['hidden_size'],
    output_size=generator_config['output_size'],
    sequence_feature_size=generator_config['sequence_feature_size']
).to(device)

# 生成测试数据
noise = torch.randn(1, generator_config['input_size'], device=device)  # 假设batch_size为1
sequence_features = torch.randn(1, generator_config['sequence_feature_size'], device=device)

# 生成器输出
generated_atoms, generated_coords = generator(noise, sequence_features)
print("生成器输出原子类型维度:", generated_atoms.shape)
print("生成器输出坐标维度:", generated_coords.shape)

# 元素映射
element_mapping = {'C': 0, 'O': 1, 'N': 2, 'P': 3}

# 特征提取
generated_data_list = extract_features(generated_atoms, generated_coords, element_mapping)
if generated_data_list:
    print("特征提取后数据对象数量:", len(generated_data_list))
    # 打印第一个对象的维度信息
    data = generated_data_list[0]
    print("特征提取后节点特征维度:", data.x.shape)
    print("特征提取后边索引维度:", data.edge_index.shape)

# 加载一个样本数据查看其维度（假设train_loader已经准备好）
_, train_loader, _ = load_data_loaders('../data/data_loader_cache')
for batch in train_loader:
    batch = batch.to(device)
    print("数据集中三维结构数据的节点特征维度:", batch.x.shape)
    print("数据集中三维结构数据的边索引维度:", batch.edge_index.shape)
    break  # 只查看第一个batch
