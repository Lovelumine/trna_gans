# trainers/model_initialization
from models.generator import Generator
from models.discriminator import Discriminator
from trainers.loss_functions import discriminator_loss, generator_loss
import torch
from torch import optim

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
def initialize_models(generator_config, discriminator_config):
    """
    根据配置初始化生成器和判别器模型。

    参数:
    - generator_config: 包含生成器配置参数的字典。
    - discriminator_config: 包含判别器配置参数的字典。

    返回:
    - generator: 初始化后的生成器模型。
    - discriminator: 初始化后的判别器模型。
    """
    generator = Generator(
        input_size=generator_config['input_size'],
        hidden_size=generator_config['hidden_size'],
        output_size=generator_config['output_size'],
        sequence_feature_size=generator_config['sequence_feature_size']
    ).to(generator_config['device'])

    discriminator = Discriminator(
        input_feature_dim=discriminator_config['input_feature_dim'],
        hidden_dim=discriminator_config['hidden_dim'],
        edge_attr_dim=discriminator_config['edge_attr_dim'],
        pos_dim=discriminator_config['pos_dim'],
        target_dim=discriminator_config['target_dim'],
        sequence_feature_dim=discriminator_config['sequence_feature_size']
    ).to(discriminator_config['device'])

    return generator, discriminator


def train_discriminator(discriminator, real_features, real_labels, fake_features, fake_labels, d_optimizer):
    """
    训练判别器模型。
    """
    d_optimizer.zero_grad()
    # 真实数据的损失
    real_loss = discriminator_loss(discriminator(real_features), real_labels)
    # 生成数据的损失
    fake_loss = discriminator_loss(discriminator(fake_features.detach()), fake_labels)
    # 总损失
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()

    return d_loss.item()


def train_generator(discriminator, fake_features, fake_labels, g_optimizer):
    """
    训练生成器模型。
    """
    g_optimizer.zero_grad()
    # 生成数据的损失
    g_loss = generator_loss(discriminator(fake_features), fake_labels)
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item()


def setup_optimizers(generator, discriminator, learning_rate=0.0002):
    """
    设置优化器。
    """
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    return g_optimizer, d_optimizer


# 测试代码
if __name__ == "__main__":
    # 确定设备，如果CUDA可用则使用第一个GPU，否则使用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 更新的生成器和判别器配置，包括设备信息
    generator_config = {
        'input_size': 100,  # 假定的噪声向量维度
        'hidden_size': 256,
        'output_size': 7,  # 4个原子类型 + 3维坐标
        'sequence_feature_size': 24,  # 更新为实际的序列特征维度
        'device': device  # 添加设备信息
    }

    discriminator_config = {
        'input_feature_dim': 4,  # 假设每个节点的特征维度是4
        'hidden_dim': 256,
        'sequence_feature_size': 24,  # 应与generator_config中的sequence_feature_size一致
        'device': device  # 添加设备信息
    }

    generator, discriminator = initialize_models(generator_config, discriminator_config)

    print(f"Generator model:\n{generator}")
    print(f"Discriminator model:\n{discriminator}")

