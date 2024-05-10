#trainers/loss_functions.py
import torch
import torch.nn.functional as F

def generator_loss(disc_generated_output):
    """
    计算生成器的损失。
    参数:
    - disc_generated_output: 判别器对生成数据的输出。
    返回:
    - 损失值。
    """
    return F.binary_cross_entropy(disc_generated_output, torch.ones_like(disc_generated_output))

def discriminator_loss(disc_real_output, disc_generated_output):
    """
    计算判别器的损失。
    参数:
    - disc_real_output: 判别器对真实数据的输出。
    - disc_generated_output: 判别器对生成数据的输出。
    返回:
    - 损失值。
    """
    real_loss = F.binary_cross_entropy(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = F.binary_cross_entropy(disc_generated_output, torch.zeros_like(disc_generated_output))
    total_loss = real_loss + generated_loss
    return total_loss
