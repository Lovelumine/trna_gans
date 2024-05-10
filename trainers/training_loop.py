import torch
from torch import optim
from trainers.data_preparation import load_data_loaders
from trainers.model_initialization import initialize_models, load_model, save_model
from trainers.loss_functions import generator_loss, discriminator_loss
from utils.feature_extractor import extract_features
import os
import sys

from utils.line_character import tRNATransformerFeatureExtractor

def train_gan(generator_config, discriminator_config, data_loader_cache_path, learning_rate, batch_size, num_epochs,
              device, model_save_dir):
    print("检查模型保存目录是否存在...")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"创建目录：{model_save_dir}")

    generator_path = os.path.join(model_save_dir, 'generator.pth')
    discriminator_path = os.path.join(model_save_dir, 'discriminator.pth')

    print("加载数据...")
    train_loader, _, _ = load_data_loaders(data_loader_cache_path)

    print("初始化模型...")
    generator, discriminator = initialize_models(generator_config, discriminator_config)
    generator.to(device)
    discriminator.to(device)

    if os.path.exists(generator_path) and os.path.exists(discriminator_path):
        print("检测到预训练模型，是否继续训练? [y/n]:")
        choice = input().strip().lower()
        if choice == 'y':
            load_model(generator, generator_path)
            load_model(discriminator, discriminator_path)
            print("模型加载成功，继续训练...")
        elif choice == 'n':
            print("将从头开始训练...")
        else:
            print("无效输入，退出程序。")
            sys.exit(1)

    print("设置优化器...")
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    element_mapping = {'C': 0, 'O': 1, 'N': 2, 'P': 3}
    line_feature_extractor = tRNATransformerFeatureExtractor(device=device)

    print("开始训练...")
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            print(f"批次 {i+1} 训练开始...")

            # 训练判别器
            print("训练判别器...")
            d_optimizer.zero_grad()
            real_data_out = discriminator(batch)
            d_real_loss = discriminator_loss(real_data_out, torch.ones_like(real_data_out))
            # print("处理真实数据完成.")
            #
            # print("生成模拟数据用于判别器训练...")
            noise = torch.randn(batch_size, generator_config['input_size'], device=device)
            generated_atoms, generated_coords = generator(noise, batch)
            # print("生成模拟数据完成...")
            generated_data_list = extract_features(batch.sequence_character, generated_atoms, generated_coords, element_mapping)
            # print("模拟数据处理完成...")
            d_generated_loss = 0
            for generated_data in generated_data_list:
                generated_data = generated_data.to(device)
                generated_data_out = discriminator(generated_data)
                d_generated_loss += discriminator_loss(generated_data_out, torch.zeros_like(generated_data_out))
            d_generated_loss /= len(generated_data_list)
            d_loss = d_real_loss + d_generated_loss
            d_loss.backward()
            d_optimizer.step()
            print("判别器训练完成.")

            # 训练生成器
            print("训练生成器...")
            g_optimizer.zero_grad()
            noise = torch.randn(batch_size, generator_config['input_size'], device=device)
            generated_atoms, generated_coords = generator(noise,batch)
            generated_data_list = extract_features(batch.sequence_character, generated_atoms, generated_coords, element_mapping)

            g_loss = 0
            for generated_data in generated_data_list:
                generated_data = generated_data.to(device)
                generated_data_out = discriminator(generated_data)
                g_loss += generator_loss(generated_data_out)
            g_loss /= len(generated_data_list)
            g_loss.backward()
            g_optimizer.step()
            print("生成器训练完成.")

            print(f"批次 {i+1} 训练完成：判别器损失 = {d_loss.item()}, 生成器损失 = {g_loss.item()}")

        # 每个epoch结束时保存模型
        save_model(generator, generator_path)
        save_model(discriminator, discriminator_path)
        print(f"Epoch {epoch + 1}/{num_epochs} 完成, 模型已保存至 {model_save_dir}")

    print("训练完成。")
