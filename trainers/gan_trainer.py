from trainers.training_loop import train_gan
import torch



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义生成器和判别器的配置
    generator_config = {
        'input_size': 100,
        'hidden_size': 256,
        'output_size': 8,
        'sequence_feature_size': 256,
        'device':device
    }

    discriminator_config = {
        'input_feature_dim': 4,
        'hidden_dim': 16,
        'pos_dim':3,
        'edge_attr_dim':3,
        'sequence_feature_size': 256,
        'target_dim':64,
        'device': device
    }


    # 定义训练参数
    learning_rate = 2e-4
    batch_size = 8
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader_cache_path = '../data/data_loader_cache'
    model_save_dir = './saved_models'  # 模型保存目录

    # 开始训练
    train_gan(
        generator_config,
        discriminator_config,
        data_loader_cache_path,
        learning_rate,
        batch_size,
        num_epochs,
        device,
        model_save_dir  # 传递模型保存目录
    )

if __name__ == "__main__":
    main()
