# trainers/data_preparation.py
import os
import pickle


def load_data_loaders(data_loader_cache_path):
    """
    从指定的路径加载训练集、验证集和测试集的DataLoader对象。

    参数:
    - data_loader_cache_path: 存储DataLoader对象pickle文件的路径。

    返回:
    - train_loader, val_loader, test_loader: 训练、验证和测试的DataLoader对象。
    """
    train_loader_path = os.path.join(data_loader_cache_path, 'train_loader.pkl')
    val_loader_path = os.path.join(data_loader_cache_path, 'val_loader.pkl')
    test_loader_path = os.path.join(data_loader_cache_path, 'test_loader.pkl')

    with open(train_loader_path, 'rb') as f:
        train_loader = pickle.load(f)

    with open(val_loader_path, 'rb') as f:
        val_loader = pickle.load(f)

    with open(test_loader_path, 'rb') as f:
        test_loader = pickle.load(f)

    return train_loader, val_loader, test_loader


# 测试代码
if __name__ == "__main__":
    # data_loader_cache文件夹在../data/data_loader_cache目录下
    data_loader_cache_path = '../data/data_loader_cache'

    train_loader, val_loader, test_loader = load_data_loaders(data_loader_cache_path)

    # 打印加载的DataLoader对象的一些信息
    print(f"Loaded {len(train_loader.dataset)} items in train_loader.")
    print(f"Loaded {len(val_loader.dataset)} items in val_loader.")
    print(f"Loaded {len(test_loader.dataset)} items in test_loader.")
