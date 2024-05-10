# data/data_save.py
import pickle
import os
from tqdm import tqdm  # 引入tqdm库
from data.data_loader import load_and_process_data, create_base_to_idx


def save_data_loaders(train_loader, val_loader, test_loader, save_path='data_loader_cache'):
    """保存DataLoader对象到本地"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'train_loader.pkl'), 'wb') as f:
        pickle.dump(train_loader, f)
    with open(os.path.join(save_path, 'val_loader.pkl'), 'wb') as f:
        pickle.dump(val_loader, f)
    with open(os.path.join(save_path, 'test_loader.pkl'), 'wb') as f:
        pickle.dump(test_loader, f)
    print("DataLoader对象已保存至本地。")


def load_data_loaders(save_path='data_loader_cache'):
    """从本地加载DataLoader对象"""
    try:
        with open(os.path.join(save_path, 'train_loader.pkl'), 'rb') as f:
            train_loader = pickle.load(f)
        with open(os.path.join(save_path, 'val_loader.pkl'), 'rb') as f:
            val_loader = pickle.load(f)
        with open(os.path.join(save_path, 'test_loader.pkl'), 'rb') as f:
            test_loader = pickle.load(f)
        print("DataLoader对象已从本地加载。")
        return train_loader, val_loader, test_loader
    except FileNotFoundError:
        print("本地未找到DataLoader对象。")
        return None, None, None


def print_batch_info(data_loader, data_loader_name):
    """打印DataLoader中第一批数据的相关信息"""
    for batch in tqdm(data_loader, desc=f"遍历{data_loader_name}", unit="batch"):
        print(f"{data_loader_name}批次大小: {batch.num_graphs}")
        print(f"{data_loader_name}第一个图的节点特征维度: {batch.x[0].size()}")
        print(f"{data_loader_name}第一个图的边索引维度: {batch.edge_index.size()}")
        if hasattr(batch, 'pdb_id'):
            print(f"{data_loader_name}第一个图的PDB ID: {batch.pdb_id[0]}")
        if hasattr(batch, 'sequence'):
            print(f"{data_loader_name}第一个图的序列: {batch.sequence[0]}")
        # 这里添加了对 edge_attr 和 pos 的打印
        print(f"{data_loader_name}第一个图的边属性维度: {batch.edge_attr.size()}")
        print(f"{data_loader_name}第一个图的位置信息维度: {batch.pos.size()}")
        if hasattr(batch, 'sequence_character'):
            print(f"{data_loader_name}第一个图的序列特征维度: {batch.sequence_character.size()}")
        else:
            print(f"{data_loader_name}没有找到'sequence_character'属性")

        # 检查序列特征和图特征的一致性
        if hasattr(batch, 'sequence_character'):
            # 假设序列特征是每个图的一个特征，检查其与图数量的一致性
            sequence_feature_size = batch.sequence_character.size(0)
            if sequence_feature_size != batch.num_graphs:
                print(f"警告: {data_loader_name}序列特征的数量({sequence_feature_size})与图的数量({batch.num_graphs})不一致")
            else:
                print(f"{data_loader_name}序列特征的数量与图的数量一致")
        break  # 仅展示第一批次数据



if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data_loaders()
    if not train_loader:
        trna_csv_path = 'trna_sequences.csv'
        pdb_folder_path = 'pdb'
        non_standard_bases = 'P6N;]KBLQ!$R<*+D?7TZ#=5I}ÿ[4OJ,\'W3.X8Ê>M°∃9Y&H2)\E/V%{⊄S(_t1`:^~F’¿«υc≠ÐguaeŁ¹'
        all_bases = 'AUGC' + non_standard_bases
        base_to_idx = create_base_to_idx(all_bases)

        manual_max_workers = 4

        print("正在加载并处理数据，请稍候...")
        train_loader, val_loader, test_loader = load_and_process_data(
            trna_csv_path,
            pdb_folder_path,
            base_to_idx,
            batch_size=8,
            max_workers=manual_max_workers
        )

        save_data_loaders(train_loader, val_loader, test_loader)

    print_batch_info(train_loader, "训练集")
    print_batch_info(val_loader, "验证集")
    print_batch_info(test_loader, "测试集")

    print("数据加载和批次处理测试完成。")
