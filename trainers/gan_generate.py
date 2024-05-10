import torch
from models.generator import Generator
from utils.line_character import tRNATransformerFeatureExtractor


def save_as_pdb(atom_type_probs, atom_coords, filename):
    # 元素类型的映射
    element_mapping = {0: 'C', 1: 'O', 2: 'N', 3: 'P'}
    # 选择每个位置最可能的原子类型
    atom_types = torch.argmax(atom_type_probs, dim=-1)

    with open(filename, 'w') as pdb_file:
        for i, (atom_type, coord) in enumerate(zip(atom_types[0], atom_coords[0])):
            # 获取元素类型
            element = element_mapping[atom_type.item()]
            # PDB格式化字符串
            pdb_line = f"ATOM  {i + 1:5d}  {element:<2}  MOL     1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}\n"
            pdb_file.write(pdb_line)
        pdb_file.write("END\n")

def load_model(model_path, config, device):
    """加载训练好的模型"""
    model = Generator(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size'],
        sequence_feature_size=config['sequence_feature_size']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 将模型设置为评估模式
    return model


def prepare_input(sequence, device):
    """准备输入数据：将序列转换为特征向量"""
    # 假设我们已经有了一个可以将序列转换为特征向量的函数
    feature_extractor = tRNATransformerFeatureExtractor(device=device)
    sequence_features = feature_extractor(sequence)
    return sequence_features.to(device)


def generate_structure(generator, noise, sequence_features):
    """使用生成器生成三级结构"""
    with torch.no_grad():  # 在推理模式下，不计算梯度
        atom_type_probs, atom_coords = generator(noise, sequence_features)
    return atom_type_probs, atom_coords


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./saved_models/generator.pth"
    generator_config = {
        'input_size': 100,
        'hidden_size': 256,
        'output_size': 8,  # 包括4维原子类型和3维坐标
        'sequence_feature_size': 256
    }

    # 加载模型
    generator = load_model(model_path, generator_config, device)

    # 用户输入的tRNA一级序列
    user_input_sequence = "AAAUAUGA\"GCGAUUUAUUGCAAPUAGPUUCGACCUAAUCUUAGGUGAAAUUCACCCAPAUUUUCCA"

    # 准备输入数据
    sequence_features = prepare_input(user_input_sequence, device)

    # 生成噪声向量
    noise = torch.randn(1, generator_config['input_size'], device=device)

    # 生成三级结构数据
    atom_type_probs, atom_coords = generate_structure(generator, noise, sequence_features)

    # 输出或处理生成的三级结构数据
    print("原子类型概率:", atom_type_probs)
    print("原子坐标:", atom_coords)

    save_as_pdb(atom_type_probs, atom_coords, "generated_structure.pdb")


if __name__ == "__main__":
    main()


# # 示例使用
# sequence = "AAAUAUGA\"GCGAUUUAUUGCAAPUAGPUUCGACCUAAUCUUAGGUGAAAUUCACCCAPAUUUUCCA"
# atom_type_probs, atom_coords = generate_trna_3d_structure(sequence)
#
# print("生成的原子类型概率:", atom_type_probs)
# print("生成的原子坐标:", atom_coords)
