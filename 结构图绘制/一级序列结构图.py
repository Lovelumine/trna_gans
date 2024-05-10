import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 指定支持中文的字体，这里使用SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 如果你有这个字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题
def draw_model_structure_with_io_and_fix(num_encoder_layers, nhead, embedding_size, dim_feedforward):
    """
    Draws a schematic representation of the tRNATransformerFeatureExtractor models structure, including input and output.
    This version fixes the Chinese character support issue and correctly represents the input as a 1D tRNA sequence.
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    # Setting up the canvas
    ax.set_xlim(0, 12)
    ax.set_ylim(0, num_encoder_layers + 5)
    ax.axis('off')

    # Input Layer
    ax.text(6, num_encoder_layers + 4.5, '输入: tRNA一维序列', ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='red', alpha=0.5))

    # Embedding Layer
    embedding_rect = patches.Rectangle((5, num_encoder_layers + 3.5), 2, 0.8, linewidth=1, edgecolor='r',
                                       facecolor='none')
    ax.add_patch(embedding_rect)
    ax.text(6, num_encoder_layers + 4, '嵌入层', ha='center', va='center', fontsize=12)

    # Transformer Encoder Layers
    for i in range(num_encoder_layers):
        layer_y = num_encoder_layers + 2.5 - i
        encoder_rect = patches.Rectangle((5, layer_y), 2, 0.8, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(encoder_rect)
        ax.text(6, layer_y + 0.4, f'编码器层 {i + 1}', ha='center', va='center', fontsize=12)

    # Output Layer
    ax.text(6, 0.5, '输出: 特征向量', ha='center', va='center', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))

    # Annotations for parameters
    ax.text(1, num_encoder_layers + 4, f'嵌入维度: {embedding_size}\n头数: {nhead}\n前馈网络维度: {dim_feedforward}',
            ha='left', va='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()


# Drawing the models structure with input and output, fixing Chinese support
draw_model_structure_with_io_and_fix(num_encoder_layers=3, nhead=4, embedding_size=12, dim_feedforward=512)
