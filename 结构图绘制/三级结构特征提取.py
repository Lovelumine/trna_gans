import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 指定支持中文的字体，这里使用SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 如果你有这个字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题
def draw_deep_learning_structure():
    """
    Draws a schematic representation of the deep learning structure for neighbor list generation.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Setting up the canvas
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Input Layer
    ax.text(7, 8.5, '输入: 原子特征和3D坐标', ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='red', alpha=0.5))

    # Embedding Layer (not shown for simplicity)
    ax.text(7, 7.5, '嵌入层', ha='center', va='center', fontsize=12, bbox=dict(facecolor='orange', alpha=0.5))

    # Neighbor List Generation
    ax.text(7, 6.5, '邻居列表生成', ha='center', va='center', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
    ax.arrow(7, 7, 0, -0.7, head_width=0.2, head_length=0.1, fc='blue', ec='blue')

    # Neighbor List (Edge Index)
    ax.text(4, 5, '边的索引', ha='center', va='center', fontsize=12, bbox=dict(facecolor='green', alpha=0.5))
    ax.arrow(7, 6, -2.5, -0.5, head_width=0.2, head_length=0.1, fc='green', ec='green')

    # Relative Position Vectors (Edge Attributes)
    ax.text(10, 5, '相对位置向量', ha='center', va='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    ax.arrow(7, 6, 2.5, -0.5, head_width=0.2, head_length=0.1, fc='yellow', ec='yellow')

    # Graph Data Structure
    ax.text(7, 3.5, '图数据结构', ha='center', va='center', fontsize=12, bbox=dict(facecolor='purple', alpha=0.5))
    ax.arrow(7, 4, 0, -0.7, head_width=0.2, head_length=0.1, fc='purple', ec='purple')

    plt.show()


# Drawing the deep learning structure
draw_deep_learning_structure()
