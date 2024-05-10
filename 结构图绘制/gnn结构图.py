import matplotlib.pyplot as plt

# 设置中文和负号支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或使用 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和轴
fig, ax = plt.subplots()

# 输入层、隐藏层和输出层的节点数
input_nodes = 3
hidden_nodes = 5
output_nodes = 2

# 绘制输入层
for i in range(input_nodes):
    ax.plot([0], [i], 'o', label='输入层' if i == 0 else "", color='skyblue')

# 绘制隐藏层
for i in range(hidden_nodes):
    ax.plot([1], [i - (hidden_nodes - input_nodes) / 2], 'o', label='隐藏层' if i == 0 else "", color='lightgreen')

# 绘制输出层
for i in range(output_nodes):
    ax.plot([2], [i + 0.5 * (input_nodes - output_nodes)], 'o', label='输出层' if i == 0 else "", color='salmon')

# 设置图例和标题
ax.legend()
ax.set_title('深度学习结构示意图')

# 隐藏坐标轴
ax.axis('off')

plt.show()
