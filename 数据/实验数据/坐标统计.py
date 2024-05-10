import os
import numpy as np
import matplotlib.pyplot as plt

# 用来保存所有坐标的列表
all_coords = []

#包含PDB文件的文件夹路径
pdb_folder_path = 'pdb'

# 遍历文件夹下所有的PDB文件
for pdb_file in os.listdir(pdb_folder_path):
    if pdb_file.endswith('.pdb'):
        with open(os.path.join(pdb_folder_path, pdb_file), 'r') as file:
            for line in file:
                if line.startswith('ATOM'):
                    # 解析x, y, z坐标并将其加入列表
                    x, y, z = map(float, [line[30:38], line[38:46], line[46:54]])
                    all_coords.append((x, y, z))

# 转换为NumPy数组以便处理
all_coords = np.array(all_coords)

# # 计算所有坐标的绝对值位置
# abs_coords = np.abs(all_coords)

# 找出所有坐标的最大值和最小值
min_coords = all_coords.min(axis=0)
max_coords = all_coords.max(axis=0)

# 控制台输出最大值和最小值
print(f"Min X: {min_coords[0]}, Min Y: {min_coords[1]}, Min Z: {min_coords[2]}")
print(f"Max X: {max_coords[0]}, Max Y: {max_coords[1]}, Max Z: {max_coords[2]}")

# 绘制三个维度的密度分布图
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# 分别为x, y, z坐标
labels = ['X coordinate', 'Y coordinate', 'Z coordinate']

for i in range(3):
    ax[i].hist(all_coords[:, i], bins=50, density=True, alpha=0.6, color='b')
    ax[i].set_title(f'Density Plot of {labels[i]}')
    ax[i].set_xlabel('Absolute Position')
    ax[i].set_ylabel('Density')

plt.tight_layout()
plt.savefig("coordinate_density_plots.png")
plt.show()
