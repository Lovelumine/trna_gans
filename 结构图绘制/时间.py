import matplotlib.pyplot as plt
import pandas as pd

# Ensure that matplotlib can display Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # Specify a font that supports Chinese.
plt.rcParams['axes.unicode_minus'] = False

# Create DataFrame
data = {
    "Model": ["RNAJP", "Vfold-Pipeline", "trRosettaRNA", "RhoFlod", "IsRNA1",
              "3dRNA", "SimRNA", "RNAcomposer", "Vfold3D", "MC-Sym", "本模型"],
    "Min Time (min)": [120, 10, 1, 1, 40, 90, 120, 1, 10, 1, 1],
    "Max Time (min)": [480, 120, 120, 10, 480, 90, 480, 5, 120, 120, 1]
}
df = pd.DataFrame(data)

# Plotting the data
fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size

# Plot each model's time range
for i in range(len(df)):
    ax.plot([df.loc[i, "Min Time (min)"], df.loc[i, "Max Time (min)"]], [i, i],
            marker='o', label=df.loc[i, "Model"])

ax.set_yticks(range(len(df)))
ax.set_yticklabels(df["Model"], fontsize=12)  # Bigger font for Y-axis labels
ax.invert_yaxis()  # Fastest models at the top

ax.set_xlabel("时间（分钟）", fontsize=14)  # Bigger font for X-axis label
ax.set_title("RNA结构预测时间比较", fontsize=16)  # Bigger font for title

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)  # Bigger font for legend
plt.tight_layout()

plt.show()
