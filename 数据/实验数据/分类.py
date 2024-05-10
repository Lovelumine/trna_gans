import pandas as pd

# 读取带有分类信息的CSV文件
csv_file_path = 'combination_counts_with_category.csv'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 对Category列进行分组并对Count列进行求和
category_counts = df.groupby('Category')['Count'].sum()

# 打印结果
print(category_counts)
