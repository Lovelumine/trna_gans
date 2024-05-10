import pandas as pd

# CSV文件路径
csv_file_path = 'data_with_pdb_status.csv'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 定义正常碱基
normal_bases = set('AUGC')

# 统计非正常碱基的种类和数量
def count_abnormal_bases(sequence):
    # 去除可能干扰统计的字符，如双引号
    sequence = sequence.replace('"', '')
    # 统计当前序列中非正常碱基的出现次数
    return {base: sequence.count(base) for base in set(sequence) - normal_bases}

# 初始化一个字典来累加所有序列中非正常碱基的数量
total_abnormal_bases_count = {}

# 遍历DataFrame中的每个序列
for sequence in df['Sequence']:
    # 对当前序列统计非正常碱基的数量
    count_dict = count_abnormal_bases(sequence)
    # 累加到总计数中
    for base, count in count_dict.items():
        if base in total_abnormal_bases_count:
            total_abnormal_bases_count[base] += count
        else:
            total_abnormal_bases_count[base] = count

# 将累加的结果转换为DataFrame，以便保存到CSV文件
abnormal_bases_df = pd.DataFrame(list(total_abnormal_bases_count.items()), columns=['非正常碱基', '数量'])

# 保存统计结果到CSV文件
abnormal_bases_df.to_csv('非标准碱基统计.csv', index=False)

# 控制台输出结果
print("非标准碱基统计结果已保存到 '非标准碱基统计.csv'。")
print("非正常碱基种类及数量如下：")
print(abnormal_bases_df)

# 打印所有非标准碱基为一个字符串
all_abnormal_bases_str = ''.join(total_abnormal_bases_count.keys())
print("所有非标准碱基：", all_abnormal_bases_str)
