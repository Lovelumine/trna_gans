import pandas as pd
from Bio import Entrez

# 请替换成你的电子邮件地址
Entrez.email = "shendekoudai@lovelumine.com"


def get_taxonomy(organism):
    print(f"正在查询：{organism}")
    try:
        search_handle = Entrez.esearch(db="taxonomy", term=organism, retmode="xml")
        search_results = Entrez.read(search_handle)
        search_handle.close()

        if search_results["Count"] == "0":
            print(f"未找到：{organism}")
            return "未知"

        tax_id = search_results["IdList"][0]
        print(f"{organism}的Tax ID：{tax_id}")

        fetch_handle = Entrez.efetch(db="taxonomy", id=tax_id, retmode="xml")
        fetch_results = Entrez.read(fetch_handle)
        fetch_handle.close()

        category = "未知"
        for lineage in fetch_results[0]["LineageEx"]:
            if lineage["Rank"] in ["superkingdom", "clade"]:
                category = lineage["ScientificName"]
                break

        print(f"{organism}分类为：{category}")
        return category
    except Exception as e:
        print(f"查询{organism}时出错：{str(e)}")
        return "查询失败"


# 处理后的CSV文件路径
csv_file_path = 'data_with_pdb_status.csv'
# 输出CSV文件的路径
output_csv_file_path = 'combination_counts_with_category.csv'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 根据第四列（tRNA type）和第五列（Organism）的组合进行分组统计
combination_counts = df.groupby(['tRNA type', 'Organism']).size().reset_index(name='Count')

# 为每个组合查询分类信息并添加到新列
combination_counts['Category'] = combination_counts['Organism'].apply(get_taxonomy)

# 保存统计结果到CSV文件
combination_counts.to_csv(output_csv_file_path, index=False)

print(f"组合数量及分类结果已保存到 '{output_csv_file_path}'。")
