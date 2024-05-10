# generate_urls.py

# 定义URL模板和范围
url_template = "https://tpsic.igcz.poznan.pl/trna/pdb-download/{}"
start_id = 1
end_id = 2000

# 打开一个文本文件以写入URLs
with open("链接.txt", "w") as file:
    for i in range(start_id, end_id + 1):
        file.write(url_template.format(i) + "\n")

print("URLs 已经保存在 链接.txt.")
