
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 引入进度条库

# 确保pdb文件夹存在
pdb_folder = "pdb"
os.makedirs(pdb_folder, exist_ok=True)

# 读取URL列表
def read_urls():
    with open("链接.txt", "r") as file:
        urls = [line.strip() for line in file.readlines()]
    return urls

# 定义下载函数
def download_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # 尝试从Content-Disposition头中提取文件名
        content_disposition = response.headers.get('Content-Disposition', '')
        if 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"')
        else:
            filename = url.split("/")[-1] + ".pdb"

        filepath = os.path.join(pdb_folder, filename)

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"{filename} 下载完成，保存在 {pdb_folder} 文件夹中。")
        return url, None
    except Exception as e:
        print(f"下载 {url} 时发生错误: {e}")
        return url, e

# 使用ThreadPoolExecutor进行多线程下载
def download_files(urls):
    with ThreadPoolExecutor(max_workers=200) as executor:
        # 准备工作任务
        future_to_url = {executor.submit(download_url, url): url for url in urls}

        # 设置进度条
        for future in tqdm(as_completed(future_to_url), total=len(urls), desc="下载进度", unit="file"):
            url, error = future.result()
            if error is None:
                urls.remove(url)  # 删除成功下载的URL

        # 如果有失败的下载，重新写入文件
        if urls:
            with open("链接.txt", "w") as file:
                for url in urls:
                    file.write(url + "\n")
            print("部分下载失败，剩余的URLs 保留在 链接.txt 文件中以供重试。")
        else:
            print("所有文件均已成功下载。")
            os.remove("链接.txt")  # 如果所有文件都下载成功，则删除URL文件

# 执行下载
if __name__ == "__main__":
    urls = read_urls()
    download_files(urls)

