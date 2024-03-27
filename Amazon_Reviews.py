import urllib.request
import gzip
import json
import pandas as pd

def uncompress_file(file_path, out_path):
    out_file = out_path
    file_path = file_path
    # Download archive
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:  # 使用文本模式并指定编码
            lines = [next(f) for _ in range(1000)]

        with open(out_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    except Exception as e:
        print(e)
        return 1


def data_preprocessing(file_path):
    # 初始化一个空列表来存储解析后的JSON数据
    data = []

    # 打开文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每行为JSON对象并添加到列表中
            json_object = json.loads(line)
            data.append(json_object)
    # 将列表转换为Pandas DataFrame
    df = pd.DataFrame(data)
    print(df.head())  # 打印DataFrame的前几行以验证
    return df



Reviews_file = 'Books.jsonl.gz'
Reviews_out_file = 'Amazon_Book_Reviews.jsonl'
# uncompress_file(Reviews_file, Reviews_out_file)

with open(Reviews_out_file, 'r') as fp:
    for line in fp:
        print(json.loads(line.strip()))
        break

Item_Meta_file = 'meta_Books.jsonl.gz'
Item_Meta_out_file = 'Amazon_Item_Meta.jsonl'
uncompress_file(Item_Meta_file, Item_Meta_out_file)

with open(Item_Meta_out_file, 'r') as f:
    for line in f:
        print(json.loads(line.strip()))
        break

Reviews_pd = data_preprocessing(Reviews_out_file)
Item_Mate_pd = data_preprocessing(Item_Meta_out_file)
