import pandas as pd


def data_preprocessing(file_path):
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')
    print(df.head())  # 打印DataFrame的前几行以验证
    return df


rating_path = '../ml-latest-small/ratings.csv'
data_preprocessing(rating_path)
