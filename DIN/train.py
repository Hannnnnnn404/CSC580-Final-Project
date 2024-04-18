import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt
from DIN.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat, get_feature_names)
from DIN.models.din import DIN
import tensorflow as tf


def load_data():
    movies = pd.read_table('../ml-1m/movies.dat',
                           encoding='unicode_escape',
                           sep='::',
                           header=None,
                           names=['movieId', 'title', 'genres'],
                           engine='python')

    movies.head()
    users = pd.read_table("../ml-1m/users.dat", sep="::", engine="python",
                          header=None,
                          names=["userId", "gender", "age", "occupation", "zip"]
                          )
    users.head()
    ratings = pd.read_table("../ml-1m/ratings.dat", sep="::", engine="python",
                            header=None,
                            names=["userId", "movieId", "rating", "timestamp"]
                            )
    ratings.head()
    # merge
    data = pd.merge(pd.merge(ratings, users), movies, on="movieId")
    data.head()
    return data


def num_unique_features(data):
    feat_nunique = {}
    for feature in data.columns:
        try:
            feat_nunique[feature] = data[feature].nunique()
        except:
            print(feature + ' unable to count!')

    return pd.DataFrame(feat_nunique, index=['num_unique'])


def get_user_hist_beavior(df):
    """
    user data
    """
    # List of movies each user watch recently 50 movies
    # Group by useId, make a last 50 list of movieId user_hist_data =df{user_id, [50 movieId list]}
    user_hist_data = pd.DataFrame(
        df.groupby("userId")["movieId"].agg({list})["list"].map(lambda x: x[-50:])).reset_index()
    # Number of movies each user watched, add col hist_len
    user_hist_data["hist_len"] = user_hist_data["list"].map(lambda x: len(x))
    # the same as number of movies, add col seq_length = hist_len
    user_hist_data["seq_length"] = user_hist_data["hist_len"].map(lambda x: x)
    user_hist_data.rename(columns={"list": "hist_movieId"}, inplace=True)
    # 原表合并
    df = df.merge(user_hist_data, how="left", on="userId")
    return df


# 分割函数：为每个用户分割数据
def split_train_test(user_group, train_ratio):
    split_idx = int(len(user_group) * train_ratio)  # 计算80%的索引位置
    return user_group.iloc[:split_idx], user_group.iloc[split_idx:]


def get_feats_columns_info(df, spase_columns, dense_columns, behavior_feat, his_behavior_fea, embed_dim=8, max_len=50):
    """
    获取特征编码信息
    df: 数据集
    spase_columns: 稀疏特征, list
    dense_columns： 稠密特征, list
    behavior_feat： 候选物品, list
    embed_dim： embedding维度, int
    maxlen: 序列最大长度, int
    """

    behavior_feature_list = behavior_feat  # List of history feature_name: [movieId, len]
    feature_columns = []  # feature embedding list

    # SpaseFeat
    for feat_name in spase_columns:  # 稀疏的特征的名字["userId", "movieId", "gender", "age", "occupation", "zip"]
        feature_columns.append(SparseFeat(feat_name, max(df[feat_name]) + 1,
                                          embedding_dim=8))  # 添加一个有10000个可能值的user_id稀疏特征，每个用户ID都会被映射到一个8维的嵌入向量。

    # DenseFeat
    for name in dense_columns:  # 稠密的特征的名字["timestamp", "hist_len"]
        feature_columns.append(DenseFeat(name=name, dimension=1))  # 添加一个有timestamp的稀疏特征，每个连续的时刻都会被映射到一个1维的嵌入向量

    # VarLenSpaseFeat 可变长空间特征嵌入，特征的长度不是固定的，而是变化的。在推荐系统中，用户的历史点击记录可能对每个用户来说长度都不同。
    # 例如 userId=0的用户只有42个历史点击，但是userId=1有50个历史点击
    # 用户历史记录，feature name = hist_+"movieId" = hist_movieId
    for hist_name in behavior_feature_list:  # hist_name = "movieId"
        feature_columns.append(VarLenSparseFeat(SparseFeat("hist_" + hist_name
                                                           , vocabulary_size=max(df[hist_name]) + 1
                                                           , embedding_dim=embed_dim
                                                           , embedding_name=hist_name
                                                           )
                                                , maxlen=max_len
                                                , length_name="seq_length")
                               )

    # 模型的输入数据X
    x = {}
    for name in get_feature_names(feature_columns):  # 从data中获取一个feature name
        if name in his_behavior_fea:  # 这个feature name是his_behavior_feature(hist_movieId)
            # 每一位用户的行为序列编码
            his_list = [l for l in df[name]]  # [[用户0看过的电影的id], [用户1看过的电影的id], [...], ...]
            x[name] = np.array(df[name].apply(lambda x: np.pad(x, (0, max_len - len(x)), 'constant', constant_values=0)).tolist())
        else:
            x[name] = df[name].values
    return feature_columns, x


if __name__ == '__main__':
    data = load_data()
    num_unique_data = num_unique_features(data)
    # dense feature: normalize
    ContinuousFeatures = ['timestamp']
    MMS = MinMaxScaler()
    for feature in ContinuousFeatures:
        data[feature] = MMS.fit_transform(data[[feature]].to_numpy())

    # Other variables are discrete, user_id->index
    DiscreteFeatures = ['userId', 'movieId', 'gender', 'age', 'occupation', 'zip']
    LE = LabelEncoder()
    for feature in DiscreteFeatures:
        data[feature] = LE.fit_transform(data[[feature]].to_numpy())

    # First, sort the DataFrame by 'userId' and 'timestamp' so that the most recent interactions are last
    data = data.sort_values(by=['userId', 'timestamp'])
    # Split Train and Test
    split_data = data.groupby('userId', group_keys=False).apply(split_train_test, 0.8)
    # split_data 现在包含了每个用户的训练集和测试集，需要分别提取
    # 提取训练集和测试集
    train_data = pd.concat([train for train, test in split_data])
    test_data = pd.concat([test for train, test in split_data])
    print(train_data.shape, test_data.shape)

    train_data = get_user_hist_beavior(df=train_data)
    train_data["label"] = train_data["rating"].map(lambda x: 1 if x > 3 else 0)
    test_data["label"] = test_data["rating"].map(lambda x: 1 if x > 3 else 0)

    # 从train_data中获取feature_columns和输入数据X
    feature_columns, X = get_feats_columns_info(train_data
                                                , spase_columns=DiscreteFeatures
                                                , dense_columns=ContinuousFeatures + ["hist_len"]
                                                , behavior_feat=["movieId"]
                                                , his_behavior_fea=["hist_movieId"]
                                                )
    # print(feature_columns)
    # [SparseFeat(name='user_id', vocabulary_size=6040, embedding_dim=8, use_hash=False, dtype='int32', embedding_name='user_id', group_name='default_group'),
    # SparseFeat(name='movie_id', vocabulary_size=3706, embedding_dim=8, use_hash=False, dtype='int32', embedding_name='movie_id', group_name='default_group'),
    # SparseFeat(name='gender', vocabulary_size=2, embedding_dim=8, use_hash=False, dtype='int32', embedding_name='gender', group_name='default_group'),
    # SparseFeat(name='age', vocabulary_size=7, embedding_dim=8, use_hash=False, dtype='int32', embedding_name='age', group_name='default_group'),
    # SparseFeat(name='occupation', vocabulary_size=21, embedding_dim=8, use_hash=False, dtype='int32', embedding_name='occupation', group_name='default_group'),
    # SparseFeat(name='zip', vocabulary_size=3439, embedding_dim=8, use_hash=False, dtype='int32', embedding_name='zip', group_name='default_group'),
    # DenseFeat(name='timestamp', dimension=1, dtype='float32'),
    # DenseFeat(name='hist_len', dimension=1, dtype='float32'),
    # VarLenSparseFeat(sparsefeat=SparseFeat(name='hist_movieId', vocabulary_size=3706, embedding_dim=8, use_hash=False, dtype='int32', embedding_name='movie_id', group_name='default_group'), maxlen=50, combiner='mean', length_name='seq_length'),

    behavior_feature_list = ["movieId"]  # history feature name
    y = train_data["label"].values

    # feature name 如下
    # ['user_id',
    #  'movie_id',
    #  'gender',
    #  'age',
    #  'occupation',
    #  'zip',
    #  'timestamp',
    #  'hist_len',
    #  'hist_movieId',
    #  'seq_length']

    # X是训练集，作为模型的输入
    # print(x)
    # x的格式如下，是一个字典: {feature name: np array of data}
    # {'user_id': array([0, 0, 0, ..., 6039, 6039, 6039]),
    #  'movie_id': array([2969, 1178, 1574, ..., 1741, 155, 1131]),
    #  'gender': array([0, 0, 0, ..., 1, 1, 1]),
    #  'age': array([0, 0, 0, ..., 2, 2, 2]),
    #  'occupation': array([10, 10, 10, ..., 6, 6, 6]),
    #  'zip': array([1588, 1588, 1588, ..., 466, 466, 466]),
    #  'timestamp': array([0.24062316, 0.24062356, 0.24062356, ..., 0.4540416, 0.45404184,
    #                      0.46363028]),
    #  'hist_len': array([50, 50, 50, ..., 50, 50, 50]),  pad过后长度相同
    #  'hist_movieId': array([[957, 2147, 1658, ..., 1439, 1727, 47],
    #                          [957, 2147, 1658, ..., 1439, 1727, 47],
    #                          [957, 2147, 1658, ..., 1439, 1727, 47],
    #                          ...,
    #                          [3313, 1132, 2711, ..., 1741, 155, 1131],
    #                          [3313, 1132, 2711, ..., 1741, 155, 1131],
    #                          [3313, 1132, 2711, ..., 1741, 155, 1131]], dtype=int32), 每一个user的看过的最后50个电影的id
    #  'seq_length': array([50, 50, 50, ..., 50, 50, 50]),

    device = 'cpu'
    use_cuda = False
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    model.compile('adagrad', 'binary_crossentropy', metrics=['binary_crossentropy', "auc"])
    # print(model)
    # 训练
    history = model.fit(X, y, batch_size=1024, epochs=10, shuffle=False, verbose=2, validation_split=0.2)

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.savefig("./imgs/loss.png")
    plt.show()
