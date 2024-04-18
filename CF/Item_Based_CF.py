import random
import json
import pandas as pd
from collections import defaultdict
from operator import itemgetter
import math
from sklearn.model_selection import train_test_split


def LoadReviewsData(file_path, train_rate):
    data = []

    df = pd.read_csv(file_path, sep=',', encoding='utf-8')

    ratings = df[["userId", "movieId", "rating", "timestamp"]]
    ratings = ratings[['userId', 'movieId']]
    train = []
    test = []
    random.seed(42)
    for idx, row in ratings.iterrows():
        user = row['userId']
        item = row['movieId']
        if random.random() < train_rate:
            train.append([user, item])
        else:
            test.append([user, item])
    # 使用train_test_split来分割数据集
    # train, test = train_test_split(ratings, train_size=train_size, random_state=3)
    return PreProcessData(train), PreProcessData(test)


def PreProcessData(originData):
    """
    建立User-Item表，结构如下：
        {"User1": {ItemID1, ItemID2, ItemID3,...}
         "User2": {ItemID12, ItemID5, ItemID8,...}
         ...
        }
    """
    trainData = dict()  # 字典
    for user, item in originData:
        trainData.setdefault(user, set())
        trainData[user].add(item)
    return trainData


class ItemCF(object):
    """ Item based Collaborative Filtering Algorithm Implementation"""

    def __init__(self, trainData, similarity="cosine", norm=True):
        self._trainData = trainData
        self._similarity = similarity
        self._isNorm = norm
        self._itemSimMatrix = dict()  # 物品相似度矩阵

    def similarity(self):
        # 实现共现矩阵
        N = defaultdict(int)  # 记录每个物品的喜爱人数，创建一个defaultdict，指定默认类型为int，{Item1: 3}
        for user, items in self._trainData.items():  # "User1": {ItemID1, ItemID2, ItemID3,...}
            for i in items:  # {ItemID1, ItemID2, ItemID3,...}
                self._itemSimMatrix.setdefault(i, dict())  # itemSimMatrix[ItemID1]
                N[i] += 1  # N[ItemID1] += 1
                for j in items:
                    if i == j:
                        continue
                    # i != j
                    self._itemSimMatrix[i].setdefault(j, 0)  # itemSimMatrix[ItemID1] = 0
                    if self._similarity == "cosine":
                        self._itemSimMatrix[i][j] += 1  # itemSimMatrix[ItemID1] + 1
                    elif self._similarity == "iuf":
                        self._itemSimMatrix[i][j] += 1. / math.log1p(len(items) * 1.)

        count = 0
        for n, item in N.items():
            if N[n] != 1:
                count += 1
        print(count)
        print(len(N.keys()))
        for i, related_items in self._itemSimMatrix.items():
            for j, cij in related_items.items():
                self._itemSimMatrix[i][j] = cij / math.sqrt(N[i] * N[j])

        # 是否要标准化物品相似度矩阵
        if self._isNorm:
            for i, relations in self._itemSimMatrix.items():
                if relations:
                    max_num = relations[max(relations, key=relations.get)]
                else:
                    print(relations)
                # 对字典进行归一化操作之后返回新的字典
                self._itemSimMatrix[i] = {k: v / max_num for k, v in relations.items()}

    def recommend(self, user, N, K):
        """
        :param user: 被推荐的用户user
        :param N: 推荐的商品个数
        :param K: 查找的最相似的用户个数
        :return: 按照user对推荐物品的感兴趣程度排序的N个商品
        """
        recommends = dict()
        # 先获取user的喜爱物品列表
        items = self._trainData[user]
        for item in items:
            # 对每个用户喜爱物品在物品相似矩阵中找到与其最相似的K个
            for i, sim in sorted(self._itemSimMatrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                if i in items:
                    continue  # 如果与user喜爱的物品重复了，则直接跳过
                recommends.setdefault(i, 0.)
                recommends[i] += sim
        # 根据被推荐物品的相似度逆序排列，然后推荐前N个物品给到用户
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

    def train(self):
        self.similarity()


if __name__ == "__main__":
    train, test = LoadReviewsData('../ml-latest-small/ratings.csv', 0.8)
    print("train data size: %d, test data size: %d" % (len(train), len(test)))
    ItemCF = ItemCF(train, similarity='cosine', norm=True)
    ItemCF.train()

    # 分别对以下4个用户进行物品推荐
    print(ItemCF.recommend(1, 10, 10))
