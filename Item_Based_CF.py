import random
import json
import pandas as pd
from collections import defaultdict
from operator import itemgetter
import math


def LoadReviewsData(file_path, train_rate):
    data = []

    # 打开文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每行为JSON对象并添加到列表中
            json_object = json.loads(line)
            data.append(json_object)
    # 将列表转换为Pandas DataFrame
    df = pd.DataFrame(data)

    ratings = df[["user_id", "parent_asin", "rating", "timestamp"]]
    ratings = ratings[['user_id', 'parent_asin']]
    train = []
    test = []
    random.seed(3)
    for idx, row in ratings.iterrows():
        user = row['user_id']
        item = row['parent_asin']
        if random.random() < train_rate:
            train.append([user, item])
        else:
            test.append([user, item])
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
    train, test = LoadReviewsData("Amazon_Book_Reviews.jsonl", 0.8)
    print("train data size: %d, test data size: %d" % (len(train), len(test)))
    ItemCF = ItemCF(train, similarity='cosine', norm=True)
    ItemCF.train()

    # 分别对以下4个用户进行物品推荐
    print(ItemCF.recommend("AFKZENTNBQ7A7V7UXW5JJI6UGRYQ", 2, 10))
