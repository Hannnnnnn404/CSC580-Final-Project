import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def get_neighbors(similarity_matrix, num_neighbors):
    top_k_similar = {}
    for i in range(similarity_matrix.shape[0]):
        # 从大到小排序索引
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        # 排除电影本身，选择 top-k
        top_k_list = [(index, similarity_matrix[i][index]) for index in sorted_indices if index != i][:num_neighbors]
        top_k_similar[i] = top_k_list
    return top_k_similar


def predict_rating(ratings, top_k_similar):
    # 复制原始评分矩阵以防修改原始数据
    predicted_ratings = np.copy(ratings)

    num_movies, num_users = ratings.shape

    # 遍历每个用户和电影
    for user in range(num_users):
        for movie in range(num_movies):
            if ratings[movie, user] == -1:
                # 对于未观看的电影，找到相似的电影并计算预测评分
                similar_movies = top_k_similar.get(movie, [])  # [(1, 0.5), (2, 0.2)]
                weighted_sum = 0
                similarity_sum = 0
                for sim_movie, sim_score in similar_movies:
                    if ratings[sim_movie, user] != -1:
                        weighted_sum += ratings[sim_movie, user] * sim_score
                        similarity_sum += sim_score

                # 如果有有效的相似电影，计算加权平均评分
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    predicted_ratings[movie, user] = predicted_rating
                else:
                    predicted_ratings[movie, user] = 0
    return predicted_ratings  # np array


def recommend_movies(user, num_recommend_movies, predict_ratings, ratings):
    print('The list of the movies {} has watched \n'.format(user))
    # 找出用户未看过的电影（评分为 -1 的电影）
    unseen_movies = np.where(ratings[:, user] == -1)[0]
    # 获取这些未看过的电影的预测评分
    unseen_movies_ratings = predict_ratings[unseen_movies, user]

    # 获取预测评分最高的10个电影的索引
    top_k_indices = np.argsort(unseen_movies_ratings)[-num_recommend_movies:][::-1]

    # 生成推荐列表，包括电影的索引和相应的预测评分
    recommendations = [(unseen_movies[i], predict_ratings[unseen_movies[i], user]) for i in top_k_indices]
    return recommendations


if __name__ == '__main__':
    movie_ratings = pd.read_csv('../ml-latest-small/ratings.csv')
    # Split Train/Test dataset
    Train_ratings = movie_ratings[['rating', 'userId', 'movieId']]
    Train_ratings = Train_ratings.pivot_table(index='movieId', columns='userId',
                                              values='rating')  # We have the same users who bought the same item more than once
    Train_ratings = Train_ratings.fillna(-1)  # dataframe <movie, user>矩阵
    movie_id_to_index_mapping = pd.Series(range(0, len(Train_ratings)), index=Train_ratings.index).to_dict()

    test_size_ratio = 0.2
    train_set, Test_movie_ratings = train_test_split(movie_ratings, test_size=test_size_ratio, random_state=42)
    Test_movie_ratings = Test_movie_ratings[['rating', 'userId', 'movieId']]  # dataframe
    for index, row in Test_movie_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']
        Train_ratings.at[movie_id, user_id] = -1

    # Train
    similarity_matrix = cosine_similarity(Train_ratings.to_numpy())
    num_neighbor = 20
    top_k_similar = get_neighbors(similarity_matrix, num_neighbor)
    predict_ratings = predict_rating(Train_ratings.to_numpy(), top_k_similar)

    # Test and Evaluate, MAE
    N = Test_movie_ratings.shape[0]
    loss_rating = 0
    for index, row in Test_movie_ratings.iterrows():
        user_id = int(row['userId'])
        movie_id = int(row['movieId'])
        rating = row['rating']
        loss_rating += abs(rating - predict_ratings[movie_id_to_index_mapping[movie_id], user_id-1])
    MAE = loss_rating / N
    print(MAE)

    # plt




