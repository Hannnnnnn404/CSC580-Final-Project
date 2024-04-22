import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler



def get_neighbors(similarity_matrix, num_neighbors):
    top_k_similar = {}
    for i in range(similarity_matrix.shape[0]):
        # 从大到小排序索引
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        # Select the k most similar movies
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

def split_train_test(user_group, train_ratio):
    split_idx = int(len(user_group) * train_ratio)
    return user_group.iloc[:split_idx], user_group.iloc[split_idx:]


if __name__ == '__main__':
    movie_ratings = pd.read_table("../ml-1m/ratings.dat", sep="::", engine="python",
                                  header=None,
                                  names=["userId", "movieId", "rating", "timestamp"]
                                 )
    movie_ratings = movie_ratings.sort_values(by=['userId', 'timestamp'])
    # Timestamp is a continuous variable
    ContinuousFeatures = ['timestamp']
    MMS = MinMaxScaler()
    for feature in ContinuousFeatures:
        movie_ratings[feature] = MMS.fit_transform(movie_ratings[[feature]].to_numpy())
    # Other variables are discrete
    DiscreteFeatures = ['userId', 'movieId']
    LE = LabelEncoder()
    for feature in DiscreteFeatures:
        movie_ratings[feature] = LE.fit_transform(movie_ratings[[feature]].to_numpy())

    # Split train and test data by user
    split_data = movie_ratings.groupby('userId', group_keys=False).apply(split_train_test, 0.8)
    test_data = pd.concat([test for train, test in split_data])
    print(test_data.shape)
    Train_ratings = movie_ratings[['rating', 'userId', 'movieId']]
    # Transform
    Train_ratings = Train_ratings.pivot_table(index='movieId', columns='userId',
                                              values='rating')
    Train_ratings = Train_ratings.fillna(-1)
    # movie_id_to_index_mapping = pd.Series(range(0, len(Train_ratings)), index=Train_ratings.index).to_dict()

    Test_movie_ratings = test_data[['rating', 'userId', 'movieId']]  # dataframe
    for index, row in Test_movie_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']
        Train_ratings.at[movie_id, user_id] = -1

    # Train
    similarity_matrix = cosine_similarity(Train_ratings.to_numpy())
    num_neighbor = 10
    top_k_similar = get_neighbors(similarity_matrix, num_neighbor)
    predict_ratings = predict_rating(Train_ratings.to_numpy(), top_k_similar)

    # Test and Evaluate, MAE
    N = Test_movie_ratings.shape[0]
    loss_rating = 0
    prediction_list = []
    for index, row in Test_movie_ratings.iterrows():
        user_id = int(row['userId'])
        movie_id = int(row['movieId'])
        rating = row['rating']
        prediction_list.append(predict_ratings[movie_id, user_id-1])
        loss_rating += abs(rating - predict_ratings[movie_id, user_id-1])
    MAE = loss_rating / N
    print(MAE)

    # Tune the parameter of neighborhood numbers
    num_neighbors_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
    MAE_list = []
    predictions_list ={}
    recommendations_list ={}

    for num_neighbor in num_neighbors_list:
        top_k_similar = get_neighbors(similarity_matrix, num_neighbor)
        predict_ratings = predict_rating(Train_ratings.to_numpy(),
                                         top_k_similar)

        N = Test_movie_ratings.shape[0]
        loss_rating = 0
        prediction_list = []

        for index, row in Test_movie_ratings.iterrows():
            user_id = int(row['userId'])
            movie_id = int(row['movieId'])
            rating = row['rating']
            prediction_list.append(predict_ratings[movie_id, user_id - 1])
            loss_rating += abs(rating - predict_ratings[movie_id, user_id - 1])
        MAE_list.append(loss_rating / N)
        print(MAE_list)
        Test_movie_ratings['prediction'] = prediction_list
        sorted_test_data = Test_movie_ratings.sort_values(['userId', 'prediction'], ascending=[True, False])
        top10_rec_movies = sorted_test_data.groupby('userId').head(10)
        top10_rec_movies.to_csv('../Evaluation/top10_rec_movies_' + str(num_neighbor) + '.csv', index=False)
        predictions_list[num_neighbor] = prediction_list

    MAE_df = pd.DataFrame(list(zip(num_neighbors_list, MAE_list)),
                          columns=['No. of Neighbors', 'MAE'])

    print(MAE_df)

    plt.scatter(MAE_df['No. of Neighbors'], MAE_df['MAE'])
    plt.plot(MAE_df['No. of Neighbors'], MAE_df['MAE'])
    plt.xlabel('No. of Neighbors')
    plt.ylabel('MAE')
    plt.savefig('../Evaluation/MAE_ItemCF.png')

    df_predictions = pd.DataFrame(predictions_list)
    test_data_with_predictions = pd.concat([Test_movie_ratings, df_predictions], axis=1, join='inner')
    test_data_with_predictions.to_csv('../Evaluation/test_data_with_predictions.csv', index=False)








