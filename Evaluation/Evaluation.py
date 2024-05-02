import numpy as np
import pandas as pd

def precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    relevant_set = set(relevant)
    relevant_count = sum(1 for movie in recommended if movie in relevant_set)
    return relevant_count / len(recommended)

def average_precision(recommended, relevant):
    relevant_set = set(relevant)
    score = 0.0
    relevant_count = 0
    for i, movie in enumerate(recommended):
        if movie in relevant_set:
            relevant_count += 1
            score += relevant_count / (i + 1)
    return score / max(1, len(relevant))

def mean_average_precision(recommended, relevant):
    return np.mean([average_precision(rec, rel) for rec, rel in zip(recommended, relevant)])

def ndcg_at_k(recommended, relevant, k):
    relevant_set = set(relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    dcg = sum(1.0 / np.log2(i + 2) if recommended[i] in relevant_set else 0 for i in range(len(recommended[:k])))
    return dcg / idcg if idcg > 0 else 0

test_data = pd.read_csv('../Evaluation/test_data.csv')

test_data = test_data.groupby('userId').head(10)



top10_rec_movies_CF = pd.read_csv('../Evaluation/top10_rec_movies_200.csv')

top10_rec_movies_DIN = pd.read_csv('../Evaluation/Movie_Recommendations_DIN.csv')

relevants = []
for user_id, group in test_data.groupby('userId'):
    movie_id_list = group['movieId'].tolist()
    relevants.append(movie_id_list)

recommends_CF = []
for user_id, group in top10_rec_movies_CF.groupby('userId'):
    movie_id_list = group['movieId'].tolist()
    recommends_CF.append(movie_id_list)

recommends_DIN = []
for user_id, group in top10_rec_movies_DIN.groupby('userId'):
    movie_id_list = group['movieId'].tolist()
    recommends_DIN.append(movie_id_list)

precisions_CF = []
precisions_DIN = []
for k in [1, 3, 5, 8, 10]:
    precisions_at_k_CF = 0
    precisions_at_k_DIN = 0
    for i in range(len(relevants)):
        precisions_at_k_CF += precision_at_k(recommends_CF[i], relevants[i], k)
        precisions_at_k_DIN += precision_at_k(recommends_DIN[i], relevants[i], k)
    precisions_CF.append(round(precisions_at_k_CF / len(relevants), 2))
    precisions_DIN.append(round(precisions_at_k_DIN / len(relevants), 2))
print(precisions_CF)
print(precisions_DIN)


MAP_CF = mean_average_precision(recommends_CF, relevants)
MAP_DIN = mean_average_precision(recommends_DIN, relevants)
MAP_CF = round(MAP_CF, 2)
MAP_DIN = round(MAP_DIN, 2)
print(MAP_CF)
print(MAP_DIN)

NDCG_CF = []
NDCG_DIN = []

for k in [1, 3, 5, 8, 10]:
    NDCG_at_k_CF = 0
    NDCG_at_k_DIN = 0
    for i in range(len(relevants)):
        NDCG_at_k_CF += ndcg_at_k(recommends_CF[i], relevants[i], k)
        NDCG_at_k_DIN += ndcg_at_k(recommends_DIN[i], relevants[i], k)
    NDCG_CF.append(round(NDCG_at_k_CF / len(relevants), 2))
    NDCG_DIN.append(round(NDCG_at_k_DIN / len(relevants), 2))
print(NDCG_CF)
print(NDCG_DIN)

column_names = ['precision@1', 'precision@3', 'precision@5', 'precision@8', 'precision@10']
row_names = ['Item-based CF', 'DIN']

precisions = np.array((precisions_CF, precisions_DIN))
precisions_df = pd.DataFrame(data=precisions,
                             index=row_names,
                             columns=column_names)
precisions_df = precisions_df.rename_axis('Method').reset_index()

print(precisions_df)
precisions_df.to_csv('../Evaluation/Precisions.csv')

NDCG = np.array((NDCG_CF, NDCG_DIN))
NDCG_df = pd.DataFrame(data=NDCG,
                       index=row_names,
                       columns=column_names)

NDCG_df = NDCG_df.rename_axis('Method').reset_index()

print(NDCG_df)
NDCG_df.to_csv('../Evaluation/NDCG.csv')
