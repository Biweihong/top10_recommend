import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# 加载数据集
train_file = 'training.txt'
test_file = 'test.txt'
output_file = '2023140721_result.txt'  # 更改为适当的输出文件名

# 读取训练数据
train_data = pd.read_csv(train_file, sep=' ', names=['user_id', 'item_id', 'click'])

# 读取测试数据
test_data = pd.read_csv(test_file, sep=' ', names=['user_id'])

# 使用Surprise库来处理数据
reader = Reader(rating_scale=(0, 1))
train_dataset = Dataset.load_from_df(train_data, reader)

# 划分训练集和验证集
trainset, valset = train_test_split(train_dataset, test_size=0.2, random_state=42)

# 使用SVD模型训练
algo = SVD()
algo.fit(trainset)

# 在验证集上验证推荐效果
predictions = algo.test(valset)


# 定义函数获取top-n推荐项
def get_top_n(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid in top_n:
            top_n[uid].append((iid, est))
        else:
            top_n[uid] = [(iid, est)]

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for iid, _ in user_ratings[:n]]

    return top_n


# 使用训练好的模型进行预测
test_user_ids = test_data['user_id'].unique()
top_n_recommendations = {}

# 基于矩阵分解的推荐
for user_id in test_user_ids:
    # 为每个用户预测top-10推荐列表
    testset = [(user_id, item_id, 0) for item_id in train_data['item_id'].unique()]
    user_predictions = algo.test(testset)
    user_recommendations = get_top_n(user_predictions, n=10)

    # 检查是否获得了足够的推荐项
    if len(user_recommendations[user_id]) < 10:
        # 基于统计意义的推荐方法
        popular_items = train_data['item_id'].value_counts().index.tolist()
        popular_items_count = 10 - len(user_recommendations[user_id])
        additional_items = popular_items[:popular_items_count]
        user_recommendations[user_id].extend(additional_items)

    top_n_recommendations[user_id] = user_recommendations[user_id]

# 将预测结果写入文本文件
with open(output_file, 'w') as file:
    for user_id, recommendations in top_n_recommendations.items():
        recommendation_str = ','.join(str(item_id) for item_id in recommendations)
        file.write(f"UserID: {user_id}, Recommendations: {recommendation_str}\n")
