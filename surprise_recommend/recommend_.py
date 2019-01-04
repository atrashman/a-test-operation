# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:17:20 2018

@author: abc
"""
#SVD算法构架的推荐系统

import numpy as np
import time
import math
import os
from surprise.model_selection import KFold
from surprise import accuracy, KNNBasic, Reader
from collections import defaultdict
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import NMF, KNNBaseline


def get_top_n(predictions, n=10, threshold = 3.5):
    '''从一组预测中返回每个用户的前n个推荐.

    Args:
        从一组预测(预测对象列表)中返回每个用户的前n个建议推荐:预测列表，由算法的测试方法返回。

        n(int):每个用户的推荐输出数量。默认的是10。

    Returns:
    一个字典 keys是user (raw) ids and 值为 为 lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''
    # 每个用户对应推荐物品的映射.
    top_n_est = defaultdict(list)
    true_ratings = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:  #predictions包含uid、iid、r_ui（真实评分）、 est（预测评分）
        top_n_est[uid].append((iid, est))   #物品id 预测评分
        true_ratings[uid].append((iid, true_r))  #物品id真实评分
        #print(true_ratings)
    # k个最高排列
    for uid, user_ratings in top_n_est.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        user_ratings = [x for x in user_ratings if x[1] > threshold]
        top_n_est[uid] = user_ratings[:n]       # top n
        # add 0 if less than n
        est_len = len(top_n_est[uid])
        if est_len < n:
            for i in range(est_len, n):
                top_n_est[uid].append(('0', 0)) # append 0 if not enough
    # 每个用户rating排行和K个最高
    for uid, user_ratings in true_ratings.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        true_ratings[uid] = [x for x in user_ratings if x[1] > threshold]          # len
    return top_n_est, true_ratings


def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg if idcg != 0 else 0

#
def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg


def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0


def RR(ranked_list, ground_list):
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0


def precision_and_racall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits / (1.0 * len(ranked_list) if len(ground_list) != 0 else 1)
    rec = hits / (1.0 * len(ground_list) if len(ground_list) != 0 else 1)
    return pre, rec
'''
def evaluate(top_n_est, true_ratings):
    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    for u, user_ratings in top_n_est.items():
        tmp_r = top_n_est.get(u)  # [('302', 4.2889227920390836), ('258', 3.9492992642799027)]
        tmp_t = true_ratings.get(u) #[('196',3.52986)]
        tmp_r_list = []  #
        tmp_t_list = []
        for (item, rate) in tmp_r:
            tmp_r_list.append(item)
        for (item, rate) in tmp_t:
            tmp_t_list.append(item)
        #tmp_r_list  tmp_t_list  都装用户名
        #print(tmp_r_list, "-->", tmp_t_list)
        #评价函数
        pre, rec = precision_and_racall(tmp_r_list, tmp_t_list)#
        ap = AP(tmp_r_list, tmp_t_list)#
        rr = RR(tmp_r_list, tmp_t_list)  #
        ndcg = nDCG(tmp_r_list, tmp_t_list)
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(ndcg)
    precison = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    f1 = 2 * precison * recall / (precison + recall)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1, map, mrr, mndcg
'''
#评价函数
def evaluate_model_new(model, test_user, test_item, test_rate, top_n):
    recommend_dict = {}#建立推荐字典
    for u in test_user:
        recommend_dict[u] = {}
        for i in test_item:
            pred = model.predict(str(u), str(i), r_ui=4)
            est_str = '{est:1.4f}'.format(est=pred.est)
            recommend_dict[u][i] = float(est_str)
    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    for u in test_user:
        #按照预测评分排序
        tmp_r = sorted(recommend_dict[u].items(), key = lambda x:x[1], reverse=True)[
                0:min(len(recommend_dict[u]), top_n)]
        #按照真实评分排序
        tmp_t = sorted(test_rate[u].items(), key = lambda x:x[1], reverse=True)[
                0:min(len(test_rate[u]), len(test_rate[u]))]
        tmp_r_list = []
        tmp_t_list = []
        for (item, rate) in tmp_r:
            tmp_r_list.append(item)

        for (item, rate) in tmp_t:
            tmp_t_list.append(item)
        #print(tmp_r_list, "-->", tmp_t_list)

        pre, rec = precision_and_racall(tmp_r_list, tmp_t_list)
        ap = AP(tmp_r_list, tmp_t_list)
        rr = RR(tmp_r_list, tmp_t_list)
        ndcg = nDCG(tmp_r_list, tmp_t_list)
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(ndcg)
    precison = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    # print(precison, recall)
    f1 = 2 * precison * recall / (precison + recall)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1, map, mrr, mndcg

#读取文件并且分三列的函数
def read_data(filename):
    users, items, rates = set(), set(), {}
    with open(filename, "r") as fin:
        line = fin.readline()
        while line:
            user, item, rate ,time= line.strip().split('\t')
            if rates.get(user) is None:
                rates[user] = {}
            rates[user][item] = float(rate)
            users.add(user)
            items.add(item)
            line = fin.readline()
    return users, items, rates


def main(rec= 'SVD',threshold = 4,topK = 10):
    # First train an SVD algorithm on the movielens dataset.
    print("load data...")
    '''
    data = Dataset.load_builtin('ml-1m')
    # test set is made of 40% of the ratings.
    test_size = 0.4
    trainset, testset = train_test_split(data, test_size=test_size)
    '''
    
    # path to dataset file
    test_data_path = r'C:\Users\abc\.surprise_data\ml-100k\ml-100k\u.data'    #这个还不知道干嘛用
    file_path = os.path.expanduser(r'C:\Users\abc\.surprise_data\ml-100k\ml-100k\u.data')
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    
    test_user, test_item, test_rate = read_data(test_data_path)  #分为三组
    #print("test size %.1f..." % test_size)
    print("training...")
    
    sim_options = {'name': 'cosine',
                   'user_based': False  # 计算物品相似度
                   }
    #选择算法
    if rec == 'NMF':
        algo = NMF()
    elif rec == 'SVD':
        algo = SVD()
        name = ['SVD']
    else:
        algo = KNNBaseline(sim_options=sim_options)
        name = ['ItemKNN']
    
    train_start = time.time()
    algo.fit(trainset)
    train_end = time.time()
    print('train time:%.1f s' % (train_end - train_start))
    
    #Than predict ratings for all pairs (u, i) that are NOT in the training set.
    ######填充空值，预测trainset的值
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    
    test_end = time.time()
    print('test time:%.1f s' % (test_end - train_end))
    #top_n_est 是元组列表，元组里边是itemid 和 对应预测评分
    top_n_est, true_ratings = get_top_n(predictions, n=10, threshold = threshold)
    #模型评估
    f1, map, mrr, mndcg = evaluate_model_new(algo, test_user, test_item, test_rate, topK)
    eval_end = time.time()
    print('evaluate time:%.1f s' % (eval_end - test_end))
    print("algorithm : %s" % rec)
    print('recommendation metrics: F1 : %0.4f, NDCG : %0.4f, MAP : %0.4f, MRR : %0.4f' % (f1, mndcg, map, mrr))
    print('%0.4f个用户' % algo.pu.shape)
    print('%0.4f个物品'% algo.qi.shape)
    return top_n_est
# Print the recommended items for each user
'''
for uid, user_ratings in top_n_est.items():
    print(uid, [iid for (iid, _) in user_ratings])
print("#" * 150)
for uid, user_ratings in true_ratings.items():
    print(uid, [iid for (iid, _) in user_ratings])
'''
'''
原文：https://blog.csdn.net/Dooonald/article/details/80787159 
'''
