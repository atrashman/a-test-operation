# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:52:41 2018

@author: abc
"""

import numpy as np
import pandas as pd
import math


#预处理
#传入dataframe 
#df = pd.read_excel(filename,header = 0)

#record数据样式

#召回率 准确率

#--------------------------------------------------------------------------

##提取数字部分 去除时间戳,将id变为索引 去除年龄列 改成二位索引
####################  rating部分  ##########################
    #######去除时间戳和年龄#############    看具体修改    ###
def load_moivelen():
    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] 
    data_movies = pd.read_csv('u.item', sep='|', names=m_cols, usecols=range(5),encoding='latin-1')
    data_rating = pd.read_csv('u.data',header = 0,names = ['uid','mid','rating','timestamp'], sep='\t')
    return data_movies,data_rating
    
def to_digit(data_movies,data_rating): #x为dataframe格式  
    data_rating.pop('timestamp')#具体看时间的标签名
    data_movies.pop('release_date')
    data_movies.pop('video_release_date')
    data_movies.pop('imdb_url')
    #data_rating = data_rating.set_index(['uid'])#具体看用户标签名 
    #data_movies = data_movies.set_index(['mid'])
    return data_rating,data_movies
    ###############################################
    
#################################### main_load_rating #########################################
def pivot_rating(data_rating,users,goods_id,rating): 
    return data_rating.pivot(columns = goods_id,values = rating,index = users)

def Non2zeros(data_rating):
    return data_rating.fillna(0)

def mid2m_name(data_movies):
    mid2name = defaultdict(str)
    mid = data_movies['movie_id']
    title = data_movies['title']
    for i in range(len(mid)):
        mid2name[i] = title[i]
    return mid2name
def result(data_movies,recommend_result):
    mid2name = mid2m_name(data_movies)
    results = []
    for i in recommend_result:
        temp = []
        for j in i:
            temp.append(mid2name[j])
        results.append(temp)
    return results
#100k的数据集没有这个        
###########################输出###### 用户商品交互的dataframe  ################################
    #x为用户 商品 评价  的两列dataframe  goods为标签列的标签名  标签列中元素rating为分割符号（comedy|tragedy）
    #返回用户/商品 二维表  元素为评级 
##################### 处理movies型 （商品标签） #################################
 #过程函数
     ########################################################
      #标签分离（有分隔符要去除）
'''
def classification(x,goods,label,div):
  label_series = x[label]
  temp_label = []
  temp_label1 = []
  for some_label in label_series:
      temp_label.append(some_label.split(div)) #
      temp_label1+=some_label.split(div)
  return temp_label,list(set(temp_label1)) 
'''
#返回每个电影对应id  和  所有标签集合

 # 将每个个案的标签变成数字列表形式
'''
def genres2num(df,list1,list2):
        temp = []
        for i,ii in enumerate(list2):
            temp1 = []
            for j in list1:
                if j in ii:
                    temp1.append(1)
                else :
                    temp1.append(0)   
            temp.append(temp1)
        return temp
'''
#################### main_load_movies #################################
#moives型数据
#load进 物品和标签 的两列dataframe
#good商品名  label标签  div标签分割符 （movies里面默认为 '|'）
#def goods_load(x,goods,label,div):  
#    genres_list,genres_list1 = classification(x,goods,label,div)   
#    a = genres2num(x,genres_list1,genres_list)
    #a = [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0], [1, 0, 0, 1, 1]]
    #将数字列表形式load进df里
#    for k in range(len(a[0])):
#        temp = [] 
#        for j in a:
#            temp.append(j[k])
#        x[genres_list1[k]] = temp  
#返回
###########################################################################



#测评指标
    #用户满意度
    #预测准确度(RMSE 和MAE)
def RMSE(records):
    return math.sqrt(sum([(rui-pui)*(rui-pui) 
                          for u,i,rui,pui in records]/float(len(records))))
'''备用：
def MAE(records):  
    return sum([abs(rui-pui) for u,i,rui,pui in records]) / float(len(records)) 
'''

#基于用户的协同过滤法
#计算两个用户的兴趣相似度，余弦相似度计算 推荐topN个同兴趣用户
#load进dataframe 型的改过的rating数据集  选择TOPk
#返回narray类型
def cos_similarity(rating,top_k):
    rate_value = rating.values#转成array
    #先转变为0/1矩阵 ,用rate_value01接收
    rate_value01 = rate_value
    index = np.nonzero(rate_value)
    for i in range(len(index[0])):
        rate_value01[index[0][i]][index[1][i]] = 1    #转成0/1型  
    #计算
    #temp = []#temp为每个用户与其他用户的交集个数和索引 (已经排序的)
    temp_ = []#temp_为每个用户与其他用户的交集个数和索引 （未排序的）
    for  i in rate_value01:     
        temp1 = []
        temp3 = []
        for j in rate_value01:            
            a = 0
            for k in range(len(i)):
                if i[k] == 1 and i[k] == j[k]:
                    a+=1
            temp1.append(a)
        #temp2 = list(enumerate(temp1))  #带编号        #temp3 = sorted(temp2, key=lambda x:x[1]) # list里面是元组
        temp_.append(temp1)
    #每个用户购买物件
    num_purchase = rate_value01.sum(axis = 1)   #每个用户买了多少件
    num_purchase1 = 1/num_purchase;#1维的
   
    num_purchase1 = np.expand_dims(num_purchase1,axis = 1)
    cos_similaritys = np.sqrt(num_purchase1)*(np.sqrt(num_purchase1).T*np.array(temp_)) #余弦相似度矩阵    
    temp2 = []
    temp3 = []
    for i in cos_similaritys:
        temp2+=(list(enumerate(i)))
        temp3.append(sorted(temp2, key=lambda x:x[1])) # list里面是元组
    topk = []
    a = 0
    for i in temp3:
        t = []
        times = 0
        for j in range(len(i)):
            if i[j][0] != a :
                t.append(i[j][0])
                times +=1
            if times == top_k:
                break
        topk.append(t)
        a += 1
    return cos_similaritys,pd.DataFrame(data = topk,index = range(len(topk)))       
###############################################################################
        #余弦相似度改进
def UserSimilarity(rating,top_k):  # build inverse table for item_users  
    #建立N（i）向量 ， 每个元素代表该商品多少人买
    ratings = rating.copy(deep = True)
    rating_values = ratings.values
    #0/1处理
    rating_value01 = rating_values
    index = np.nonzero(rating_values)
    for i in range(len(index[0])):
        rating_value01[index[0][i]][index[1][i]] = 1
    NI = []
    for i in range(rating_value01.shape[1]):
        NI.append(np.sum(rating_value01[:,i]))
    collect = []#为每个用户与其他用户的交集的商品编号
    for  i in rating_value01:
        temp2 = []
        a = 0
        for j in rating_value01:
            a+=1
            if i is not j:
                k = i+j
                temp1=(list(np.where(k == 2)[0]))  #where返回的是元组
            temp2.append(temp1)
       # print(a)
        collect.append(temp2)
    #[[[0, 1], [0], [], [0]],
    #[[0], [0, 2, 3], [2, 3], [0, 3]],
    #[[], [2, 3], [2, 3], [3]],
    #[[0], [0, 3], [3], [0, 3]]]
    #collect[0][0] = [0,1]  user0和user0的交集商品
    #做分子部分
    log1 = math.log(1)
    temp1 = []#用户交互NI
    for i in collect :
        temp = []
        for j in i:
            a = 0
            for index in j:
                a +=1/(NI[index]+log1)
            temp.append(a)
        temp1.append(temp)
    head = np.array(temp1)
    num_purchase = rating_value01.sum(axis = 1)   #每个用户买了多少件
    num_purchase1 = 1/num_purchase;#1维的
    num_purchase1 = np.expand_dims(num_purchase1,axis = 1)
    similarity = np.sqrt(num_purchase1)*(np.sqrt(num_purchase1).T*np.array(head))
    #选前N个
    temp3 = []
    for i in similarity:
        temp2 = []
        temp2+=(list(enumerate(i)))
        temp3.append(sorted(temp2, key=lambda x:x[1],reverse = True)) # list里面是元组
    topk = []
    for i in temp3:
        t = []
        times = 0
        for j in range(len(i)):
            if i[j][0] != a :
                t.append(i[j][0])
                times +=1
            if times == top_k:
                break
        topk.append(t)
    return similarity,pd.DataFrame(data = topk,index = range(len(topk))) 
    
###########################################################################
def topN_favor(rating,N): #每个用户评分最高的topN个商品
    goodsrating = []  #每个用户对每个商品的打分
    for i in rating.values:
        goodsrating.append(list(enumerate(i)))
    sort_goodsrating = []
    for i in goodsrating: #排一下序
        sort_goodsrating.append(sorted(i,key=lambda x:x[1],reverse = True))
    topnfavor = [] #选前topN个
    for i in sort_goodsrating:
        temp = i[0:N]
        topnfavor.append(temp)
    return pd.DataFrame(data = topnfavor,index = range(len(topnfavor)))
#返回dataframe每个元素都是元组  元组内第一个元素为商品编号第二个为打分
#需要注意的是上面的预先相似度可能会改了rating数据集
#rating 为改过的rating数据集userid/goods  top_user为以上余弦相似度选出的topk个用户集合 dataframe
#还需要输入商品总数  top_k为选择K个商品
def UserCF(top_favors,top_user,cos_similaritys,num_goods,top_k):
    values_top_user = top_user.values
    values_top_favors = top_favors.values
    
    temp1 = [] #每个用户的所有相似用户的top商品的集合
    for i in values_top_user:
        temp = []
        for j in i:  #找到人
             temp.append([j]+list(values_top_favors[j])) #推荐这个人的东西        
        temp1.append(temp)  #得到[推荐用户 、 推荐商品 ]
    #[[[2, (2, 1), (3, 1)], [1, (0, 1), (2, 1)]], [[2, (2, 1), (3, 1)], [0, (0, 1), (1, 1)]], [[0, (0, 1), (1, 1)], [1, (0, 1), (2, 1)]], [[2, (2, 1), (3, 1)], [0, (0, 1), (1, 1)]]]
    #[从哪个用户中得到 ,（商品，该用户对商品的评分）]
    for j in range(len(temp1)): #第J个用户
        for k in temp1[j]:
            for i in range(len(k)-1):  #根据一个用户推荐多少商品
                k[i+1] = list(k[i+1])
                k[i+1][1] = cos_similaritys[k[0]][j]*k[i+1][1]
    #得到temp1 = [[[2, [2, 1/cos], [3, 1/cos]],[1,[],[]] , ]]  被推用户 推荐物品 评分
    #用户对某个商品的希望被推荐度加起来
    s = [0*x  for x in range(num_goods)]
    s = list(enumerate(s))
    sum_ = []
    r = range(num_goods)
    for i in range(len(s)):
       s[i] = list(s[i])
    #s[[1,0],[2,0]...]
    for i in temp1:
        s = [0*x  for x in range(num_goods)]
        s = list(enumerate(s))
        for ss in range(len(s)):
            s[ss] = list(s[ss])
        temp3 = s
        for j in i:
            n = 0
            temp3[j[0]][1] +=j[n+1][1]
            n+=1
        sum_.append(temp3)

    #选择topN个用户中推荐最多的K个商品        
    #print(sum_)
    #转换为[推荐用户 、 推荐商品 、usercf评分]    
    sort_goods = []
    for i in sum_:
        sort_goods.append(sorted(i,key=lambda x:x[1],reverse = True))    
    
    topninterest = [] #选前topN个
    #去重
    for i in sort_goods:
        temp2 = []
        for j in range(top_k):            
            temp2.append(i[j][0])
        topninterest.append(temp2)   
    return topninterest
#输出为推荐的topN个商品
#####################################################################33
'''
            for n in range(len(j)-1):
                for k in r:
                    if j[n+1][0] == k:
                        temp3[k][1] +=j[n+1][1]
                        break
'''
'''
                temp_count = 0
                for k in i+j:  
                    if k == 2:
                        temp1.append(temp_count)
                    temp_count+=1 
''' 


#基于物品的协同过滤                    

    





#协同过滤法数据集  ：训练集提取和测试集提取
#K折交叉验证
from sklearn.model_selection import KFold
def K_FOLD(x,k):#x为dataframe类型的后端传输过来的
    traindata1 = []
    testdata1 = []
    kf = sklearn.model_selection.KFold(n_splits = k,shuffle = Flase,random_states = None)
    for train_index,test_index in kf.split(x):#x为narray对象
        traindata1.append(x[train_index])
        testdata1.append(x[test_index])
    return traindata1,testdata1
#使用方法返回数据为元组，元组中有两个列表，两个列表一一对应训练集和测试集  

    
