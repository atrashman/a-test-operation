函数实现：
load_moivelen() 
从目录下读取数据，并且设置好列名
input ：无 
output：movies和rating的dataframe
to_digit()
设置索引 删除rating的timestamp标签
input ： rating和movies数据的dataframe
pivot_rating（） 
要输入movies_id的列名（测试中用的是mid）、评分的列名（rating）、user的列名（uid）
topN_favor（）
得到每个用户评分最高的N个物品
Input：ratingdataframe   N
Output：每个用户topN个物品集合
Usersimilarity（）
计算余弦相似度
Input ：ratingdataframe 和 参数top_k
返回：用户余弦相似度矩阵  每个用户最相近的N个用户
UserCF()
Input :  
Output：推荐结果的inner_id
result()
字典映射最终结果
input：data_movies   ,recommend_result
output:最终推荐结果
执行代码
#数据预处理和参数
top_k = 5#推荐top_k个
N = 5#每个人最喜欢的N个商品
data_movies,data_rating = load_moivelen()
data_rating,data_movies=to_digit(data_movies,data_rating)
data_rating = pivot_rating(data_rating,'uid','mid','rating')
data_rating = data_rating.fillna(0)
#each_m_label ,total_label = classification(data_movies,'mid','label','|')
#each_m_label_num = genres2num(data_movies,total_label,each_m_label)
#得到余弦相似度和最近top_k个用户
similarity, top_user = UserSimilarity(data_rating,top_k) 
#每个人最喜欢的N个商品
each_user_topNrating_item = topN_favor(data_rating,N) 
#推荐结果
num_goods = data_rating.shape[1];
recommend_result  =UserCF(each_user_topNrating_item,top_user,similarity,num_goods,top_k)
results = result(data_movies,recommend_result)

