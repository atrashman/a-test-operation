����ʵ�֣�
load_moivelen() 
��Ŀ¼�¶�ȡ���ݣ��������ú�����
input ���� 
output��movies��rating��dataframe
to_digit()
�������� ɾ��rating��timestamp��ǩ
input �� rating��movies���ݵ�dataframe
pivot_rating���� 
Ҫ����movies_id���������������õ���mid�������ֵ�������rating����user��������uid��
topN_favor����
�õ�ÿ���û�������ߵ�N����Ʒ
Input��ratingdataframe   N
Output��ÿ���û�topN����Ʒ����
Usersimilarity����
�����������ƶ�
Input ��ratingdataframe �� ����top_k
���أ��û��������ƶȾ���  ÿ���û��������N���û�
UserCF()
Input :  
Output���Ƽ������inner_id
result()
�ֵ�ӳ�����ս��
input��data_movies   ,recommend_result
output:�����Ƽ����
ִ�д���
#����Ԥ����Ͳ���
top_k = 5#�Ƽ�top_k��
N = 5#ÿ������ϲ����N����Ʒ
data_movies,data_rating = load_moivelen()
data_rating,data_movies=to_digit(data_movies,data_rating)
data_rating = pivot_rating(data_rating,'uid','mid','rating')
data_rating = data_rating.fillna(0)
#each_m_label ,total_label = classification(data_movies,'mid','label','|')
#each_m_label_num = genres2num(data_movies,total_label,each_m_label)
#�õ��������ƶȺ����top_k���û�
similarity, top_user = UserSimilarity(data_rating,top_k) 
#ÿ������ϲ����N����Ʒ
each_user_topNrating_item = topN_favor(data_rating,N) 
#�Ƽ����
num_goods = data_rating.shape[1];
recommend_result  =UserCF(each_user_topNrating_item,top_user,similarity,num_goods,top_k)
results = result(data_movies,recommend_result)

