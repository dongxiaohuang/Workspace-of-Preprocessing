#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://www.ioiogoo.cn/2018/05/31/%e4%bd%bf%e7%94%a8k-means%e5%8f%8atf-idf%e7%ae%97%e6%b3%95%e5%af%b9%e4%b8%ad%e6%96%87%e6%96%87%e6%9c%ac%e8%81%9a%e7%b1%bb%e5%b9%b6%e5%8f%af%e8%a7%86%e5%8c%96/


# In[3]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# from data_utils import *
import jieba
import matplotlib.pyplot as plt


# In[4]:


# bigram分词
# segment_bigram = lambda text: " ".join([word + text[idx + 1] for idx, word in enumerate(text) if idx &lt; len(text) - 1])
# 结巴中文分词
segment_jieba = lambda text: " ".join(jieba.cut(text))
 
'''
加载语料
'''
corpus = []
with open("demo_1.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # print(">>>", line)
        tmp = segment_jieba(line)
        # print(tmp)
        corpus.append(tmp)
 


# In[5]:


# corpus = [
# ...     'This is the first document.',
# ...     'This is the second second document.',
# ...     'And the third one.',
# ...     'Is this the first document?',
# ... ]
# vectorizer = CountVectorizer()
# transformer = TfidfTransformer()
# X = vectorizer.fit_transform(corpus)

# print(vectorizer.get_feature_names())
# print(X)
# print(transformer.fit_transform(X))


# In[6]:


'''
计算tf-idf设为权重
'''
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# print(tfidf[:50])


# In[7]:


''' 
获取词袋模型中的所有词语特征
如果特征数量非常多的情况下可以按照权重降维
'''
word = vectorizer.get_feature_names()


# In[8]:


''' 
导出权重，到这边就实现了将文字向量化的过程，矩阵中的每一行就是一个文档的向量表示
'''
tfidf_weight = tfidf.toarray()


# In[9]:


SSE = []  # 存放每次结果的误差平方和
for k in range(1,50):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(tfidf_weight)
    SSE.append(estimator.inertia_)
X = range(1,50)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')


# In[13]:


# 聚类
import math
n_clusters=int(math.sqrt(len(corpus)/2))
print("n_clusters is ", n_clusters)
kmeans = KMeans(n_clusters)
kmeans.fit(tfidf_weight)

# 显示聚类结果
# print(kmeans.cluster_centers_)

result_dict = {}
print(len(kmeans.labels_))
print(len(corpus))
for index, label in enumerate(kmeans.labels_, 0):
#     print(" ======== new item =========")
#     print("index: {}, label: {}".format(corpus[index], label))
    if label in result_dict.keys():
        tmp_list = result_dict[label]
#         print(corpus[index])
        tmp_list.extend([corpus[index]])
#         print(">>> tmp_list", tmp_list)
        result_dict[label] = tmp_list
    else:
        result_dict[label] = [corpus[index]]

print(result_dict)
# print(result_dict[0])
# print(result_dict[1])
 
# 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
# k-means的超参数n_clusters可以通过该值来评估
# print("inertia: {}".format(kmeans.inertia_))

# '''
# 可视化
# '''
# # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
# tsne = TSNE(n_components=2) # 降低到2维度，方便显示
# decomposition_data = tsne.fit_transform(tfidf_weight)
 
# x = []
# y = []
 
# for i in decomposition_data:
#     x.append(i[0])
#     y.append(i[1])
 
# fig = plt.figure(figsize=(10, 10))
# ax = plt.axes()
# plt.scatter(x, y, c=kmeans.labels_, marker="x")
# plt.xticks(())
# plt.yticks(())
# # plt.show()
# plt.savefig('./sample.png', aspect=1)

