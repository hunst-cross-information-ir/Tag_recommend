#encoding=utf-8
from nltk.tokenize import RegexpTokenizer
from gensim.models import LdaModel
from stop_words import get_stop_words
from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
import gensim
import numpy as np
import pandas as pd
from gensim import corpora

def get_word_index(filepath):
	rf = open(filepath,'r',encoding='utf-8',errors = 'ignore')
	lines=rf.readlines()
	en = []
	paper_index = []
	for line in lines:
		word = line.split()
		word_list = word[1:]
		paper_index.append(word[0])
		en.append(word_list)
	return en,paper_index
en,paper_index = get_word_index('F:/Label recommendation/LDA/xunlianjiID.txt')
# print (en)
# 创建语料的词语词典，每个单独的词语都会被赋予一个索引
dictionary = corpora.Dictionary(en)
# print(dictionary.keys(),dictionary.values())
corpus_model = [dictionary.doc2bow(doc) for doc in en]
# print(doc_term_matrix)
tfidf_model = models.TfidfModel(corpus_model)
corpus_tfidf = tfidf_model[corpus_model]

lda_model = models.LdaModel(corpus= corpus_model, id2word=dictionary, num_topics=5)
corpus_lda = lda_model[corpus_tfidf]

lda = LdaModel(corpus= corpus_model, id2word=dictionary, num_topics=5)
print(lda.print_topics(num_topics=5))

模型的保存/ 加载
lda.save('test_lda.model')
lda_load = models.ldamodel.LdaModel.load('test_lda.model')


# #改动之前
# # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
# doc_term_matrix = [dictionary.doc2bow(doc) for doc in en]
# # print(doc_term_matrix)
# lda_model = models.LdaModel(corpus= doc_term_matrix, id2word=dictionary, num_topics=7)
# corpus_lda = lda_model[doc_term_matrix]

# lda = LdaModel(corpus= doc_term_matrix, id2word=dictionary, num_topics=7)
# #模型的保存/ 加载
# lda.save('test_lda.model')
# lda_load = models.ldamodel.LdaModel.load('test_lda.model')




# test_cut,test_index=get_word_index('F:/Label recommendation/TFIDF/ceshijiID.txt')
# test_lda=[]
# for test_c in test_cut:
# 	test_bow = dictionary.doc2bow([word for word in test_c])
# 	test_lda = lda_model[test_bow]
# 	# test_lda.append(test_lda)
# 	# print (test_lda)
# # print (corpus_lda)

# 计算相似度
index = similarities.MatrixSimilarity(lda_model[corpus_lda]) #把所有评论做成索引
# print(index)
test_cut,test_index=get_word_index('F:/Label recommendation/LDA/ceshijiID.txt')
# test_lda=[]
sims_list = []
for test_c in test_cut:
	test_bow = dictionary.doc2bow([word for word in test_c])
	test_lda = lda_model[test_bow]
	sims = index[test_lda]  #利用索引计算每一条评论和商品描述之间的相似度
	sims_list.append(sims)
print(sims_list)

#提取相似度靠前的推荐文章
sim_array=np.array(sims_list)
# print(sim_array)
sim_array_frame=pd.DataFrame(sim_array,index=test_index,columns=paper_index)
# print(sim_array_frame.head(5))
# # cc=sorted(sim_array.loc['one'],reverse=True)

#写入排序前n的文章
out=open('F:/Label recommendation/LDA/sim_sort.txt','w')
rec_test_id_lists=[]
for i in test_index:
    rec_test_id=sim_array_frame.loc[i].sort_values(ascending=False)[:2].index.tolist()#改推荐文章数
    # print(rec_test_id)
    rec_test_id_lists.append(rec_test_id)
    rec_test_sim=[]
    for j in rec_test_id:
        rec_test_sim.append(sim_array_frame.loc[i][j])
    # print(rec_test_sim)
    rec_test=zip(rec_test_id,rec_test_sim)
    for temp in rec_test:
    	out.write(str(i)+' '+str(temp[0])+':'+str(temp[1])+' ')
    out.write('\n')
# print(rec_test_id_list)





# def lda_sim(s1,s2):
#     lda = models.ldamodel.LdaModel.load('test_lda.model')
#     test_doc = list(jieba.cut(s1))  # 新文档进行分词
#     dictionary=get_dict()[0]
#     doc_bow = dictionary.doc2bow(test_doc)  # 文档转换成bow
#     doc_lda = lda[doc_bow]  # 得到新文档的主题分布
#     # 输出新文档的主题分布
#     # print(doc_lda)
#     list_doc1 = [i[1] for i in doc_lda]
#     # print('list_doc1',list_doc1)

#     test_doc2 = list(jieba.cut(s2))  # 新文档进行分词
#     doc_bow2 = dictionary.doc2bow(test_doc2)  # 文档转换成bow
#     doc_lda2 = lda[doc_bow2]  # 得到新文档的主题分布
#     # 输出新文档的主题分布
#     # print(doc_lda)
#     list_doc2 = [i[1] for i in doc_lda2]
#     # print('list_doc2',list_doc2)
#     try:
#         sim = np.dot(list_doc1, list_doc2) / (np.linalg.norm(list_doc1) * np.linalg.norm(list_doc2))
#     except ValueError:
#         sim=0
#     #得到文档之间的相似度，越大表示越相近
#     return sim