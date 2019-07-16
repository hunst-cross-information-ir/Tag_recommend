#coding=utf-8
import sys
import heapq,random
import numpy as np
import pandas as pd
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import CountVectorizer

# gensim的模型model模块，可以对corpus进行进一步的处理，比如tf-idf模型，lsi模型，lda模型等
rf = open('F:/Label recommendation/TFIDF/xunlianjiID.txt','r',encoding='utf-8',errors = 'ignore')
# file_handle=open('F:/Label recommendation/TFIDF/model.txt','w')
# file = open("F:/Label recommendation/TFIDF/sim.txt","w")
# for line in readCSV:
# 	abstract.append(line[2])
lines=rf.readlines()
# c=[]
# for line in lines:
# 	c.append(line[1])
# print(c)
#分词
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
en,paper_index = get_word_index('F:/Label recommendation/TFIDF/xunlianjiID.txt')
# print(en)
# en = []
# paper_index = []
# for line in lines:
# 	word = line.split()
# 	word_list = word[1:]
# 	paper_index.append(word[0])
# 	en.append(word_list)
# test_model = [[word for word in jieba.cut(words)] for words in wordstest_model]
# print(test_model)
# #为python中的字典对象, 其Key是字典中的词，其Val是词对应的唯一数值型ID 
dictionary = corpora.Dictionary(en,prune_at=2000000)
# # dictionary.save('F:/Label recommendation/TFIDF/model.dict.dict')
# # dictionary.save_as_text('F:/Label recommendation/TFIDF/save_as_text_dict.dict',  sort_by_word=True)
# for key in dictionary.iterkeys():
#     print (key,dictionary.get(key),dictionary.dfs[key])
# # 生成语料库中单个文档对应词id和词频
corpus_model= [dictionary.doc2bow(test) for test in en]
# print (corpus_model)

# # # 目前只是生成了一个模型,并不是将对应的corpus转化后的结果,里面存储有各个单词的词频，文频等信息
tfidf_model = models.TfidfModel(corpus_model)
# print(tfidf_model)
# # with open("F:/Label recommendation/ctrsr_datasets/citeulike-a/model.txt","w",encoding='utf-8',errors = 'ignore') as f:
# # 	f.write(tfidf_model)
# # 对语料生成tfidf
corpus_tfidf = tfidf_model[corpus_model]
# print(corpus_tfidf)
# #查看model中的内容
# for item in corpus_tfidf:
# 	print (item)

#使用测试文本来测试模型，提取关键词,test_bow提供当前文本词频，tfidf_model提供idf计算
test_cut,test_index=get_word_index('F:/Label recommendation/TFIDF/ceshijiID.txt')
# print (test_index)
test_tfidfs=[]
for test_c in test_cut:
	test_bow = dictionary.doc2bow([word for word in test_c])
	test_tfidf = tfidf_model[test_bow]
	test_tfidfs.append(test_tfidf)
# print (test_tfidfs)

# 计算相似度
index = similarities.MatrixSimilarity(corpus_tfidf) #把训练集做成索引
#print(index)
sims = index[test_tfidfs]  #利用索引计算每一条训练集和测试集之间的相似度
# print(sims)

#提取相似度靠前的推荐文章
sim_array=np.array(sims)
# print(sim_array)
sim_array_frame=pd.DataFrame(sim_array,index=test_index,columns=paper_index)
# print(sim_array_frame.head(5))
# cc=sorted(sim_array.loc['one'],reverse=True)

#写入排序前n的文章
out=open('F:/Label recommendation/TFIDF/sim_sort.txt','w')
rec_test_id_lists=[]
for i in test_index:
	#提取前n对应的训练集的文档id
    rec_test_id=sim_array_frame.loc[i].sort_values(ascending=False)[:2].index.tolist()#改推荐文章数
    # print(rec_test_id)
    rec_test_id_lists.append(rec_test_id)
    rec_test_sim=[]
    #提取对应id的相似度
    for j in rec_test_id:
        rec_test_sim.append(sim_array_frame.loc[i][j])
    # print(rec_test_sim)
    #依次取出每一个数组的元素，然后组合
    rec_test=zip(rec_test_id,rec_test_sim)
    for temp in rec_test:
    	out.write(str(i)+' '+str(temp[0])+':'+str(temp[1])+' ')
    out.write('\n')
# print(rec_test_id_list)







