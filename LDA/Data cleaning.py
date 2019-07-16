#encoding=utf-8
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import gensim
import pandas as pd
import csv
from gensim import corpora
import sys
import re


tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
en_stopset = set()
for i in en_stop:
	en_stopset.add(i)
# print(en_stopset)
p_stemmer = PorterStemmer()

csvfile = open('F:/Label recommendation/ctrsr_datasets/citeulike-a/raw-data.csv','r',encoding='utf-8',errors = 'ignore')
#打开一个文件
file_handle=open('F:/Label recommendation/ctrsr_datasets/citeulike-a/Data cleaning.txt','w')
#定义一个变量，进行读取
readCSV = csv.reader(csvfile)
abstract = []
sklda_corpus=[]
for line in readCSV:
	abstract.append(line[0],line[4])
# print(abstract)
for doc in abstract:
	tokens = tokenizer.tokenize(doc)#分词
	stopped_tokens = [w for w in tokens if w not in en_stop]#停用词处理
	# print(stopped_tokens)
	stemmed_tokens = [p_stemmer.stem(w) for w in stopped_tokens]#词干还原
	# print(stemmed_tokens)
	a=" ".join(stemmed_tokens)#连接句子
	sklda_corpus.append(a)
# print(corpus)
# print(sklda_corpus)#得到一个语句的语料，去掉了停用词  

# abstract = " \n".join(abstract)
#写入txt文件
with open("F:/Label recommendation/ctrsr_datasets/citeulike-a/Data cleaning.txt","w",encoding='utf-8',errors = 'ignore') as f:
	for text in sklda_corpus:
		f.write(text+'\n')

		# print (tokens)
		# for i in tokens:
		# 	new_word=filter(str.isalpha,i)
		# 	print(new_word)

#选取数据集中为摘要的部分
# text=[]
# for doc in readCSV['raw.abstract']:
# 	tokens = tokenizer.tokenize(doc)#分词
# 	abstract=[]
# 		#提取数据集中的字母部分
# 	for i in tokens:
# 		new_word=filter(str.isalpha,i)
# 		print(new_word)
	# docs=row
	# # print(row)
	# corpus=[]
	# sklda_corpus=[]
	# for doc in docs:
	#     tokens = tokenizer.tokenize(doc)#分词
	#     tokens1=[]
	#     for x in tokens:
 #    		if not x.isdigit():
 #        		# print (x)
 #        		tokens1.append(x)
	#     # b=[]
	#     # for w in tokens:
 #    	# 	a=filter(lambda x:x.isalpha(),w)
 #    	# 	if len(list(a))>0:
 #     # 		   b.append(a)
	    # print(b)
	    # print(tokens)
	#     stopped_tokens = [w for w in tokens1 if w not in en_stop]#去停用词
	#     # stopped_tokens = [w for w in tokens if w not in en_stop]#去停用词
	#     # print(stopped_tokens)
	#     stemmed_tokens = [p_stemmer.stem(w) for w in stopped_tokens]#词干还原
	#     # print(stemmed_tokens)
	#     #连接句子
	#     a=" ".join(stemmed_tokens)
	#     sklda_corpus.append(a)
	#     corpus.append(stemmed_tokens)
	# # print(corpus)
	# print(sklda_corpus)#得到一个语句的语料，去掉了停用词  、


	# #把print输出到一个txt文件中
	# output=sys.stdout
	# outputfile=open("F:/Label recommendation/ctrsr_datasets/citeulike-a/shuzi.txt","a",encoding='utf-8',errors = 'ignore')
	# sys.stdout=outputfile



	# # for i in range(len(sklda_corpus)):
	# # 	file_handle.write(sklda_corpus[i]+'\n')