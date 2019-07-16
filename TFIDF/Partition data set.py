from sklearn.model_selection import train_test_split
import pandas as pd
import pandas
import csv
import sys

csvfile = open('F:/Label recommendation/ctrsr_datasets/citeulike-a/Data cleaning.txt','r',encoding='utf-8',errors = 'ignore')
file_handle=open('F:/Label recommendation/TFIDF/xunlianjiID.txt','w')
file_handle=open('F:/Label recommendation/TFIDF/ceshijiID.txt','w')

doc=[]
for i in csvfile:
	doc.append(i)

train, test = train_test_split(doc, test_size = 0.2)
# print(train)
with open("F:/Label recommendation/TFIDF/xunlianjiID.txt","w",encoding='utf-8',errors = 'ignore') as f:
	for text in train:
		f.write(text)
with open("F:/Label recommendation/TFIDF/ceshijiID.txt","w",encoding='utf-8',errors = 'ignore') as f:
	for text in test:
		f.write(text)
