from sklearn.model_selection import train_test_split
import pandas as pd
import pandas
import csv
import sys
datfile = open('F:\\Label recommendation\\ctrsr_datasets\\citeulike-a\\tags.dat','r',encoding='utf-8',errors = 'ignore')
readCSV1 = csv.reader(datfile)
dictionary = {}
i = 1
string1 = ['none']
dictionary[0] = string1
for line in readCSV1:
    dictionary[i] = line
    i += 1
csvfile = open('F:\\Label recommendation\\TFIDF\\boost_ID.txt','r',encoding='utf-8',errors = 'ignore')
readCSV2 = csv.reader(csvfile)
string = str
tagline = []

i = 0
out=open('F:\\Label recommendation\\TFIDF\\true_tag.txt','w')
k = 0
for line in readCSV2:
    k += 1
    for i in line:
        final_tag = []
        string = str(i)
        tagline = string.strip().split(" ")
        for j in tagline:
            final_tag.extend(dictionary[int(j)])
        # print(final_tag)
        a=str(k)+' '+' '.join(final_tag)
        out.write(a+'\n')













