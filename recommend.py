#coding=utf-8
import sys
import heapq,random
import numpy as np
import pandas as pd

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
test_cut,test_index=get_word_index('F:/Label recommendation/TFIDF/ceshijiID.txt')


similar = open('F:/Label recommendation/TFIDF/sim_sort.txt', 'r', encoding='utf-8',errors='ignore')
sim_sort = similar.readlines()
diction = open('F:/Label recommendation/TFIDF/true_tag.txt', 'r', encoding='utf-8',errors='ignore')
dictionary = diction.readlines()

values1 = []
dd = []
#建立了列表里面存储的是每一个文章的tag
for i in dictionary:
    string1 = str(i)
    string = string1.strip().split(" ",1)
    # print(string)
    # print('\n')
    dd.append(string[0])
    values1.extend(string[1:])
    # print(string[1:])
    # print('\n')
# print(values1[0])

#对应相似度
re={}
for i in sim_sort:
    string1 = str(i)
    temp = string1.strip().split(" ")
    docID = temp[0]
    # print(docID)
    docID2, sim=temp[1].split(':')
    # print(docID2 + '\n')
    # print(sim)
    docID3, sim2 = temp[3].split(':')
    re[docID] = {}
    re[docID][docID2] = float(sim)
    re[docID][docID3] = float(sim2)
# print(re)

dict={}


def merge(dict1,dict2,similarity):
    keys1=dict1.keys()
    keys2=dict2.keys()
    keys=set(list(keys1)+list(keys2))
    result1={}
    for key in keys:
        value=0
        if key in keys1:
            value+=dict1[key]
        if key in keys2:
            value+=dict2[key]*similarity
        result1[key]=value
    return result1

# print(dd[:-1])
# print(values1[1])
#文章单词权重初始值为1
num=0
for key in dd:
    dict[key]={}
    # print(type(d2))
    # print(d2[i])
    # print(num)
    # print(values1[num])
    tags = values1[num].split(' ')
    for tag in tags:
        dict[key][tag] = 1
    num += 1
# print(dict['10982'])
# print(dict)
# print(dictSim)
#

#{'12913': {'disaffinity': 0.3604323, 'triadic_closure': 0.3604323}}
#得到每个测试集的被推荐的标签的权重
expand={}
# print(re)
# print(re['7378'])
for doc in re.keys():
    expand[doc]={}
    # print(doc)
    try:
        for redoc in re[doc].keys():
            # print(dict[redoc])
            expand[doc]=merge(expand[doc],dict[redoc],similarity=re[doc][redoc])
            # print(expand)
    except:
        print(doc)
    # print(expand[doc])
    
#将权重排序，得到前n的标签
# tuple1 = []
dic_re_tag={}
for docs in expand.keys():
    retags=sorted(expand[docs].items(),key=lambda e:e[1])
    reTag=[t[0] for t in retags[-50:]]#改推荐标签数
    dic_re_tag[docs] = reTag
# print (dic_re_tag)
    # print(docs+':'+str(reTag))

#计算回归率
rf2 = open('F:/Label recommendation/TFIDF/true_tag.txt','r',encoding='utf-8',errors = 'ignore')
lines=rf2.readlines()
result = {}
all_recall = 0
ave = 0
for line in lines:
  paper = line.split()
  # print(line)
  if paper[0] in dic_re_tag:
        keyword_set = set()
        for word in paper[1:]:
            keyword_set.add(word)
        # print(paper[0],keyword_set)
        same = 0
        for word1 in dic_re_tag[paper[0]]:
            if word1 in keyword_set:
                same = same + 1
                # print('Yes')
        # print (same)
        recall = same/len(keyword_set)
        result[paper[0]] = recall
        all_recall = recall + all_recall
ave = all_recall/len(result)
# print(all_recall)
# print(result)
print(ave)
