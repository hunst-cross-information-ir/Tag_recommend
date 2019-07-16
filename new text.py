# encoding=utf-8
import csv
# print(abs_lines[1].split(',')[0])
abs_num_file = open('F:/Label recommendation/ctrsr_datasets/citeulike-a/raw-data.csv', 'r', encoding='utf-8',errors='ignore')
#text
readabs = csv.reader(abs_num_file)
tag_num_file = open('F:/Label recommendation/ctrsr_datasets/citeulike-a/item-tag.dat', 'r', encoding='utf-8',errors='ignore')
readtag = csv.reader(tag_num_file)
tag_lines = tag_num_file.readlines()
abs_lines = abs_num_file.readlines()
out2 = open('F:\\Label recommendation\\TFIDF\\boost_ID.txt','w')
add = []
i = 1
for tag_line in tag_lines:
    if len(tag_line) >= 5:
        add.append(i)
        i = i + 1
        out2.write(tag_line)
    else:
        i = i + 1
# print(add)
j = 1
out1 = open('F:\\Label recommendation\\TFIDF\\boost_text.txt','w',encoding='utf-8')
out1.write(abs_lines[0])
for abs_line in abs_lines[1:]:
    temp = abs_line.split(',')[0]
    # print(type(abs_line))
    if int(temp) in add:
        # print(temp)
        out1.write(str(j)+','+' '.join(abs_line.split(',',1)[1:]))
        j += 1        


    # out1.write(abs_line+'\n')
#
# final_text1 = []
# final_text2 = []
# # print(tag_lines[add[0]])
# j = 0
# out1 = open('boost_text.txt','w')
# out2 = open('boost_ID.txt','w')
# for i in range(0, len(add)):
#     final_text1.append(tag_lines[add[i]].strip())
#     final_text2.append(str(j) + ' ' + str(abs_lines[add[i]].split(',')[1:]))
#     # print(abs_lines[1].split(',')[0])
#     j += 1
# for k in range(26):
#     # print(final_text1[k])
#     print(final_text2[k])
# # print(abs_lines[1].split(',')[1:])
# print(final_text2)
# out1.write(str(final_text1) + '\n')
# out2.write(str(final_text2))


# print(abs_lines[add[1]])

