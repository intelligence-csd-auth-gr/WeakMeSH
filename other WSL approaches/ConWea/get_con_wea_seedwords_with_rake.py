# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:20:26 2021

@author: room5
"""

from rake_nltk import Rake
import numpy as np




def read_txt(f):
    file = open(f)
    top_labels = list()
    for line in file:
        top_labels.append(line[:-1])

    return top_labels




##################################### Read Abstracts and Labels for the dat
labels= read_txt(r'D:\Google Drive\AMULET\62_previous_host_labels.txt')
x_true = list()
y = list()
file = open('D:\Google Drive\AMULET\pure_zero_shot_test_set_top100.txt')
for line in file:
		y.append(line[2:-2].split("labels: #")[1])
		x_true.append(line[2:-2].split("labels: #")[0])
        


y_true = []
instances=list()
for i in range(0,len(y)):
	string = ""
	flag = "false"
	for label in y[i].split("#"):
		if label in labels:#label_embeddings.keys():
			flag = "true"
			string = string + label + "#"
	if (flag == "false"):
		string = "None#"
		instances.append(i)
	y_true.append(string[:-1].split("#")[0])

x_true=np.delete(x_true,instances)
y_true=np.delete(y_true,instances)


################################# Extract all 1 word keyphrases from each abstract and create the keyword dict
r = Rake()
keyword_dict = dict()
for i in range(0,len(y_true)):
    if y_true[i] not in keyword_dict:
        keyword_dict[y_true[i]] = set()
    r.extract_keywords_from_text(x_true[i])
    for keyword in r.get_ranked_phrases():
        if(len(keyword.split(" ")) == 1):
            keyword_dict[y_true[i]].add(keyword)



############################ Clean the keyword dict so 2 labels do not have the same keyword

cleaned_keyword_dict = dict()
used_keywords = list()
for key in keyword_dict.keys():
    cleaned_keyword_dict[key] = list()
    keyword_count = 0
    for keyword in keyword_dict[key]:
        if(keyword not in used_keywords):
            cleaned_keyword_dict[key].append(keyword)
            keyword_count+=1
            used_keywords.append(keyword)
            if(keyword_count == 5):
                break
    


################################ Print the final dict #########################################

for key in cleaned_keyword_dict:
    print(len(cleaned_keyword_dict[key]))
    print(cleaned_keyword_dict[key])
