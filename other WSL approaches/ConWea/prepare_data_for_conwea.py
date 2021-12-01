# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:36:28 2021

@author: room5
"""
import pickle
import numpy as np
import pandas as pd
from rake_nltk import Rake

def read_pickles(pickle_name):
	name = pickle_name
	with open(name + ".pkl", "rb") as f:
		label_info = pickle.load(f)
	f.close()

	return label_info


def read_pickles_2(pickle_name):
	name = pickle_name
	with open(name + ".pickle", "rb") as f:
		label_info = pickle.load(f)
	f.close()

	return label_info

def read_txt(f):
    file = open(f)
    top_labels = list()
    for line in file:
        top_labels.append(line[:-1])

    return top_labels



################################ Read Abstract and labels for our data ###########################
labels= read_txt(r'D:\Google Drive\AMULET\62_previous_host_labels.txt')
#data = read_txt('D:\Google Drive\AMULET\pure_zero_shot_test_set_top100.txt')

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

#print(len(x_true))
#print(y_true[0])

#################################### Change multi-word labels into one word ones #######################
new_labels_dict = dict()
y_true_single_words=list()
for label in y_true:
    if(label.__contains__('Whole')):
        y_true_single_words.append(label.split(" ")[1].split('-')[0].replace(",",""))
        new_labels_dict[label] = label.split(" ")[1].split('-')[0].replace(",","")
    elif(label.__contains__('Development')):
         y_true_single_words.append(label.split(" ")[0].split('-')[0].replace(",",""))
         new_labels_dict[label] = label.split(" ")[0].split('-')[0].replace(",","")
    elif(label.__contains__('Contraception')):
         y_true_single_words.append('Contraception')
         new_labels_dict[label] = 'Contraception'
    elif(label.__contains__('Cigarette Smoking')):
         y_true_single_words.append('Cigarette')
         new_labels_dict[label] = 'Cigarette'
    elif(label.__contains__('Marijuana Use')):
         y_true_single_words.append('Marijuana')
         new_labels_dict[label] = 'Marijuana'
    elif(label.__contains__('Sexual Health')):
         y_true_single_words.append('Sexual')
         new_labels_dict[label] = 'Sexual'
    elif(label.__contains__('Cancer Survivors')):
         y_true_single_words.append('Cancer')
         new_labels_dict[label] = 'Cancer'
    elif(label.__contains__('Protein Antibodies')):
         y_true_single_words.append('Antibodies')
         new_labels_dict[label] = 'Antibodies'
    elif(label.__contains__('Standing Position')):
         y_true_single_words.append('Position')
         new_labels_dict[label] = 'Position'
    elif(label.__contains__('Big Data')):
         y_true_single_words.append('Metadata')
         new_labels_dict[label] = 'Metadata'
    elif(label.__contains__('Non')):
         y_true_single_words.append('Nonsmokers')
         new_labels_dict[label] = 'Nonsmokers'
    elif(label.__contains__('Water Pipe Smoking')):
         y_true_single_words.append('Pipe')
         new_labels_dict[label] = 'Pipe'
    elif(label.__contains__('Biological Variation, Population')):
         y_true_single_words.append('Population')
         new_labels_dict[label] = 'Population'
    elif(label.__contains__('Gait')):
         y_true_single_words.append('Gait')
         new_labels_dict[label] = 'Gait'
    elif(label.__contains__('Cells')):
         y_true_single_words.append(label.split(" ")[0].split('-')[0].replace(",",""))
         new_labels_dict[label] = label.split(" ")[0].split('-')[0].replace(",","")
    elif(label.__contains__('Stress')):
         y_true_single_words.append(label.split(" ")[0].split('-')[0].replace(",",""))
         new_labels_dict[label] = label.split(" ")[0].split('-')[0].replace(",","")
    elif(len(label.split(" ")) == 2):
         y_true_single_words.append(label.split(" ")[1].split('-')[0].replace(",",""))
         new_labels_dict[label] = label.split(" ")[1].split('-')[0].replace(",","")
    else:
         y_true_single_words.append(label.split(" ")[0].split('-')[0].replace(",",""))
         new_labels_dict[label] = label.split(" ")[0].split('-')[0].replace(",","")



#%%
########################## Prepare df.pkl
cols = ['sentence','label']
lst = []
for i in range(0,len(y_true_single_words)):
    lst.append([x_true[i],y_true_single_words[i]])
df1 = pd.DataFrame(lst, columns=cols)
print(df1['sentence'])

with open('df.pkl', 'wb') as f:
      pickle.dump(df1, f)
      f.close()

#%%
####################### Prepare seedwords.json using Rake 
import json

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
    cleaned_keyword_dict[new_labels_dict[key]] = list()
    keyword_count = 0
    for keyword in keyword_dict[key]:
        if(keyword not in used_keywords):
            cleaned_keyword_dict[new_labels_dict[key]].append(keyword)
            keyword_count+=1
            used_keywords.append(keyword)
            if(keyword_count == 5):
                break
    


################################ Print the final dict #########################################

for key in cleaned_keyword_dict:
    print(len(cleaned_keyword_dict[key]))
    print(cleaned_keyword_dict[key])    
    



# Serializing json  
json_object = json.dumps(cleaned_keyword_dict, indent = 4) 
  
# Writing to sample.json 

with open("seedwords.json", "w") as outfile: 
    outfile.write(json_object) 






