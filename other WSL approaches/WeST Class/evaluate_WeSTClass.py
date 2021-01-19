# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 08:42:00 2020

@author: room5
"""

import numpy as np
import os
import pickle


def read_labels(f):
	file = open(f)
	top_labels = list()
	for line in file:
		top_labels.append(line[:-1])

	return top_labels


def create_test_set(path,top_labels,y_true_path):
	sentence_embeddings=list()
	for file in os.listdir(path):
		with open(path+file , "rb") as f:
			sentence_embeddings0 = pickle.load(f)
		f.close()
		sentence_embeddings=np.concatenate((sentence_embeddings,sentence_embeddings0), axis=0)
	test_file = y_true_path
	x_true = list()
	y = list()
	file = open(test_file)
	for line in file:
		y.append(line[2:-2].split("labels: #")[1])
		x_true.append(line[2:-2].split("labels: #")[0])

	y_true = []
	instances=list()
	for i in range(0,len(y)):
		string = ""
		flag = "false"
		for label in y[i].split("#"):
			if label in top_labels:#label_embeddings.keys():
				flag = "true"
				string = string + label + "#"
		if (flag == "false"):
			string = "None#"
			instances.append(i)
		y_true.append(string[:-1])
		
	
	sentence_embeddings=np.delete(sentence_embeddings,instances)
	y_true=np.delete(y_true,instances)
	print("x_test set size: "+ str(len(sentence_embeddings)))
	print("y_test set size: "+str(len(y_true)))

	return sentence_embeddings,y_true


def find_label_categories(y_true, brand_new, complex_new):
	brand_new_indexes=list()
	complex_new_indexes=list()
	for i in range(0,len(y_true)):
		for label in y_true[i]:
			if(label in brand_new):
				brand_new_indexes.append(i)
			if(label in complex_new):
				complex_new_indexes.append(i)
		
	print('Size of brand new instances into test set: ', len(brand_new_indexes))
	print('Size of complex new instances into test set: ', len(complex_new_indexes))

	return brand_new_indexes,complex_new_indexes



top_10_labels =  read_labels(r'D:\Google Drive\AMULET\62_previous_host_labels.txt')
test_embeddings,y_true = create_test_set(r'C:\Users\room5\PycharmProjects\use_PH_dataset\top100 embeddings/',top_10_labels,r"C:\Users\room5\PycharmProjects\Self_train_and_biozslmax\pure_zero_shot_test_set_top100.txt")
labels=read_labels('classes_for_eval.txt')
west_predictions=read_labels('out_cnn_agnews_params.txt')
brand_new=read_labels(r'D:\Google Drive\AMULET\completely_new_labels.txt')
complex_new=read_labels(r'D:\Google Drive\AMULET\new_labels_from_complex_changes.txt')

#%%
west_predictions_text_label_names=list()

for west in west_predictions:
	for label in labels:
		if(label.__contains__(west)):
			west_predictions_text_label_names.append([label.split(":")[1]])
			break

#%%
print(len(west_predictions_text_label_names))
for thing in west_predictions_text_label_names:
	print(thing)


#%%
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score

mlb = MultiLabelBinarizer()
mlb.fit(west_predictions_text_label_names)
y_pred=mlb.transform(west_predictions_text_label_names)
y_true_final=list()
for i in range(0,len(y_true)):
	y_true_final.append(y_true[i].split("#"))

brand_new_indices, complex_new_indices = find_label_categories(y_true_final, brand_new, complex_new)
y_test=mlb.transform(y_true_final)

brand_new_pred=list()
brand_new_true=list()
for index in brand_new_indices:
		brand_new_pred.append(y_pred[index])
		brand_new_true.append(y_test[index])
	
complex_new_pred=list()
complex_new_true=list()
for index in complex_new_indices:
		complex_new_pred.append(y_pred[index])
		complex_new_true.append(y_test[index])
		
		
print(f1_score(y_test,y_pred,average='macro'))
print(f1_score(brand_new_true,brand_new_pred,average='macro'))
print(f1_score(complex_new_true,complex_new_pred,average='macro'))



