


import pickle
import numpy as np
import os
import pandas as pd
import collections 
from scipy.spatial.distance import  cosine
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import  OneVsRestClassifier
from skmultilearn.problem_transform import BinaryRelevance,ClassifierChain
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import contextlib
import time
import random
import sys
import biobert_embedding
import torch
from scipy.spatial import distance
import sklearn.metrics as skm


biobert = biobert_embedding.BiobertEmbedding()
@contextlib.contextmanager
def timer():
	"""Time the execution of a context block.

	Yields:
		None
	"""
	start = time.time()
	# Send control back to the context block
	yield
	end = time.time()
	print('Elapsed: {:.2f}s'.format(end - start))
	

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def arrayisin(array, list_of_arrays):
	for a in list_of_arrays:
		if np.array_equal(array, a):
			return True
	return False

def get_averaged_embedding(sentence_embeddings):
	final_sentence=[]
	for sentence in sentence_embeddings:
		if(len(sentence) !=0 ):
			if (len(final_sentence) == 0):
				final_sentence = np.array(sentence.cpu())
			else:
				final_sentence+=np.array(sentence.cpu())
	final_sentence=final_sentence/len(sentence_embeddings)
	
	return final_sentence

def get_averaged_embedding_with_titles(sentence_embeddings,title):
	final_sentence=[]
	for sentence in sentence_embeddings:
		if(len(sentence) !=0 ):
			if (len(final_sentence) == 0):
				final_sentence = np.array(sentence.cpu())
			else:
				final_sentence+=np.array(sentence.cpu())
				
	if(title != 'NoTitle'):
			final_sentence= final_sentence + np.array(biobert.sentence_vector(title))
			final_sentence = final_sentence / (len(sentence_embeddings)+1)
	else:
			final_sentence = final_sentence / len(sentence_embeddings)
	
	return final_sentence


def get_aveaged_test_set(sentence_embeddings):
	x=list()
	for i in range(0,len(sentence_embeddings)):
		final_sentence = []
		for sentence in sentence_embeddings[i]:
			if(len(sentence!=0)):
				if (len(final_sentence) == 0):
					final_sentence = np.array(sentence.cpu())
				else:
					final_sentence += np.array(sentence.cpu())
		final_sentence = final_sentence / len(sentence_embeddings[i])
		x.append(final_sentence)
	return x


def get_test_set_with_title(sentence_embeddings,titles):
	x=list()
	indices_without_titles=list()
	indices_with_titles=list()
	for i in range(0,len(sentence_embeddings)):
		if(titles[i] != 'NoTitle'):
			final_sentence=np.array(biobert.sentence_vector(titles[i]))
			indices_with_titles.append(i)
		else:
			final_sentence = np.zeros(768)
			indices_without_titles.append(i)
		x.append(final_sentence)
	return x, indices_with_titles, indices_without_titles


def get_aveaged_test_set_extended_with_titles(sentence_embeddings,titles,transform):
	x=list()
	indices_without_titles=list()
	indices_with_titles=list()
	for i in range(0,len(sentence_embeddings)):
		final_sentence = []
		for sentence in sentence_embeddings[i]:
			if(len(sentence!=0)):
				if (len(final_sentence) == 0):
					final_sentence = np.array(sentence.cpu())
				else:
					final_sentence += np.array(sentence.cpu())
		final_sentence = final_sentence / len(sentence_embeddings[i])			 
		if(titles[i] != 'NoTitle'):
				  final_sentence_with_titles = np.concatenate([final_sentence,np.array(biobert.sentence_vector(titles[i]))])
				  indices_with_titles.append(i)
		else:
				  final_sentence_with_titles = np.concatenate([final_sentence,np.zeros(768)])
				  indices_without_titles.append(i)
		   
		x.append(final_sentence_with_titles)
		   
	#print(len(x))
	print("Test set with titles size: "+ str(len(indices_with_titles)))
	print("Test set without titles size: "+ str(len(indices_without_titles)))
	return x,indices_with_titles,indices_without_titles

def read_labels(f):
	file = open(f)
	top_labels = list()
	for line in file:
		top_labels.append(line[:-1])

	return top_labels


def read_pickles(name):
	if '.pickle' in name:
		name = name[:-7]
	with open(name + ".pickle", "rb") as f:
		pickle_file = pickle.load(f)
	f.close()

	return pickle_file

def zero_shot_prediction_setn(x_test,label_embeddings,top_labels,threshold,y_true_final):
	predictions=list()
	for i in range(0,len(x_test)):
		labels=list()
		for label in top_labels:
				max_sim=0
				for sentence in x_test[i]:
					if(len(sentence) != 0):
						sim_label_sentence=1-cosine(label_embeddings[label],np.array(sentence.cpu()))
						if (sim_label_sentence >= max_sim):
							max_sim=sim_label_sentence
				if(max_sim >= threshold[label]):
					labels.append(label)
		if(len(labels)!=0):
			predictions.append(labels)
		else:
			predictions.append('None')
					
	print(len(predictions))
	mlb = MultiLabelBinarizer()
	mlb.fit(predictions)
	y_pred=mlb.transform(predictions)
	y_test=mlb.transform(y_true_final)
	return f1_score (y_test, y_pred,average='macro')

def read_train_set_pickles_from_similarities(top_labels,threshold,title_threshold,transformation_method,data_range, input2, choice = 'other'):
		final_x_train = list()
		final_y_train = list()
		final_x_train_titles= list()
		final_y_train_titles=list()
		instances_per_batch= list()
		if choice == 'other':
				path = main_path+'\\'+'Previous Host Train_Set for top 100 labels'
		else:
				path = r'D:\Google Drive\AMULET\COMPLEX CHANGES\Previous Host Train_Set for top 100 labels'
		counter = 0
		for i in os.listdir(path)[0:data_range]:
				print('reading similarity batch ', i)
				counter+=1
				if 'from ' in i:
						os.chdir(path + '\\' + i)
						print('I am into the folder: ', i)
						for j in os.listdir(os.getcwd()):
								if 'similarities' in j and 'title' not in j:
										similarities = read_pickles(j.split('.')[0])
								if 'title_similarities' in j:
										title_similarities = read_pickles(j.split('.')[0])
								if 'embeddings' in j:
										x_emb = read_pickles(j.split('.')[0])
								if 'titles_part' in j:
										train_titles=read_labels(j)
								if 'new_y' in j:
										y_cand = read_pickles(j.split('.')[0])
										#print(train_titles[0])
						if(transformation_method == 1):			
								x,y = read_weakly_labeled_similarity_extension_1(similarities, x_emb, threshold,top_labels, transformation_method, input2, train_titles,batch=i)
						if(transformation_method == 2):			
								x,y = read_weakly_labeled_similarity_extension_2(similarities ,x_emb, threshold,title_threshold,top_labels, transformation_method, input2, train_titles,batch=i)
						if(transformation_method == 3):
								x,y= read_weakly_labeled_similarity_extension_3(title_similarities, x_emb, title_threshold,top_labels, transformation_method, input2, train_titles,y_cand,batch=i)
						if(transformation_method == 4):			
								x,y,x_titl,y_titl = read_weakly_labeled_similarity_extension_4(similarities ,x_emb, threshold,title_threshold,top_labels, transformation_method, input2, train_titles,batch=i)		
								final_x_train_titles+=x_titl
								final_y_train_titles+=y_titl
						instances_per_batch.append(len(x))
						final_x_train+=x
						final_y_train+=y
						#del x,y,x_emb,x_titl,y_titl
		
		print("x_train size: "+str(len(final_x_train)))
		print("y_train size: "+str(len(final_y_train)))
		print("x_train size for titles: "+str(len(final_x_train_titles)))
		print("y_train size: "+str(len(final_y_train_titles)))
		return final_x_train,final_y_train,final_x_train_titles,final_y_train_titles,instances_per_batch
					 

	

def read_weakly_labeled_similarity_extension_1(similarities,x_train,threshold,top_labels,input3, input2, train_titles ,batch=None):
		new_x_train = list()
		new_y_train = list()
		labeled = dict()
		ind = list()
		for key in top_labels:
			for pair in similarities[key]:
					if (np.max(pair[0]) >= threshold[key]):
							if(pair[1] not in labeled.keys()):
									labeled[pair[1]] = list()
							labeled[pair[1]].append(key)
		
		for key in labeled.keys():
				if(train_titles[key] != 'NoTitle'):
					sentence_with_titles = np.concatenate([get_averaged_embedding(x_train[key]),np.array(biobert.sentence_vector(train_titles[key]))])
				else:
					sentence_with_titles = np.concatenate([get_averaged_embedding(x_train[key]),np.zeros(768)])
				
				new_x_train.append(sentence_with_titles)
				new_y_train.append(labeled[key])
				ind.append(key)
		
		print("x_train length from current batch:"+ str(len(new_x_train)))
		print("y_train length from current batch:"+ str(len(new_y_train)))
		if type(batch) == str:
			pd.DataFrame(ind).to_csv('indices_' + batch.split('_')[0][-4:] + '_mode_' + str(input3) + '_' + input2 + '.csv')
		return new_x_train, new_y_train
	




def read_weakly_labeled_similarity_extension_2(similarities,x_train,threshold,title_threshold,top_labels,input3, input2, train_titles ,batch=None):
		new_x_train = list()
		new_y_train = list()
		labeled = dict()
		ind = list()
		label_embeddings = read_pickles(r'D:\Google Drive\AMULET\various_files\label_embeddings')
		for key in top_labels:
			for pair in similarities[key]:
				title_similarity = 1- distance.cosine(label_embeddings[key],np.array(biobert.sentence_vector(train_titles[pair[1]])))
				if(title_similarity >= title_threshold[key]):
						if(pair[1] not in labeled.keys()):
								labeled[pair[1]] = list()
						labeled[pair[1]].append(key)
				if (np.max(pair[0]) >= threshold[key]):
							if(pair[1] not in labeled.keys()):
									labeled[pair[1]] = list()
							labeled[pair[1]].append(key)
				
		for key in labeled.keys():
				if(train_titles[key] != 'NoTitle'):
					sentence_with_titles = np.concatenate([get_averaged_embedding(x_train[key]),np.array(biobert.sentence_vector(train_titles[key]))])
				else:
					sentence_with_titles = np.concatenate([get_averaged_embedding(x_train[key]),np.zeros(768)])
				
				new_x_train.append(sentence_with_titles)
				new_y_train.append(labeled[key])
				ind.append(key)
		
		print("x_train length from current batch:"+ str(len(new_x_train)))
		print("y_train length from current batch:"+ str(len(new_y_train)))
		if type(batch) == str:
			pd.DataFrame(ind).to_csv('indices_' + batch.split('_')[0][-4:] + '_mode_' + str(input3) + '_' + input2 + '.csv')
		return new_x_train, new_y_train



def read_weakly_labeled_similarity_extension_3(similarities,x_train,title_threshold,top_labels,input3, input2, train_titles ,y_candidates,batch=None):
		new_x_train = list()
		new_y_train = list()
		labeled = dict()
		ind = list()
		for key in top_labels:
			for pair in similarities[key]:
				title_similarity = 1- distance.cosine(label_embeddings[key],np.array(biobert.sentence_vector(train_titles[pair[1]])))
				if(title_similarity >= title_threshold[key]):
						if(pair[1] not in labeled.keys()):
									labeled[pair[1]] = list()
						labeled[pair[1]].append(key)
	
						
		
		for key in labeled.keys():
				if(train_titles[key] != 'NoTitle'):
					sentence_with_titles = np.concatenate([get_averaged_embedding(x_train[key]),np.array(biobert.sentence_vector(train_titles[key]))])
				else:
					sentence_with_titles = np.concatenate([get_averaged_embedding(x_train[key]),np.zeros(768)])
				new_x_train.append(sentence_with_titles)
				new_y_train.append(labeled[key])
				ind.append(key)
		
		print("x_train length from current batch:"+ str(len(new_x_train)))
		print("y_train length from current batch:"+ str(len(new_y_train)))
		if type(batch) == str:
			pd.DataFrame(ind).to_csv('indices_' + batch.split('_')[0][-4:] + '_mode_' + str(input3) + '_' + input2 + '.csv')
		return new_x_train, new_y_train
	
def read_weakly_labeled_similarity_extension_4(similarities,x_train,threshold,title_threshold,top_labels,input3, input2, train_titles ,batch=None):
		new_x_train = list()
		new_y_train = list()
		labeled = dict()
		title_labeled=dict()
		ind = list()
		label_embeddings = read_pickles(r'D:\Google Drive\AMULET\various_files\label_embeddings')
		flag = 'false'
		for key in top_labels:
			for pair in similarities[key]:
				title_similarity = 1- distance.cosine(label_embeddings[key],np.array(biobert.sentence_vector(train_titles[pair[1]])))
				if(title_similarity >= title_threshold[key]):
						flag = 'true'
						if(pair[1] not in labeled.keys()):
								labeled[pair[1]] = list()
						labeled[pair[1]].append(key)
				if (np.max(pair[0]) >= threshold[key] and flag == 'true'):
							flag = 'false'
							if(pair[1] not in labeled.keys()):
									labeled[pair[1]] = list()
							labeled[pair[1]].append(key)
				
		for key in labeled.keys():
				if(train_titles[key] != 'NoTitle'):
					sentence_with_titles = np.concatenate([get_averaged_embedding(x_train[key]),np.array(biobert.sentence_vector(train_titles[key]))])
				else:
					sentence_with_titles = np.concatenate([get_averaged_embedding(x_train[key]),np.zeros(768)])
				new_x_train.append(sentence_with_titles)
				new_y_train.append(labeled[key])
				ind.append(key)
		
		
		if type(batch) == str:
			pd.DataFrame(ind).to_csv('indices_' + batch.split('_')[0][-4:] + '_mode_' + str(input3) + '_' + input2 + '.csv')
		return new_x_train, new_y_train
										
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
	titles=read_labels(r'D:\Google Drive\AMULET\COMPLEX CHANGES\top100 embeddings\zero_shot_test_set_top_100_labels_titles.txt')
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
	titles=np.delete(titles,instances)
	print("x_test set size: "+ str(len(sentence_embeddings)))
	print("y_test set size: "+str(len(y_true)))

	return sentence_embeddings,y_true,titles

def evaluate_jointly(x,y,x_titles,y_titles, test_embeddings, classifiers, y_true_array, brand_new_indices, complex_new_indices, input2, input_w, input3, titles,transform,batch=''):
	x_test_titles,titles_indices,no_titles_indices = get_test_set_with_title(test_embeddings, titles)
	x_test_abstracts= get_aveaged_test_set(test_embeddings)
	pickle.dump(y_true_array, open('y_true_arrays.pkl', 'wb'))
	
	####### abstract classifier #################
	mlb=MultiLabelBinarizer()
	mlb.fit(y)
	pickle.dump(mlb, open('Binarizer.pkl', 'wb'))
	y_train=mlb.transform(y)
	classifier= OneVsRestClassifier(classifiers[0])
	classifier.fit(x,y_train)
	pickle.dump(classifier, open('LR_MODEL_FITTED.pkl', 'wb'))
	
	###### title classifier ###################
	
	mlb2=MultiLabelBinarizer()
	mlb2.fit(y_titles)
	pickle.dump(mlb, open('Binarizer_titles.pkl', 'wb'))
	y_train_titles=mlb.transform(y_titles)
	classifier2= OneVsRestClassifier(classifiers[0])
	classifier2.fit(x_titles,y_train_titles)
	pickle.dump(classifier2, open('LR_MODEL_FITTED_titles.pkl', 'wb'))
	
	y_test_prob_abstracts = classifier.predict_proba(np.array(x_test_abstracts))
	y_test_prob_titles = classifier.predict_proba(np.array(x_test_titles))
	
	combined_predictions = list()
	combined_probs= list()
	for i in range(0,len(y_test_prob_abstracts)):
		final_array = np.sum([y_test_prob_abstracts[i], y_test_prob_titles[i]], axis=0)
		combined_probs.append(final_array)
		predicted_label_pos = final_array.argsort()[-1]
		current_predictions = np.zeros(62)
		current_predictions[predicted_label_pos] = 1
		combined_predictions.append(current_predictions)
		
	pickle.dump(combined_probs, open('combined_probabilities.pkl', 'wb'))
	y_true = mlb.transform(y_true_array)
	
	brand_new_pred=list()
	brand_new_true=list()
	for index in brand_new_indices:
		brand_new_pred.append(combined_predictions[index])
		brand_new_true.append(y_true[index])
	
	complex_new_pred=list()
	complex_new_true=list()
	for index in complex_new_indices:
		complex_new_pred.append(combined_predictions[index])
		complex_new_true.append(y_true[index])
	
	df=pd.DataFrame([[str(np.round(f1_score(y_true,combined_predictions,average= 'macro'),3)),str(np.round(f1_score(brand_new_true, brand_new_pred,average= 'macro'),3)),str(np.round(f1_score(complex_new_true, complex_new_pred,average= 'macro'),3))]],columns=['F1_MACRO','F1_MACRO_BRAND_NEW','F1_MACRO_COMPLEX_NEW'])
		
	df.reset_index().to_csv("weakMeSH_Extension_titles_abstracts_combination_results.csv",index=False)	
	
	

def preditct_and_evaluate_model_1st_transformation(x, y, classi, mlb_method, x_test, y_true_array, brand_new_indices, complex_new_indices,title_indices,no_title_indices):
	
	
	pickle.dump(y_true_array, open('y_true_arrays.pkl', 'wb'))
	mlb=MultiLabelBinarizer()
	mlb.fit(y)
	pickle.dump(mlb, open('Binarizer.pkl', 'wb'))
	y_train=mlb.transform(y)
	
	if(mlb_method==1):
		classifier= OneVsRestClassifier(classi)    
	else:
		classifier = ClassifierChain(classi)    
	classifier.fit(x,y_train)
	pickle.dump(classifier, open('LR_MODEL_FITTED.pkl', 'wb'))
	
	y_pred = classifier.predict(np.array(x_test))
	pickle.dump(y_pred, open('y_pred.pkl', 'wb'))
	y_test = mlb.transform(y_true_array)
	
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
		
	title_pred=list()
	title_true=list()
	for index in title_indices:
		title_pred.append(y_pred[index])
		title_true.append(y_test[index])
	
	no_title_pred=list()
	no_title_true=list()
	for index in no_title_indices:
		no_title_pred.append(y_pred[index])
		no_title_true.append(y_test[index])
	
	pickle.dump(skm.multilabel_confusion_matrix(y_pred, y_test), open('Confusion_matrix_LR_900K_WeakMeSH_Extension.pickle','wb'))
	return f1_score(y_pred,y_test,average='macro'),f1_score(y_pred,y_test,average='micro'),f1_score(brand_new_pred,brand_new_true,average='macro'),f1_score(complex_new_pred,complex_new_true,average='macro'),f1_score(title_pred,title_true,average='macro'),f1_score(no_title_pred,no_title_true,average='macro'),skm.multilabel_confusion_matrix(y_pred, y_test)



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
	
	return brand_new_indexes, complex_new_indexes
	
def evaluate_mode_1(x, y, test_embeddings, classifiers, y_true_array, brand_new_indices, complex_new_indices, input2, input_w, input3, titles,transform,batch=''):
	   
		if(transform == 1 or transform == 2 or transform == 3 or transform == 4):
			x_test,titles_indices,no_titles_indices = get_aveaged_test_set_extended_with_titles(test_embeddings,titles,transform)
		
		elif(transform == 'n'):
			x_test,titles_indices,no_titles_indices = get_test_set_with_title(test_embeddings, titles)
	
		dfs=list()
		mlb_methods=[1]
		
		for classifier in classifiers:
			print(str(classifier))
			
			for mlb_method in mlb_methods:
				result_list=list()
				
				f1_mac,f1_mic,f1_brand,f1_complex,f1_titles,f1_no_titles,conf_matr=preditct_and_evaluate_model_1st_transformation(x, y, classifier, mlb_method, x_test, y_true_array, brand_new_indices, complex_new_indices,titles_indices,no_titles_indices)
				
				result_list.append([str(classifier), str(mlb_method), str(np.round(f1_mac,3)), str(np.round(f1_mic,3)), str(np.round(f1_brand,3)), str(np.round(f1_complex,3)),str(np.round(f1_titles,3)),str(np.round(f1_no_titles,3)),str(conf_matr)])
				print(result_list)
				
				df=pd.DataFrame(result_list,columns=['Classifier','MLB_METHOD','F1_MACRO','F1_MICRO','F1_MACRO_BRAND_NEW','F1_MACRO_COMPLEX_NEW',"F1_Titles","F1_No_Titles","Confusion_Matrix"])
				dfs.append(df)
	
		final_df= [df.set_index(['Classifier','MLB_METHOD','F1_MACRO','F1_MICRO','F1_MACRO_BRAND_NEW','F1_MACRO_COMPLEX_NEW',"F1_Titles","F1_No_Titles","Confusion_Matrix"]) for df in dfs]
		if batch == '':
			pd.concat(final_df,axis=1).reset_index().to_csv("RESULTS_PH_similarity_thr_"+str(input2).replace(".", "")+"_top_"+str(input1)+"_labels_weak_mesh_extensions_" + input3 + "_" + input_w + "00k instances.csv",index=False)
		else:
			pd.concat(final_df,axis=1).reset_index().to_csv("RESULTS_PH_similarity_thr_"+str(input2).replace(".", "")+"_top_"+str(input1)+"_labels_transformation_" + input3 + "_" + input_w + "00k instances_batch_no_multi_instance_similarity" + batch + ".csv",index=False)
		 
	
		return 
#%%
############################################## MAIN CODE ##########################################################
input1 = input("Choose Number of labels for the experiments: Top 10 (Press 10) or Top 100 (Press 100) or 62 labels with Previous Host (Press 62) ")
user = 'nikos'


if user == 'other':
	main_path = r'D:\Google Drive\AMULET'
	os.chdir(main_path)
	label_embeddings= read_pickles(main_path+'\\'+'label_embeddings') 
	thresholds = read_pickles(main_path+'\\'+'gmms_threshold_full')
	title_thresholds = read_pickles(main_path+'\\'+'gmms_threshold_full_titles')
	
	if(input1 == '10'):
		top_labels = read_labels(main_path+'\\'+'top_10_labels.txt')
	elif(input1 == '100'):
		top_labels =  read_labels(main_path+'\\'+'top_100_labels.txt')
	else:
		top_labels = read_labels(main_path+'\\'+'62_previous_host_labels.txt')
		
	brand_new=read_labels(main_path + '\\' + 'completely_new_labels.txt')
	complex_new=read_labels(main_path + '\\' + 'new_labels_from_complex_changes.txt')
		
else:
	main_path = r'D:\Google Drive\AMULET'
	os.chdir(main_path)
	label_embeddings= read_pickles(r'C:\Users\room5\PycharmProjects\use_PH_dataset\label_embeddings')
	thresholds = read_pickles(r'D:\Google Drive\AMULET\gmms_threshold_full')
	title_thresholds = read_pickles(main_path+'\\'+'gmms_threshold_full_titles')
	
	if(input1 == '10'):
		top_labels = read_labels(r'D:\Google Drive\AMULET\top_10_labels.txt')
	elif(input1 == '100'):
		top_labels =  read_labels(r'D:\Google Drive\AMULET\top_100_labels.txt')
	else:
		top_labels = read_labels(r'D:\Google Drive\AMULET\62_previous_host_labels.txt')
	
	brand_new=read_labels(r'D:\Google Drive\AMULET\completely_new_labels.txt')
	complex_new=read_labels(r'D:\Google Drive\AMULET\new_labels_from_complex_changes.txt')

with timer():
		print("Creating the test set for selected labels please wait...")
		if user == 'other':
			test_embeddings,y_true,test_titles = create_test_set(main_path+'\\'+'top100 embeddings\\',top_labels,main_path+'\\'+'pure_zero_shot_test_set_top100.txt')
		else:
			test_embeddings,y_true,test_titles = create_test_set(r'C:\Users\room5\PycharmProjects\use_PH_dataset\top100 embeddings/',top_labels,r"C:\Users\room5\PycharmProjects\Self_train_and_biozslmax\pure_zero_shot_test_set_top100.txt")

####### compose test variables
y_true_array = list()
for i in range(0,len(y_true)):
	y_true_array.append(y_true[i].split("#"))
	
brand_new_indices, complex_new_indices = find_label_categories(y_true_array, brand_new, complex_new)
#######  

print('##########')
choice = input("Choose the process that you want to execute: \nRead raw data (Press 0) \nSample existing dataset (Press 1) \n...")

if choice == '0':
	
	print('##########')
	print('-----> Some information about our label space')
	print('Number of \n brand new instances: %d \n complex new instances: %d' %(len(brand_new), len(complex_new)) )   
	print('##########')
	input2 = input("Choose Weakly Labeling Threshold [0,1] or press any other key to use the thresholds from the Gaussian Mixture Model Approach ")
	if(is_number(input2) and 0 <= float(input2) <=1):
		for key in thresholds.keys():
			thresholds[key] = float(input2)
		print("Chosen threshold is: "+ str(input2))
	else:
		input2 = 'GMMs_full'
		print("Chosen threshold is: GMMs")
	
	print('##########')
	input3= input("Choose to run: No Title Weak Labeling (Press 1) or MultiWeakMeSH (Press 2) or Title Only Weak Labeling  (Press 3) or  Title AND Abstract Weak Labeling (Press 4) ")
	print('##########')
	input_w = input("Choose how many old data will be used to look for weak labels 100k (Press 1) 200k (Press 2) 300k (Press 3) 400k (Press 4) 500k (Press 5) 600k (Press 6) 700k (Press 7) 800k (Press 8) 900k (Press 9) ")
	print('##########')
	print()
	print("Creating the weakly-labeled train set for the selected parameters please wait...")
		
	with timer():
		print('Data collection transformation...')
		#x,y,instances_per_batch = read_train_set_pickles(top_labels, label_embeddings,thresholds,int(input3), int(input_w))
		x,y,x_titles,y_titles,instances_per_batch=read_train_set_pickles_from_similarities(top_labels,thresholds,title_thresholds ,int(input3), int(input_w), input2, user)
			 
	print('##########')
	#input4 = input("Write the created train_set into .pickles files?: Yes (Press 1)/No (Press2) ")
	input4 = '1'
	
	os.chdir(main_path)
	if(input4 == '1'):
		with open('x_weakly_top_'+str(input1)+'_'+str(input2).replace(".", "")+"_"+ str(input3)+'.pickle', 'wb') as f:
			pickle.dump(x, f)
		f.close()
		with open('y_weakly_top_'+str(input1)+'_'+str(input2).replace(".", "")+"_"+ str(input3)+'.pickle', 'wb') as f:
			pickle.dump(y, f)
		f.close()
		
	pd.DataFrame(instances_per_batch).to_csv('x_weakly_top_'+str(input1)+'_'+str(input2).replace(".", "")+"_"+ str(input3)+"_mapping_for_" +  input_w + "00k instances.csv")


			
	with open('dataset_sizes_x_weakly_top_'+str(input1)+'_'+str(input2).replace(".", "") +'_'+ str(input3) + '_' + input_w + '00k instances.txt', "w") as out:
		out.write("Train size: "+str(len(x))+" Test Size: "+str(len(test_embeddings)))
	out.close()    
	print('##########')


#%%
os.chdir(main_path)
#classifiers=[BernoulliNB(),LinearDiscriminantAnalysis(),SGDClassifier(loss='modified_huber', random_state = 24, n_jobs=3),LogisticRegression(C=100,max_iter=100,random_state = 24, n_jobs=3),GaussianNB(),AdaBoostClassifier(),RandomForestClassifier(n_jobs=-1)]
classifiers = [LogisticRegression(C=100,max_iter=100,random_state = 24, n_jobs=3)]

#####

if choice == '0':
	
	with timer():
		print('Evaluation time...')
		#################################### First transformation Experiments ###########################################
		if(input3 == '1' or input3 == '2' or input3 == '3'):
	
			evaluate_mode_1(x, y, test_embeddings, classifiers, y_true_array, brand_new_indices, complex_new_indices, input2, input_w, input3,test_titles,int(input3))
	
		elif(input3 == '4'):
			evaluate_jointly(x,y,x_titles,y_titles,test_embeddings,classifiers, y_true_array, brand_new_indices, complex_new_indices, input2, input_w, input3,test_titles,int(input3))
		

		
