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
import torch
import contextlib
import time
import random
import sys

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

def read_train_set_pickles_from_similarities(top_labels,threshold,transformation_method,data_range, input2, choice = 'other'):
		final_x_train = list()
		final_y_train = list()
		instances_per_batch= list()
		if choice == 'other':
				path = main_path+'\\'+'Previous Host Train_Set for top 100 labels'
		else:
				path = r'D:\Google Drive\AMULET\Previous Host Train_Set for top 100 labels'
		counter = 0
		for i in os.listdir(path)[0:data_range]:
				print('reading similarity batch ', i)
				counter+=1
				if 'from' in i:
						os.chdir(path + '\\' + i)
						print('I am into the folder: ', i)
						for j in os.listdir(os.getcwd()):
								if 'similarities' in j:
										similarities = read_pickles(j.split('.')[0])
								if 'embeddings' in j:
										x_emb = read_pickles(j.split('.')[0])
										
						if(transformation_method == 1):
								x,y = read_weakly_labeled_similarity_transformation_1(similarities, x_emb, threshold,top_labels, transformation_method, input2, batch=i)
						elif(transformation_method == 2):
								x,y = read_weakly_labeled_similarity_transformation_2(similarities, x_emb, threshold,top_labels)
						else:
								x,y = read_weakly_labeled_similarity_transformation_3(similarities, x_emb, threshold,top_labels)
						instances_per_batch.append(len(x))
						final_x_train+=x
						final_y_train+=y
						del x,y,x_emb
		
		print("x_train size: "+str(len(final_x_train)))
		print("y_train size: "+str(len(final_y_train)))
		return final_x_train,final_y_train,instances_per_batch
					 

	

def read_weakly_labeled_similarity_transformation_1(similarities,x_train,threshold,top_labels,input3, input2, batch=None):
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
				new_x_train.append(get_averaged_embedding(x_train[key]))
				new_y_train.append(labeled[key])
				ind.append(key)
		
		print("x_train length from current batch:"+ str(len(new_x_train)))
		print("y_train length from current batch:"+ str(len(new_y_train)))
		if type(batch) == str:
			pd.DataFrame(ind).to_csv('indices_' + batch.split('_')[0][-4:] + '_mode_' + str(input3) + '_' + input2 + '.csv')
		return new_x_train, new_y_train


def read_weakly_labeled_similarity_transformation_2(similarities,x_train,threshold,top_labels):
		new_x_train = list()
		new_y_train = list()
		labeled=dict()
		for key in top_labels:
				for pair in similarities[key]:
						for i in range(0,len(pair[0])):
								if(pair[0][i] >= threshold[key]):
										if(pair[1] not in labeled.keys()):
												labeled[pair[1]] = list()
										labeled[pair[1]].append([key,i])
								
		for key in labeled.keys():
				for pair in labeled[key]:
						if(len(x_train[key][pair[1]].cpu())!=0 and arrayisin(np.array(x_train[key][pair[1]].cpu()), new_x_train) == False):
								new_x_train.append(np.array(x_train[key][pair[1]].cpu()))
								new_y_train.append(pair[0])
		
		print('x_train length from current batch: '+ str(len(new_x_train)))
		print('y_train length from current batch: '+str(len(new_y_train)))
		return new_x_train,new_y_train
				

def read_weakly_labeled_similarity_transformation_3(similarities,x_train,threshold,top_labels):
		new_x_train = list()
		new_y_train = list()
		labeled=dict()
		found_max=list()
		for key in top_labels:
				for pair in similarities[key]:
						for i in range(0,len(pair[0])):
								if(pair[0][i] >= threshold[key]):
										if(pair[1] not in labeled.keys()):
												labeled[pair[1]] = list()
										labeled[pair[1]].append([key,i])
								
		for key in labeled.keys():
				for pair in labeled[key]:
						if(len(x_train[key][pair[1]].cpu())!=0 and arrayisin(np.array(x_train[key][pair[1]].cpu()), found_max) == False):
								temp=x_train[key].copy()
								del temp[pair[1]]
								if(len(temp) != 0):
										averaged_part=get_averaged_embedding(temp)
										final_instance = np.concatenate([np.array(x_train[key][pair[1]].cpu()),averaged_part])
										found_max.append(np.array(x_train[key][pair[1]].cpu()))
										new_x_train.append(final_instance)
										new_y_train.append(pair[0])
					
				
					
					
		
		print('x_train length from current batch: '+ str(len(new_x_train)))
		print('y_train length from current batch: '+str(len(new_y_train)))
		return new_x_train,new_y_train
		
		
										
		
def read_train_set_pickles(top_labels, label_emb, threshold, transformation_method, data_range, choice='other'):
	
	instances_per_batch = []
	final_x_train=list()
	final_y_train=list()
	
	
	if choice == 'other':
			path = main_path+'\\'+'Previous Host Train_Set for top 100 labels'
	else:
			path = r'D:\Google Drive\AMULET\Previous Host Train_Set for top 100 labels'

	counter = 0
	for i in os.listdir(path)[0:data_range]:
		print('reading embedding batch ', i)
		counter += 1
		if 'from ' in i:
			os.chdir(path + '\\' + i)  # i am into the folder e.g. from 100k_200k
			print('I am into the folder: ', i)
			for j in os.listdir(os.getcwd()): # i am searching the files into the current folder
				if 'embeddings' in j:
					x_emb = read_pickles(j.split('.')[0])
				if 'new_y' in j:
					y_cand = read_pickles(j.split('.')[0])

			if(transformation_method == 1):
				x,y,similarities_dict=get_weakly_labeled_set_similarity_1st_transformation(x_emb, y_cand, label_emb, top_labels,threshold, input3, i)
			elif(transformation_method == 2):
				x,y=get_weakly_labeled_set_similarity_2nd_transformation(x_emb, y_cand, label_emb, top_labels,threshold, input3, i)
			elif(transformation_method == 3):
				x,y= get_weakly_labeled_set_similarity_3rd_transformation(x_emb, y_cand, label_emb, top_labels,threshold, input3, i)
			else:
				x,y=get_weakly_labeled_set_setn(x_emb, label_emb, top_labels, threshold, input3, i)
			final_x_train+=x
			final_y_train+=y
			instances_per_batch.append(len(x))
			with open('similarities_'+str(i)+'_.pickle', 'wb') as f:
				pickle.dump(similarities_dict,f)
				f.close()
								

			del x,y,x_emb,y_cand
	
	print("Train size: "+str(len(final_x_train)))
	print("Train size: "+str(len(final_y_train)))
	
	return final_x_train,final_y_train, instances_per_batch
	


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

def get_weakly_labeled_set_setn(x_train,label_embeddings,top_labels,threshold, batch=None):
	new_x_train=list()
	new_y_train=list()
	ind=list()
	for i in range(0,len(x_train)):
		labels=list()
		for label in top_labels:
				max_sim=0
				for sentence in x_train[i]:
					if(len(sentence) != 0):
						sim_label_sentence=1-cosine(label_embeddings[label],np.array(sentence.cpu()))
						if (sim_label_sentence >= max_sim):
							max_sim=sim_label_sentence
				if(max_sim >= threshold[label]): 
					labels.append(label)
					
					
		if(len(labels)!=0 and arrayisin(np.array(get_averaged_embedding(x_train[i])),new_x_train)==False):
			new_x_train.append(get_averaged_embedding(x_train[i]))
			new_y_train.append(labels)
			ind.append(i)

	print("x_train size from current batch: "+str(len(new_x_train)))
	print("y_train size from current batch: "+str(len(new_y_train)))
	if type(batch) == str:
		pd.DataFrame(ind).to_csv('indices_' + batch.split('_')[0][-4:] + '_mode_' + str(input3) + '.csv')
	return new_x_train,new_y_train

def get_weakly_labeled_set_similarity_1st_transformation(x_train,candidate_y,label_embeddngs,top_labels,threshold, input3, batch=None):
	new_x_train=list()
	new_y_train=list()
	ind = list()
	similarities_dict=dict()
	for i in range(0,len(x_train)):
		labels=list()
		for label in top_labels:
			if label not in similarities_dict.keys():
					similarities_dict[label] = list()
			if label in candidate_y[i]:
				small_list=list()
				max_sim=0
				for sentence in x_train[i]:
					if(len(sentence) != 0):
						sim_label_sentence=1-cosine(label_embeddings[label],np.array(sentence.cpu()))
						small_list.append(sim_label_sentence)
						if (sim_label_sentence >= max_sim):
							max_sim=sim_label_sentence
				if(max_sim >= threshold[label]): 
					labels.append(label)
				similarities_dict[label].append([small_list,i])
					
					
		if(len(labels)!=0 and arrayisin(np.array(get_averaged_embedding(x_train[i])),new_x_train)==False):
			new_x_train.append(get_averaged_embedding(x_train[i]))
			new_y_train.append(labels)
			ind.append(i)

	print("x_train size from current batch: "+str(len(new_x_train)))
	print("y_train size from current batch: "+str(len(new_y_train)))
	if type(batch) == str:
		pd.DataFrame(ind).to_csv('indices_' + batch.split('_')[0][-4:] + '_mode_' + str(input3) + '.csv')
	
	return new_x_train,new_y_train,similarities_dict

def get_weakly_labeled_set_similarity_2nd_transformation(x_train,candidate_y,label_embeddings,top_labels,threshold, input3, batch=None):
	new_x_train=list()
	new_y_train=list()
	ind=list()

	for i in range(0,len(x_train)):
		for label in top_labels:#label_embeddings.keys():
			if label in candidate_y[i]:
				#print(i)
				#print(label)
				max_sim=0
				#max_sentence=[]
				for sentence in x_train[i]:
					if(len(sentence) != 0):
						sim_label_sentence=1-cosine(label_embeddings[label],np.array(sentence.cpu()))
						if (sim_label_sentence >= max_sim and arrayisin(np.array(sentence.cpu()), new_x_train)==False):
							max_sim=sim_label_sentence
							max_sentence=np.array(sentence.cpu())
				if(max_sim >= threshold[label] and arrayisin(max_sentence, new_x_train)==False):
					#print(i)
					new_x_train.append(max_sentence)
					new_y_train.append(label)
					#print(label)
					ind.append(i)

	print("x_train size from current batch: "+str(len(new_x_train)))
	print("y_train size from current batch: "+str(len(new_y_train)))
	if type(batch) == str:
		pd.DataFrame(ind).to_csv('indices_' + batch.split('_')[0][-4:] + '_mode_' + str(input3) + '.csv')
	return new_x_train,new_y_train

def get_weakly_labeled_set_similarity_3rd_transformation(x_train,candidate_y,label_embeddings,top_labels,threshold, input3, batch=None):
	new_x_train=list()
	new_y_train=list()
	found_max=list()
	ind=list()

	for i in range(0,len(x_train)):
		for label in top_labels:
			if label in candidate_y[i]:
				max_sim=0
				for j in range(0,len(x_train[i])):
					if(len(x_train[i][j]) != 0):
						sim_label_sentence=1-cosine(label_embeddings[label],np.array(x_train[i][j].cpu()))
						if (sim_label_sentence >= max_sim and arrayisin(np.array(x_train[i][j].cpu()), found_max)==False):
							max_sim=sim_label_sentence
							max_sentence=np.array(x_train[i][j].cpu())
							max_index=j
				if(max_sim >= threshold[label] and arrayisin(max_sentence, found_max)==False):

					temp=x_train[i].copy()
					del temp[max_index]
					
					if(len(temp) != 0 ):
						averaged_part=get_averaged_embedding(temp)
						final_instance=np.concatenate([max_sentence,averaged_part])
						found_max.append(max_sentence)
						new_x_train.append(final_instance)
						new_y_train.append(label)
						ind.append(i)

	print("x_train size from current batch: "+str(len(new_x_train)))
	print("y_train size from current batch: "+str(len(new_y_train)))
	if type(batch) == str:
		pd.DataFrame(ind).to_csv('indices_' + batch.split('_')[0][-4:] + '_mode_' + str(input3) + '.csv')
	return new_x_train,new_y_train


def preditct_and_evaluate_model_1st_transformation(x, y, classi, mlb_method, x_test, y_true_array, brand_new_indices, complex_new_indices):
	
	mlb=MultiLabelBinarizer()
	mlb.fit(y)
	y_train=mlb.transform(y)
	
	if(mlb_method==1):
		classifier= OneVsRestClassifier(classi)    
	else:
		classifier = ClassifierChain(classi)    
	classifier.fit(x,y_train)
	
	y_pred = classifier.predict(np.array(x_test))
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
	

	return f1_score(y_pred,y_test,average='macro'),f1_score(y_pred,y_test,average='micro'),f1_score(brand_new_pred,brand_new_true,average='macro'),f1_score(complex_new_pred,complex_new_true,average='macro')

def get_prediction_propabilities_2nd_transformation(x,y,classifier,test_embeddings):
	propabilities=dict()
	count=0
	for instance in test_embeddings:    
		propabilities[count] = list()
		for sentence in instance:
			if (len(sentence) != 0):
				sentence_correct_format = [np.array(sentence.cpu())]
				true_propability = max(classifier.predict_proba(sentence_correct_format)[0])
				y_predict=classifier.predict(sentence_correct_format)[0]  
				propabilities[count].append([y_predict,true_propability])
		count+=1
	
	return propabilities
				 
	
def get_prediction_propabilities_3rd_transformation(x,y,classifier,test_embeddings):
	propabilities=dict()
	count=0
	for instance in test_embeddings:
		propabilities[count] = list()        
		for i in range(0,len(instance)):  
			if (len(instance[i]) != 0):
				 
				 temp=instance.copy()
				 del temp[i] # remove the current sentence 
				 
				 if(len(temp) != 0 ):
					 
					 averaged_part=get_averaged_embedding(temp)
					 final_instance=np.concatenate([np.array(instance[i].cpu()),averaged_part])
					 sentence_correct_format = [np.array(final_instance)]
					 
					 true_propability = max(classifier.predict_proba(sentence_correct_format)[0])
					 y_predict=classifier.predict(sentence_correct_format)[0]
					 propabilities[count].append([y_predict,true_propability])
		count+=1
	return propabilities
	

def predict_model_2nd_3rd_transformation(propabilities, threshold):
	predictions=dict()
	
	for key in propabilities:
		predictions[key]=list()
		for pair in propabilities[key]:
			true_propability=pair[1]
			y_predict=pair[0]
			if(true_propability > threshold):
					if(y_predict not in predictions[key]):
						predictions[key].append(y_predict)
		if(len(predictions[key]) == 0):
			predictions[key].append("None")
	
		 
				
	print(len(predictions))
	return predictions


def evaluate_2nd_and_3rd_transformations(predictions,y_true, brand_new_indices, complex_new_indices):
	mlb= MultiLabelBinarizer()

	y_predict=list()
	for key in predictions.keys():
		y_predict.append(predictions[key])

	y_true_final=list()
	for i in range(0,len(y_true)):
		y_true_final.append(y_true[i].split("#"))
	
	mlb.fit(y_predict)
	y_pred=mlb.transform(y_predict)
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


	return f1_score(y_pred,y_test,average='macro'),f1_score(y_pred,y_test,average='micro'),f1_score(brand_new_pred,brand_new_true,average='macro'),f1_score(complex_new_pred,complex_new_true,average='macro')


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
	
def evaluate_mode_1(x, y, test_embeddings, classifiers, y_true_array, brand_new_indices, complex_new_indices, input2, input_w, input3, batch=''):
	
		x_test = get_aveaged_test_set(test_embeddings)
	
		dfs=list()
		mlb_methods=[1]
		
		for classifier in classifiers[:-2]:
			print(str(classifier))
			
			for mlb_method in mlb_methods:
				result_list=list()
				
				f1_mac,f1_mic,f1_brand,f1_complex=preditct_and_evaluate_model_1st_transformation(x, y, classifier, mlb_method, x_test, y_true_array, brand_new_indices, complex_new_indices)
				
				result_list.append([str(classifier), str(mlb_method), str(np.round(f1_mac,3)), str(np.round(f1_mic,3)), str(np.round(f1_brand,3)), str(np.round(f1_complex,3))])
				print(result_list)
				
				df=pd.DataFrame(result_list,columns=['Classifier','MLB_METHOD','F1_MACRO','F1_MICRO','F1_MACRO_BRAND_NEW','F1_MACRO_COMPLEX_NEW'])
				dfs.append(df)
	
		final_df= [df.set_index(['Classifier','MLB_METHOD','F1_MACRO','F1_MICRO','F1_MACRO_BRAND_NEW','F1_MACRO_COMPLEX_NEW']) for df in dfs]
		if batch == '':
			pd.concat(final_df,axis=1).reset_index().to_csv("RESULTS_PH_similarity_thr_"+str(input2).replace(".", "")+"_top_"+str(input1)+"_labels_transformation_" + input3 + "_" + input_w + "00k instances.csv",index=False)
		else:
			pd.concat(final_df,axis=1).reset_index().to_csv("RESULTS_PH_similarity_thr_"+str(input2).replace(".", "")+"_top_"+str(input1)+"_labels_transformation_" + input3 + "_" + input_w + "00k instances_batch" + batch + ".csv",index=False)
		 
	
		return 
#%%
############################################## MAIN CODE ##########################################################
input1 = input("Choose Number of labels for the experiments: Top 10 (Press 10) or Top 100 (Press 100) or 62 labels with Previous Host (Press 62) ")
user = 'other'


if user == 'other':
	main_path = r'D:\Google Drive\AMULET'
	os.chdir(main_path)
	label_embeddings= read_pickles(main_path+'\\'+'label_embeddings') 
	thresholds = read_pickles(main_path+'\\'+'gmms_threshold_full')
	
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
			test_embeddings,y_true = create_test_set(main_path+'\\'+'top100 embeddings\\',top_labels,main_path+'\\'+'pure_zero_shot_test_set_top100.txt')
		else:
			test_embeddings,y_true = create_test_set(r'C:\Users\room5\PycharmProjects\use_PH_dataset\top100 embeddings/',top_labels,r"C:\Users\room5\PycharmProjects\Self_train_and_biozslmax\pure_zero_shot_test_set_top100.txt")

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
	input3= input("Choose to run: WeakMeSH (Press 1) or Prime (Press 2) or Extended Prime (Press 3) or  WeakMeSH without PI or PMN information (Press 4) or the ZSLBioSentMax method baseline (Press 5) ")
	print('##########')
	if(input3 != '5'):
		input_w = input("Choose how many old data will be used to look for weak labels 100k (Press 1) 200k (Press 2) 300k (Press 3) 400k (Press 4) 500k (Press 5) 600k (Press 6) 700k (Press 7) 800k (Press 8) 900k (Press 9) ")
		print('##########')
		print()
	elif(input3 == '5'):
		result_list=list()
		f1_mac = zero_shot_prediction_setn(test_embeddings, label_embeddings, top_labels, thresholds, y_true_array)
		result_list.append(['ZSL_SETN',str(len(top_labels)),input2,str(np.round(f1_mac,3))])
		df=pd.DataFrame(result_list,columns=['METHOD','TOP_LABELS','Threhsold','F1_MACRO'])
		df.reset_index().to_csv("Results_setn_method.csv",index=False)
		sys.exit()
	print("Creating the weakly-labeled train set for the selected parameters please wait...")
		
	with timer():
		print('Data collection transformation...')
		#x,y,instances_per_batch = read_train_set_pickles(top_labels, label_embeddings,thresholds,int(input3), int(input_w))
		x,y,instances_per_batch=read_train_set_pickles_from_similarities(top_labels,thresholds, int(input3), int(input_w), input2, user)
			 
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

elif choice == '1':
	
	if user == 'other':
		os.chdir(main_path+'\\'+'existing_pickles')
	else:
		pass
	
	print(os.listdir(os.getcwd()))
	select = input("Press the suffix of the file you need to load: ") # e.g. 075_4
	input2 = select.split('_')[0]
	input3 = select.split('_')[1]
	input_w = str(9)
	for _ in os.listdir(os.getcwd()):
		print(_)
		if select in _ and 'x' in _ and 'pickle' in _:
			x = read_pickles(_)
		elif select in _ and 'y' in _ and 'pickle' in _:
			y = read_pickles(_)
		elif select in _  and 'csv' in _:
			mapping = pd.read_csv(_)
	
	training_pairs = []
	
	protocol = input("Select your evaluation protocol: \n9 * 100k (Press 1) \n3 * 300k (Press 2) \n....") 
	
	if protocol == '1':
		
		for i in range(0,9):
			if i == 0:
				start = 0
				end = mapping.loc[0][1]
			else:
				start = mapping.loc[:(i-1)].sum()[1]
				end = mapping.loc[:(i-1)].sum()[1] + mapping.loc[i][1]
						
			print('Start: %d  End: %d' %(start, end))
			x_train = list(np.array(x)[start:end])
			y_train = list(np.array(y)[start:end])
			training_pairs.append([x_train, y_train])
			
	else:
		
		for i in range(0,9,3):
				if i == 0:
					start = 0
					end = mapping.loc[0:2].sum()[1]
				else:
					start = mapping.loc[:(i-1)].sum()[1]
					end = mapping.loc[:(i-1) + 3].sum()[1]
		
								
				print('Start: %d  End: %d' %(start, end))
				x_train = list(np.array(x)[start:end])
				y_train = list(np.array(y)[start:end])
				training_pairs.append([x_train, y_train])
	
else:
	exit()

#%%
os.chdir(main_path)
classifiers=[BernoulliNB(),LinearDiscriminantAnalysis(),SGDClassifier(loss='modified_huber', random_state = 24, n_jobs=3),LogisticRegression(C=100,max_iter=100,random_state = 24, n_jobs=3),GaussianNB(),AdaBoostClassifier(),RandomForestClassifier(n_jobs=-1)]


#####

if choice == '0':
	
	with timer():
		print('Evaluation time...')
		#################################### First transformation Experiments ###########################################
		if(input3 == '1' or input3 == '4'):
	
			evaluate_mode_1(x, y, test_embeddings, classifiers, y_true_array, brand_new_indices, complex_new_indices, input2, input_w, input3)
	
		######################################## Second and Third transformation Experiments #######################################
		elif(input3 == '2' or input3 == '3'):
			
			print(collections.Counter(y))
			with open('label_frequencies_x_weakly_top_'+str(input1)+'_'+str(input2).replace(".", "") +"_"+ str(input3) + '_' + input_w + '00k instances.txt', "w") as output:
				for key in collections.Counter(y).keys():
					output.write(str(key)+" frequency: "+str(collections.Counter(y)[key]))
					output.write("\n")
			output.close()
			 
			thresholds=[0.7,0.8,0.9,0.99]
			dfs=list()
		
			for classifier in classifiers[:-2]:
				print(str(classifier))
				
				classifier.fit(x,y)
				if(input3 == '2'):
					probs=get_prediction_propabilities_2nd_transformation(x, y, classifier, test_embeddings)
				if(input3 == '3'):
					probs=get_prediction_propabilities_3rd_transformation(x, y, classifier, test_embeddings)
				
				for threshold in thresholds:
					result_list = list()
					
					preds = predict_model_2nd_3rd_transformation(probs, threshold)
					
					f1_mac,f1_mic,f1_brand,f1_complex=evaluate_2nd_and_3rd_transformations(preds,y_true, brand_new_indices, complex_new_indices)
					
					result_list.append([str(classifier),str(threshold),str(np.round(f1_mac,3)),str(np.round(f1_mic,3)),str(np.round(f1_brand,3)),str(np.round(f1_complex,3))])
					print(result_list)
					
					df=pd.DataFrame(result_list,columns=['Classifier','Prediction_Threshold','F1_MACRO','F1_MICRO','F1_MACRO_BRAND_NEW','F1_MACRO_COMPLEX_NEW'])
					dfs.append(df)
				
			final_df= [df.set_index(['Classifier','Prediction_Threshold','F1_MACRO','F1_MICRO','F1_MACRO_BRAND_NEW','F1_MACRO_COMPLEX_NEW']) for df in dfs]
			pd.concat(final_df,axis=1).reset_index().to_csv("RESULTS_PH_similarity_thr_"+str(input2).replace(".", "")+"_top_"+str(input1)+"_labels_transformation_"+str(input3)+ '_' + input_w + "00k instances.csv",index=False)
		 
elif choice == '1':
	
	for i,j in enumerate(training_pairs):
		batch = str(i)
		with timer():
			print('Evaluation time...')
			#################################### First transformation Experiments ###########################################
			if(input3 == '1' or input3 == '4'):
	
				evaluate_mode_1(j[0], j[1], test_embeddings, classifiers, y_true_array, brand_new_indices, complex_new_indices, input2, input_w, input3,  batch)
				
			elif (input3 == '2' or input3 == '3'):
				
				thresholds=[0.7,0.8,0.9,0.99]
				dfs=list()
			
				for classifier in classifiers[:-2]:
					print(str(classifier))
					classifier.fit(j[0],j[1])
					if(input3 == '2'):
						 probs = get_prediction_propabilities_2nd_transformation(j[0], j[1], classifier, test_embeddings)
					if(input3 == '3'):
						 probs = get_prediction_propabilities_3rd_transformation(j[0], j[1], classifier, test_embeddings)
					for threshold in thresholds:
						result_list = list()
						preds = predict_model_2nd_3rd_transformation(probs, threshold)
						f1_mac,f1_mic,f1_brand,f1_complex=evaluate_2nd_and_3rd_transformations(preds,y_true, brand_new_indices, complex_new_indices)
						
						result_list.append([str(classifier),str(threshold),str(np.round(f1_mac,3)),str(np.round(f1_mic,3)),str(np.round(f1_brand,3)),str(np.round(f1_complex,3))])
						print(result_list)
						
						df=pd.DataFrame(result_list,columns=['Classifier','Prediction_Threshold','F1_MACRO','F1_MICRO','F1_MACRO_BRAND_NEW','F1_MACRO_COMPLEX_NEW'])
						dfs.append(df)
					
				final_df= [df.set_index(['Classifier','Prediction_Threshold','F1_MACRO','F1_MICRO','F1_MACRO_BRAND_NEW','F1_MACRO_COMPLEX_NEW']) for df in dfs]
				pd.concat(final_df,axis=1).reset_index().to_csv("RESULTS_PH_similarity_thr_"+str(input2).replace(".", "")+"_top_"+str(input1)+"_labels_transformation_"+str(input3)+ '_' + input_w + "00k instances_batch" + batch + ".csv", index=False)
				
									
			else:
					print('Error')

		
