# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
from scipy.sparse import lil_matrix
import joblib
import pandas as pd
import contextlib
import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score, recall_score, precision_score
import random

def prepare_X_Y(txt):
    
		items=[]
		with open(txt) as text:
				for line in text:
						items.append(line[2:-2])

		X,Y = [], []
		counter = 0
		for item in items:
				if(item.__contains__(" labels: #")):
						counter += 1
						X.append(item.split(" labels: #")[0])
						Y.append(item.split(" labels: #")[1])
		if counter != len(items):
				print('Check message inside prepare_X_Y: ', counter, ' ', items)

		return X,Y

def read_pickles(name):
    with open(name + ".pickle", "rb") as f:
        pickle_file = pickle.load(f)
    f.close()

    return pickle_file

def read_labels(f):
    file = open(f)
    top_labels = list()
    for line in file:
        top_labels.append(line[:-1])

    return top_labels

def search_y_text(label, y_test):
    y_test_01 = []
    for _ in y_test:
        if label in _:
            y_test_01.append([1])
        else:
            y_test_01.append([0])
    return np.array(y_test_01)


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


def mine_batches(name, label, xx, emb_labels_txt, remove_ind):
	
        
		increase = 0
		print('*** -> ', len(xx), len(emb_labels_txt), len(remove_ind))

		emb_labels_01 = search_y_text(label, emb_labels_txt)
		print(sum(emb_labels_01))
		increase += sum(emb_labels_01)
		emb_labels_01_list = np.array(emb_labels_01)#[mask]
	
		if  sum(emb_labels_01) == 0:
			print('**************Discard***************')
			return [], [], 0
		
		del emb_labels_01				
		yy = np.delete(emb_labels_01_list, remove_ind, axis = 0)
		yy = yy.astype('float64')
		
		d_info[label]['removed_indices'] = remove_ind
		del emb_labels_01_list
		
		mask1 = np.where(yy == 1)[0]
		xx_masked = []
		yy_masked = []
		for i in list(mask1):
			xx_masked.append(xx[i])
		
		q = []
		for i in range(0, len(xx)):
			if i not in mask1:
				q.append(i)
				
		mask2 = random.sample(q, len(mask1) * 1)
		for i in list(mask2):
			xx_masked.append(xx[i])
		
		yy_masked = np.concatenate((yy[mask1], yy[mask2]))       
		
		return xx_masked, yy_masked, increase
    
def train_wmil(yy_masked_all, xx_masked_all):
	
	
			model = SGDWeights()
			if len(yy_masked_all) < 100:
				minib_value = 10
			else:
				minib_value = 20
			
			b = []
			ada_value = 0.3
			mom_value = 0.3
			
			model.set_params(ada=ada_value, alpha = 0.01, momentum=mom_value, minib=minib_value)
			tot,_, epoch = model.fit(xx_masked_all, yy_masked_all)
			print(tot, epoch)
			
			a = model.predict(xx_masked_all)
			b = []
			for i in a:
				if i < 0.5:
					b.append(0)
				else:
					b.append(1)
			print(sum(b), min(a), max(a))
			
			if sum(b) == 0:
				mask = a > np.quantile(a, 0.975)
				bb = np.array(b)
				bb[mask] = 1
				b = bb
					  
					
			train_error = mean_absolute_error( np.array(b), yy_masked_all )
			print(train_error)
			f1_train_error = f1_score(yy_masked_all, np.array(b), average='macro')
			print(f1_train_error)
			
			joblib.dump(model, 'ada_030_a_001_m_030_minib_20_mode' + input_path + '_batch_' + '300k' + '_label_' + str(label) + '.pkl')
			
			return a,b, train_error, f1_train_error, tot, epoch, model
    
#%% define paths for our experiment
    
# path of github
os.chdir(r'..\git\wmil-sgd')
from wmil_sgd import SGDWeights


# path of needed data
main_path = r'..\WeakMeSH\other WSL approaches\WMIL'
os.chdir(main_path)

input_path = input("Choose Transformation mode: Proposed (Press 1) or Prime (Press 2) or Extended Prime (Press 3) \n...")

indices_path = main_path + '\\' + input_path

os.chdir(indices_path)
inds = os.listdir(os.getcwd())

masks = []

for i in inds:
    if '.ini' in i:
        continue
    else:
        ind = pd.read_csv(i)
        masks.append( list(set(ind.iloc[:,1])) )
        

path_data = '..' #Previous Host Train_Set for top 100 labels folder
arrays = '..' #WMIR input files folder
source = '..\WeakMeSH\important files'
os.chdir(source)
top_labels = read_labels('62_previous_host_labels.txt')

#%% we transform the input training dataset in the proper formal for being examined by WMIR - the needed file is given to our link, if you use it proceed to the next cell

os.chdir(path_data)
counter_folder = 0
cc = 0
increase = 0


d_info = {}
emb_labels_text_all, xx_all, remove_ind_all = [], [], []

	
with timer():

		print('****')
		emb_labels_text_all, xx_all, remove_ind_all = [], [], []

		
		for folder in list(np.array(os.listdir(os.getcwd())))[0:1]: #run for the first 100k
            
			print(folder)
			os.chdir(arrays)
			emb_labels_text_all, xx_all, remove_ind_all = [], [], []

				
			with timer():
				print('Preprocess')
				  
				name = 'PH_x_train_embeddings_top_100_labels_' + folder.split()[1] + '_numpy_vectors'
		
				emb_vectors = read_pickles(name)
				x_train_one_label = np.array(emb_vectors)[masks[counter_folder]]
					
				del emb_vectors, name
				
			os.chdir(source)
			top_labels = read_labels('62_previous_host_labels.txt')
		
			with timer():
                
				print('Transform train data')
				x_train_one_label_list = []
				for _ in x_train_one_label:
					_ = np.asmatrix(_)
					x_train_one_label_list.append(_)  
					
				name = 'y_weakly_top_62_GMMs_full_1' #the weak  labels obtained by the proposed method for the training dataset
				emb_labels_txt = read_pickles(name)[0:len(masks[counter_folder])]
				
				xx = []
				counter = 0
				remove_ind = []
				for i in x_train_one_label_list:
					if i.shape[1] != 768: #reject any sentence that is not properly transfored into the needed format
						print(i.shape)
						print(counter)
						remove_ind.append(counter)
					else:    
						xx.append(lil_matrix(i))
					counter +=1
					
				print('We have to remove ', len(remove_ind), 'instances')
	
			for i in emb_labels_txt:
				emb_labels_text_all.append(i)
			for i in xx:
				xx_all.append(i)
			
			for i in remove_ind:
				remove_ind_all.append(i + increase)
				
            # we write the corresponding dataset into pickle - we provide this file for reducing time delays
			with open('PH_x_train_embeddings_top_62_labels_' + folder.split()[1] + '_modified_sparse_matrix_mode1_GMMs' + '.pickle', 'wb') as f:
				 pickle.dump([xx_all, emb_labels_text_all, remove_ind_all], f)
			f.close()
			
			counter_folder += 1
			

#%% we load the file that we just created previously which is also provided by us
name = 'PH_x_train_embeddings_top_100_labels_' + '100k_200k' + '_modified_sparse_matrix_mode1'
with open(name + '.pickle', 'rb') as f:
				xx, emb_labels_txt, remove_ind = pickle.load(f)
f.close()
		
#%% implementation of training stage

del ind
d_info = {}


for label in top_labels:
	
		xx_masked_all, yy_masked_all = [], []
		os.chdir(path_data)
		space = list(np.array(os.listdir(os.getcwd())))[0:1] # the slice corresponds to the first 100k batch  
		
		with timer():
			print('****')
			print(label)
			print('****')
			d_info[label] = {}
			increase = 0
			
			
			for folder in space:
				print(folder)
				os.chdir(source)
				  
				xx_masked, yy_masked, inc = mine_batches(name, label, xx, emb_labels_txt, remove_ind)
				
				for i in xx_masked:
					xx_masked_all.append(i)
				for i in yy_masked:
					yy_masked_all.append(i)
				
				del xx_masked, yy_masked
				increase += inc
			
			print('End data collection: ', increase)
			d_info[label]['ones'] = increase
			
			if increase == 0:
				continue
			else:

				a,b, train_error, f1_train_error, tot, epoch, model = train_wmil(yy_masked_all, xx_masked_all)
		
				d_info[label]['max'] = max(a)
				d_info[label]['min'] = min(a)
				d_info[label]['sum'] = sum(b)
				d_info[label]['train_mae_error'] = np.round(train_error,4)
				d_info[label]['f1_train']    = np.round(f1_train_error,4)
				d_info[label]['time'] = tot
				d_info[label]['epoches'] = epoch
				d_info[label]['model'] = model
				
				print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
				del a,b, train_error, f1_train_error, tot, epoch, model, xx_masked_all, yy_masked_all, inc

 
del xx, emb_labels_txt, remove_ind

print('Saving training info...')
with open('train_information_62_labels_mode' + input_path + 'batch_' +'300k' + '.pickle', 'wb') as f:
	pickle.dump(d_info, f)
f.close()
	

#%% implementation of testing stage

print('Testing preprocessing info...')
with timer():
    
    with open('testset_top_62_labels_numpy_vectors.pickle', 'rb') as f:
           l = pickle.load(f)
    f.close()
    test_x, test_y = l[0], l[1]


    x_test_one_label_list = []
    for _ in test_x:
            _ = np.asmatrix(_)
            x_test_one_label_list.append(_)   
            
    x_test = []
    counter = 0
    remove_ind_test = []
    for i in x_test_one_label_list:
        if i.shape[1] != 768:
            print(i.shape)
            print(counter)
            remove_ind_test.append(counter)
        else:    
            x_test.append(lil_matrix(i))
        counter +=1
        
        
    del counter
    del test_x, x_test_one_label_list
    del l
#%%
counter_folder = 0
os.chdir(source)

# load the previously computed dictionary
with open('train_information_62_labels_mode' + input_path + 'batch_' + '100k' + '.pickle', 'rb') as f:
           d_info = pickle.load(f)
f.close()

print('Testing stage')
for label in top_labels:
        
        with timer():
            print('****')
            print(label)
            print('****')
            
            if 'model' not in d_info[label].keys():
                continue
            
            y_test_01 = search_y_text(label, test_y)
            print(sum(y_test_01))
            d_info[label]['ones test'] = sum(y_test_01)
            y_test_01 = y_test_01.astype('float64')
    
    
            y_test_01 = np.delete(y_test_01, remove_ind_test, axis = 0)
            y_test_01 = y_test_01.astype('float64')
    
            
            model = joblib.load('ada_030_a_001_m_030_minib_20_mode' + input_path + '_batch_' + '300k' + '_label_' + str(label) + '.pkl')
    
            a = model.predict(x_test)
            b = []
            for i in a:
                if i < 0.5:
                    b.append(0)
                else:
                    b.append(1)
            print(sum(b), min(a), max(a))
    
            
            
            test_error = mean_absolute_error( np.array(b), y_test_01)
            print(test_error)
            f1_test = f1_score(y_test_01, np.array(b), average='macro')
            print(f1_test)
            f1_test_0 = f1_score(y_test_01, np.array(b), labels = [0], average='macro')
            print(f1_test_0)
            f1_test_1 = f1_score(y_test_01, np.array(b), labels = [1], average='macro')
            print(f1_test_1)
            prec_test = recall_score(y_test_01, np.array(b), average='macro')
            print(prec_test)
            rec_test = precision_score(y_test_01, np.array(b), average='macro')
            print(rec_test)
            
            d_info[label]['removed_indices_test'] = remove_ind_test
            d_info[label]['sum test']             = sum(b)
            d_info[label]['test__mae_error']     = np.round(test_error,4)
            d_info[label]['f1_test']       = np.round(f1_test,4)
            d_info[label]['f1_test_0']      = np.round(f1_test_0,4)
            d_info[label]['f1_test_1']      = np.round(f1_test_1,4)
            d_info[label]['prec_test']      = np.round(prec_test,4)
            d_info[label]['rec_test']      = np.round(rec_test,4)
    
            
            d_info[label]['minimum_score']       = np.round(min(a),4)
            d_info[label]['maximum_score']       = np.round(max(a),4)
            #d_info[label]['decided ones']        = list(np.where(np.array(b) == 1)[0])
            
            del a, b
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

with open('test_information_62_labels_mode' + input_path + 'batch_' + '300k' + '.pickle', 'wb') as f:
    		  pickle.dump(d_info, f)
f.close()

#%% export a csv with information per label

dd = dict((k, d_info[k]) for k in d_info.keys())

for i in dd.keys():
    if 'model' in dd[i]:
        del dd[i]['removed_indices'], dd[i]['removed_indices_test'], dd[i]['model']


pd.DataFrame.from_dict(dd).to_csv('results_62_labels_mode' + input_path + 'batch_300k.csv')
	
#%% producing resultson test set
    
from sklearn.preprocessing import MultiLabelBinarizer
import copy 
counter_folder = 0
os.chdir(path_data)


y_true = []
for i in test_y:
    y_true.append(i.split('#'))
    
y_true_reduced = np.delete(y_true, remove_ind_test, axis = 0)


y_true_reduced_again = [ [] for _ in range(len(y_true_reduced)) ]
for i in range(0,len(y_true_reduced)):
    for j in y_true_reduced[i]:
        y_true_reduced_again[i].append(j)

y_true_reduced = y_true_reduced_again


os.chdir(source)
brand_new = read_labels(source + '\\' + 'completely_new_labels.txt')
complex_new = read_labels(source + '\\' + 'new_labels_from_complex_changes.txt')
brand_new_indices, complex_new_indices = find_label_categories(y_true, brand_new, complex_new)

brand_new_indices_new = []
complex_new_indices_new = []

for i in remove_ind_test:
    if i in brand_new_indices:
        brand_new_indices.remove(i)
    if i in complex_new_indices:
        complex_new_indices.remove(i)
        
        
for i in brand_new_indices:
    how = np.count_nonzero(i > np.array(remove_ind_test))
    brand_new_indices_new.append(i - how)
    
for i in complex_new_indices:
    how = np.count_nonzero(i > np.array(remove_ind_test))
    complex_new_indices_new.append(i - how)


#%%
os.chdir(path_data)
counter_folder = 0
results = []


if True:
	os.chdir(source)
	with open('test_information_62_labels_mode' + input_path + 'batch_' '100k' + '.pickle', 'rb') as f:
			  d_info = pickle.load(f)
	f.close()

	y_true_copy = copy.deepcopy(y_true_reduced)



	y_pred = [ [] for _ in range(len(y_true_reduced)) ]
	for label in d_info.keys():
				
			if 'decided ones' not in d_info[label].keys():
				continue
			else:
				if d_info[label]['decided ones'] == []:
					#print('Empty ', label)
					continue
				else:
					for j in d_info[label]['decided ones']:
						y_pred[j].append(label)


	mlb=MultiLabelBinarizer()
	mlb.fit(y_pred)
		
		
	y_pred = mlb.transform(y_pred)
	y_test = mlb.transform(y_true_copy)
	
	brand_new_pred=list()
	brand_new_true=list()
	for index in brand_new_indices_new:
		brand_new_pred.append(y_pred[index])
		brand_new_true.append(y_test[index])
	
	complex_new_pred=list()
	complex_new_true=list()
	for index in complex_new_indices_new:
		complex_new_pred.append(y_pred[index])
		complex_new_true.append(y_test[index])
	

	print(f1_score(y_pred,y_test,average='macro'),f1_score(y_pred,y_test,average='micro'),f1_score(brand_new_pred,brand_new_true,average='macro'),f1_score(complex_new_pred,complex_new_true,average='macro'))
	f1_macro = np.round(f1_score(y_pred,y_test,average='macro'),3) 
	f1_micro = np.round(f1_score(y_pred,y_test,average='micro'),3) 
	f1_macro_brand = np.round(f1_score(brand_new_pred,brand_new_true,average='macro'),3)
	f1_macro_complex = np.round(f1_score(complex_new_pred,complex_new_true,average='macro'),3) 
	
	results.append([f1_macro, f1_micro, f1_macro_brand, f1_macro_complex])


df = pd.DataFrame(results)
df.columns = ['F1_MACRO','F1_MICRO','F1_MACRO_BRAND_NEW','F1_MACRO_COMPLEX_NEW']
df.to_csv('WMIR_results_100k_dynamic_model_' + input_path + '.csv')

print('End of WMIR')
#%%