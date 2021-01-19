# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 19:43:41 2020

@author: stam
"""
import os, pickle

def read_pickles(name):

    with open(name + ".pickle", "rb") as f:
        pickle_file = pickle.load(f)
    f.close()

    return pickle_file

def tensor_to_numpy(x):
    x_cpu = []
    for i in x:
        _ = []
        for j in i:
            _.append(j.cpu().numpy())
        x_cpu.append(_)
    return x_cpu

path_data = r'C:\Users\stam\Documents\complex_changes\Previous Host Train_Set for top 100 labels'

os.chdir(path_data)
batches = 9
counter = 0
froms = os.listdir(os.getcwd())
for i in os.listdir(os.getcwd()):
    if batches == 9:
        print(i)
        os.chdir(path_data + '\\' + i)
        name = 'PH_x_train_embeddings_top_100_labels_'
        #print(os.listdir(os.getcwd()))
        print((name + i.split()[1] + '.pickle') in os.listdir(os.getcwd()))
        query = name + i.split()[1]
        
        text = read_pickles(query)
        print(len(text))
        continue
    
    
        # vectors = tensor_to_numpy(text)
        # os.chdir(path_data)
        
        
        # with open('PH_x_train_embeddings_top_100_labels_' + i.split()[1] + '_numpy_vectors.pickle', 'wb') as handle:
        #     pickle.dump(vectors, handle)                
        # handle.close()
        
        # del vectors, text, query
    
    else:
        if counter % 3 == 0:
            print(i)
            vectors = []
            for j in range(counter, counter+3):
                os.chdir(path_data + '\\' + froms[j])
                name = 'PH_x_train_embeddings_top_100_labels_'
                #print(os.listdir(os.getcwd()))
                print((name + froms[j].split()[1] + '.pickle') in os.listdir(os.getcwd()))
                query = name + froms[j].split()[1]
                
                text = read_pickles(query)
                _ = tensor_to_numpy(text)
                for k in _:
                    vectors.append(_)
                del text, query
                os.chdir(path_data)

            os.chdir(path_data)
            
            with open('PH_x_train_embeddings_top_100_labels_' + froms[j].split()[1] + '_numpy_vectors_300k.pickle', 'wb') as handle:
                pickle.dump(vectors, handle)                
            handle.close()
    
    counter += 1
    #print(os.getcwd())

#%%
#os.chdir(r'C:\Users\stam\Documents\complex_changes\Previous Host Train_Set for top 100 labels\from 100k_200k')


# name = 'PH_x_train_embeddings_top_100_labels_100k_200k'
# text = read_pickles(name)
# vectors = tensor_to_numpy(text)


# with open('PH_x_train_embeddings_top_100_labels_100k_200k_numpy_vectors.pickle', 'wb') as handle:
#     pickle.dump(vectors, handle)                
# handle.close()

# name = 'PH_x_train_text_top_100_labels_100k_200k'
# x = read_pickles(name)

# name = 'PH_y_train_top_100_labels_100k_200k'
# y = read_pickles(name)
#
#%%

# test_embeddings_vector = tensor_to_numpy(test_embeddings)


# with open('testset_top_62_labels_numpy_vectors.pickle', 'wb') as handle:
#     pickle.dump([test_embeddings_vector, y_true], handle)                
# handle.close()
