import pickle
import os
import numpy as np
import contextlib
import time


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

def read_labels(f):
    file = open(f)
    top_labels = list()
    for line in file:
        top_labels.append(line[:-1])

    return top_labels

def read_data(path):
    x_train = list()
    y_train = list()
    for file in os.listdir(path):
        if(file.__contains__(".txt")):
            #print(file)
            file = open(path+file)
            for line in file:
                y_train.append(line[2:-2].split("labels: #")[1])
                x_train.append(line[2:-2].split("labels: #")[0])

    print(len(x_train))
    print(len(y_train))

    return x_train,y_train


def read_pickles(pickle_name):
    name = pickle_name
    with open(name + ".pickle", "rb") as f:
        label_info = pickle.load(f)
    f.close()

    return label_info


def get_info_for_relevant_labels(label_info,top_labels):
    relevant_dict=dict()
    for key in label_info.keys():
        if (key in top_labels):
            relevant_dict[key]=list()
            relevant_dict[key].append(label_info[key][4])
            relevant_dict[key].append(label_info[key][5])

    print("relevant labels found in dict: "+str(len(relevant_dict)))
    return relevant_dict



def find_PH_dataset(x_train,y_train,relevant_dict):
    ph_labels=dict()
    for key in relevant_dict.keys():
        if(relevant_dict[key][0]!=''):
            for label in relevant_dict[key][0].split("#"):
                if label[:-1].split("(")[0].lower() not in ph_labels.keys(): #[:-1] since each string seems to have an empty char at the end
                    #ph_labels.append(label[:-1].split("(")[0].lower())
                    ph_labels[label[:-1].split("(")[0].lower()]=list()
                ph_labels[label[:-1].split("(")[0].lower()].append(key)
        if (relevant_dict[key][1] != ['']):
            for label in relevant_dict[key][1]:
                if label.lower() not in ph_labels.keys():
                    ph_labels[label.lower()] = list()
                    #ph_labels.append(label.lower())
                if(key not in ph_labels[label.lower()]):
                    ph_labels[label.lower()].append(key)

    ph_x=list()
    ph_y=list()
    new_y=list()
    indices=list()
    for i in range(0,len(x_train)):
        for y in y_train[i].split("#"):
            if y.lower() in ph_labels.keys():
                if(x_train[i] not in ph_x):
                    ph_x.append(x_train[i])
                    ph_y.append(y_train[i])
                    new_y.append(ph_labels[y.lower()])
                    indices.append(i)
    print(len(ph_x))
    print(len(ph_y))
    print(len(new_y))
    print(len(indices))

    return ph_x,ph_y,indices,new_y


def get_embeddings_from_pickle(path,instances,x_train,y_train,new_y):
    starting_instance=0
    final_x_train=list()
    final_y_train=list()
    final_new_y=list()
    removed_instances=list()
    for file in os.listdir(path):
        if(file.__contains__(".pickle")):
            print(starting_instance)
            with open(path + file, "rb") as f:
                sentence_embeddings = pickle.load(f)
                for i in range(0,len(instances)):
                        if (instances[i] < 10000+starting_instance and instances[i] not in removed_instances):
                            final_x_train.append(sentence_embeddings[instances[i]-starting_instance])
                            final_y_train.append(y_train[i])
                            final_new_y.append(new_y[i])
                            removed_instances.append(instances[i])
                starting_instance+=len(sentence_embeddings)

    print("final x size:"+str(len(final_x_train)))
    print("final y size:"+str(len(final_y_train)))



    with open('dataset_size.txt', 'a') as the_file:
        the_file.write('Dataset size is: '+str(len(final_x_train))+" instances")
    the_file.close()

    with open('PH_x_train_text_top_100_labels_000k_100k.pickle', 'wb') as f:
        pickle.dump(x_train, f)
        f.close()

    with open('PH_x_train_embeddings_top_100_labels_000k_100k.pickle', 'wb') as f:
        pickle.dump(final_x_train, f)
        f.close()

    with open('PH_y_train_top_100_labels_000k_100k.pickle', 'wb') as f:
        pickle.dump(final_y_train, f)
        f.close()

    with open('PH_new_y_top_100_labels_000k_100k.pickle', 'wb') as f:
        pickle.dump(final_new_y, f)
        f.close()

    return final_x_train,final_y_train






######################################## Main Code ##################################################################
with timer():
    top_labels=read_labels("top_100_labels.txt")
    label_info= read_pickles('top_100_label_info')
    os.chdir(r'C:\Users\room5\PycharmProjects\create_ezsl_train_Set\mesh_2018_train_embs/0_100k')
    x,y= read_data(r'C:\Users\room5\PycharmProjects\create_ezsl_train_Set\mesh_2018_train_embs/0_100k/')
    relevant_dict=get_info_for_relevant_labels(label_info,top_labels)
    ph_x,ph_y,indices,new_y=find_PH_dataset(x,y,relevant_dict)
    final_x,final_y=get_embeddings_from_pickle(r'C:\Users\room5\PycharmProjects\create_ezsl_train_Set\mesh_2018_train_embs/0_100k/',indices,ph_x,ph_y,new_y)
