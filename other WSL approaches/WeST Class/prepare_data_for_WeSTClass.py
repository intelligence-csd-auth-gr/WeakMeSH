# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:18:35 2020

@author: room5
"""

import numpy as np

def read_txt(f):
    file = open(f)
    top_labels = list()
    for line in file:
        top_labels.append(line[:-1])

    return top_labels





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
	y_true.append(string[:-1])

x_true=np.delete(x_true,instances)
y_true=np.delete(y_true,instances)
#print(data[0][2:].split('labels: #')[0]
import csv
with open('dataset.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    for row in x_true:
        spamwriter.writerow([row])

count=0
for label in labels:
    f=open("classes.txt","a")
    print(str(count)+":"+label)
    if(label.__contains__('Whole')):
        f.write(str(count)+":"+label.split(" ")[1].split('-')[0].replace(",",""))
    elif(label.__contains__('Development')):
        f.write(str(count)+":"+label.split(" ")[0].split('-')[0].replace(",",""))
    elif(label.__contains__('Contraception')):
        f.write(str(count)+":"+'Contraception')
    elif(label.__contains__('Cigarette Smoking')):
        f.write(str(count)+":"+'Cigarette')
    elif(label.__contains__('Marijuana Use')):
        f.write(str(count)+":"+'Marijuana')
    elif(label.__contains__('Sexual Health')):
        f.write(str(count)+":"+'Sexual')
    elif(label.__contains__('Cancer Survivors')):
        f.write(str(count)+":"+'Cancer')
    elif(label.__contains__('Protein Antibodies')):
        f.write(str(count)+":"+'Antibodies')
    elif(label.__contains__('Standing Position')):
        f.write(str(count)+":"+'Position')
    elif(label.__contains__('Big Data')):
        f.write(str(count)+":"+'Metadata')
    elif(label.__contains__('Non')):
        f.write(str(count)+":"+'Nonsmokers')
    elif(label.__contains__('Water Pipe Smoking')):
        f.write(str(count)+":"+'Pipe')
    elif(label.__contains__('Biological Variation, Population')):
        f.write(str(count)+":"+'Population')
    elif(label.__contains__('Gait')):
        f.write(str(count)+":"+'Gait')
    elif(label.__contains__('Cells')):
        f.write(str(count)+":"+label.split(" ")[0].split('-')[0].replace(",",""))
    elif(label.__contains__('Stress')):
        f.write(str(count)+":"+label.split(" ")[0].split('-')[0].replace(",",""))
    elif(len(label.split(" ")) == 2):
        f.write(str(count)+":"+label.split(" ")[1].split('-')[0].replace(",",""))
    else:
        f.write(str(count)+":"+label.split(" ")[0].split('-')[0].replace(",",""))
    f.write('\n')
    count+=1
f.close()

setaki=set()
classes= read_txt("classes.txt")
for ses in classes:
    if ses.split(":")[1] not in setaki:
        setaki.add(ses.split(":")[1])
    else:
        print(ses)

print(len(setaki))


count=0
for label in labels:
    f=open("classes_for_eval.txt","a")
    print(str(count)+":"+label)
    f.write(str(count)+":"+label)
    f.write('\n')
    count+=1
f.close()
