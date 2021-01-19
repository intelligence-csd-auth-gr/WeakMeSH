# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:30:27 2020

@author: stam
"""
import pandas as pd

def read_labels(f):
    file = open(f)
    top_labels = list()
    for line in file:
        top_labels.append(line[:-1])

    return top_labels



x = pd.read_csv(r'C:\Users\room5\PycharmProjects\create_ezsl_train_Set\freqs_pure_zerotest_set.csv')

complex_new=read_labels(r'D:\Google Drive\AMULET\complex_not_in_top_100.txt')

print(len(complex_new))        
s = 0
for i in complex_new:
    if i in x.iloc[:,0].to_list():
       s += x[x.iloc[:,0] == i].iloc[:,1].values[0]
print(s)