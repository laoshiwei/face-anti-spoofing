# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:29:52 2021

@author: Admin
"""
import os, random
import shutil
import csv
import pandas as pd
#paths
test_path =  "C:/Users/Admin/Desktop/face-spoofing/replayattack_extracted/test_all_fin"
train_path ="C:/Users/Admin/Desktop/face-spoofing/replayattack_extracted/train_all_fin"
val_path = "C:/Users/Admin/Desktop/face-spoofing/replayattack_extracted/devel_all_fin"
csv_path = "C:/Users/Admin/Desktop/full_csv_no_duplicates.csv"
#reads the general csv that includes train test and validation
general = list(csv.reader(open(csv_path)))
#changes into dataframe to make it faster
df = pd.DataFrame(general, columns=['Filepath','Label'])

#for test, train and validation, does the same things, with a couple small changes
#so only test explained

#reads all the photo adresses into a list for further use
#does this by finding all the png extension files in the specific file
test_jpgs = []
for file in os.listdir(test_path):
    if file.endswith(".png"):
        test_jpgs.append(file)
        
#then uses pandas in a for loop the pinpoint all the corresponding lines to image names
#inside the general csv and
#adds them to a list since iteratively adding to list is faster than adding to df
test = []
for testi in test_jpgs:
    test_name = "images/" + testi
    test_row = df.loc[df['Filepath'] == test_name]
    liste = test_row.values.tolist()
    #print(liste)
    new = [liste[0][0], liste[0][1]]
    print(new)
    test.append(new)
    
#finally the list is changed to dataframe and saved
df_test = pd.DataFrame(test)
df_test.to_csv("C:/Users/Admin/Desktop/images_csv/test.csv", index=False, header=False)   



train_jpgs = []
for file2 in os.listdir(train_path):
    if file2.endswith(".png"):
        train_jpgs.append(file2)

train = []
for traini in train_jpgs:
    train_name = "images/" + traini
    train_row = df.loc[df['Filepath'] == train_name]
    liste = train_row.values.tolist()
    #print(liste)
    new = [liste[0][0], liste[0][1]]
    print(new)
    train.append(new)
    #print(train)
df_tr = pd.DataFrame(train)
df_tr.to_csv("C:/Users/Admin/Desktop/images_csv/training.csv", index=False, header=False)  

          
 
val_jpgs = []
for file2 in os.listdir(val_path):
    if file2.endswith(".png"):
        val_jpgs.append(file2)
val = []
for vali in val_jpgs:
    val_name = "images/" + vali
    val_df = df.loc[df['Filepath'] == val_name]
    liste = val_df.values.tolist()
    new = [liste[0][0], liste[0][1]]
    print(new)
    val.append(new)
val_df = pd.DataFrame(val)
val_df.to_csv("C:/Users/Admin/Desktop/images_csv/devel.csv", index=False, header=False)   
