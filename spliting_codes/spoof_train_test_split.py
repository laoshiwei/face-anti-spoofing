# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:35:42 2021

@author: Admin
"""

import os, random
import shutil
import csv
import pandas as pd


#paths for the original and to be moved for train test and validation
tr_path = "C:/Users/Admin/Desktop/face-spoofing/replayattack_extracted/train_all"
train_path =  "C:/Users/Admin/Desktop/face-spoofing/replayattack_extracted/train_all_fin"
va_path = "C:/Users/Admin/Desktop/face-spoofing/replayattack_extracted/devel_all"    
val_path =  "C:/Users/Admin/Desktop/face-spoofing/replayattack_extracted/devel_all_fin"
te_path = "C:/Users/Admin/Desktop/face-spoofing/replayattack_extracted/test_all"
test_path =  "C:/Users/Admin/Desktop/face-spoofing/replayattack_extracted/test_all_fin"

#all the validation test and train are the same so only train explained
#it has a variable i to count the number of photos imported. 
#while the number of moved photos is smaller than the 70 percent of the total 
#number of photos, the loop continues. then iy chooses a random photo from the folder
#then moves it to the given path for the final train folder

i = 0
while i < int(len(os.listdir(tr_path))* 0.7): #this is for train split 70 percent
    tomove = random.choice(os.listdir(tr_path)) #name of the file not path
    shutil.move(os.path.join(tr_path,tomove), os.path.join(train_path, tomove))
    i+=1
 
i = 0
while i < int(len(os.listdir(te_path))* 0.1): #this is for test split 10 percent test
    tomove = random.choice(os.listdir(te_path))
    shutil.move(os.path.join(te_path,tomove), os.path.join(train_path, tomove))
    i+=1

i = 0
while i < int(len(os.listdir(va_path))* 0.2): #this is for validation split 20 percent
    tomove = random.choice(os.listdir(va_path))
    shutil.move(os.path.join(va_path,tomove), os.path.join(test_path, tomove))
    i+=1

    















#this part was used to merge two passes
for im in os.listdir(tr_path):
    shutil.move(os.path.join(tr_path,im), os.path.join(train_path, im))
    
#test set train_fin 
#for im in os.listdir(te_path):
#    shutil.move(os.path.join(te_path,im), os.path.join(train_path, im))
     
 

    


        

     
          
 
    