# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:01:15 2021

@author: Admin
"""

import torch
import csv
import numpy as np
from torchvision import datasets, transforms
import cv2
import models.CDCNs as CDCNs
import os
#paths for evaluation sources
test_target_csv = "C:/Users/Admin/OneDrive/face_anti_spoofing/data/nuaa/test.csv"
test_images_path = "C:/Users/Admin/OneDrive/face_anti_spoofing/data/nuaa/test_all_fin"

#this section is for loading the model
model = CDCNs.CDCNpp()
model_main = torch.load()
model.load_state_dict(model_main['state_dict'])
#make it compatible with cuda, adn in eval mode for evaluation
# Send the model to the GPU 
model.cuda()
# Set layers such as dropout and batchnorm in evaluation mode
model.eval()

def pred_im(test_image):
    path = os.path.join(test_images_path, os.path.basename(test_image))
    image = cv2.imread(path)
    print("image opened")
    #transform the face and transpose for it to be usable in the from_numpy
    data_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    image = image.transpose(2,1,0)
    image = torch.from_numpy(image).float().unsqueeze(0).cuda(0)
    #after the image is in the proper form, it is loaded to the CDCNN model for output
    outputs = model(image)
    #torch.no_grad() is used to prevent errors from clashes 
    #and only output's [0] tuple part used out of the 6 since that 
    #part is the depth map which is the wanted output.
    with torch.no_grad():
            #only the x and y axis are used since depth is only 1 
            #by taking the mean of the depth map, the result of genuineness is found
            score = torch.mean(outputs[0], axis=(1,2))
            print(score)
            
    #if the resulting score is bigger than 0.6 it is genuine otherwise it is fake       
    if score >= 0.6:
        return 1
    else:
        return 0
  
def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives
#create the target tensor by reading from the csv directly
target_list = list(csv.reader(open(test_target_csv)))
target_tensor = torch.tensor([int(target_list[i][1]) for i in range(len(target_list))])
print(target_tensor)

pred_labels = []
#for every element in csv, search the photos to find it
#when found, use pred_img function to decide if it is a 0 or 1 and append to list
for im in target_list:
    print("im name", im)
    for cur_im in os.listdir(test_images_path):
        cur_im = "images/" + cur_im
        #if names same, do prediction and add to tensor
        if cur_im == im[0]:
            print("add")
            label_0or1 = pred_im(im[0])
            print(label_0or1)
    try:       
        pred_labels.append(label_0or1)  
    except:
        continue

pred_tensor = torch.tensor(pred_labels)
print(pred_tensor)
#assign the outputs to proper variables     
TP, FP, TN, FN = confusion(pred_tensor, target_tensor)
#calculate the evaluation metrics using the formulas
APCER = FP / (TN + FP)
BPCER = FN/(FN + TP)
HTER = (FP/(TN + FP) + FN/(FN + TP)) * 0.5
#print results
print("APCER: ", APCER)
print("BPCER: ", BPCER)
print("HTER: ", HTER)










