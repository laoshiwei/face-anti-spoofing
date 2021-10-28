# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:07:19 2021

@author: Admin
"""
import numpy as np
import torch
import cv2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#https://deeplizard.com/learn/video/0LhiS6yu2qQ
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        preds[0]
        
        all_preds = torch.cat((all_preds, preds),dim=0)
    return all_preds

#https://yeseullee0311.medium.com/pytorch-performance-evaluation-of-a-classification-model-confusion-matrix-fbec6f4e8d0
def getConfusionMatrix(model, show_image=False):
    model.eval() #set the model to evaluation mode
    confusion_matrix=np.zeros((2,2),dtype=int) #initialize a confusion matrix
    num_images=testset_sizes['test'] #size of the testset
    
    with torch.no_grad(): #disable back prop to test the model
        for i, (inputs, labels) in enumerate(testloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #get predictions of the model
            outputs = model(inputs) 
            _, preds = torch.max(outputs[0], 1) 
            
            #get confusion matrix
            for j in range(inputs.size()[0]): 
                if preds[j]==1 and labels[j]==1:
                    term='TP'
                    confusion_matrix[0][0]+=1
                elif preds[j]==1 and labels[j]==0:
                    term='FP'
                    confusion_matrix[1][0]+=1
                elif preds[j]==0 and labels[j]==1:
                    term='FN'
                    confusion_matrix[0][1]+=1
                elif preds[j]==0 and labels[j]==0:
                    term='TN'
                    confusion_matrix[1][1]+=1
                #show image and its class in confusion matrix    
                """if show_image:
                    print('predicted: {}'.format(class_names[preds[j]]))
                    print(term)
                    cv2.imshow(inputs.cpu().data[j])
                    print()"""
        #print results
        print('Confusion Matrix: ')
        print(confusion_matrix)
        print()
        
        print('Sensitivity: ', 100*confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1]))
        print('Specificity: ', 100*confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0]))
        print('PPV: ', 100*confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]))
        print('NPV: ', 100*confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1]))
        
        return confusion_matrix