# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 08:52:52 2021

@author: Admin
"""
import torch, json
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import cv2
import models.CDCNs as CDCNs
import matplotlib.pyplot as plt

test_image = "C:/Users/Admin/Desktop/dumped/CDCN-Face-Anti-Spoofing.pytorch-master/ \
data/nuaa/images/attack_highdef_client001_session01_highdef_photo_adverse_0.png"

# Prepare the labels
#with open("labels.json") as f:
#    labels = json.load(f)
# First prepare the transformations: resize the image to what the model was trained on 
#and convert it to a tensor
data_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
#read image
image = cv2.imread(test_image)
#print(image.shape)
#transpose for the correct shape to feed into the next steps
image = image.transpose(2,1,0)
#print(image.shape)
#plt.imshow(image), plt.xticks([]), plt.yticks([])
# Now apply the transformation, expand the batch dimension, and send the image to the GPU
#image = data_transform(image).unsqueeze(0).cuda()
image = torch.from_numpy(image).float().unsqueeze(0).cuda(0)
#print(type(image))

model = CDCNs.CDCNpp()#(pretrained=True)
# Send the model to the GPU 
model.cuda()
# Set layers such as dropout and batchnorm in evaluation mode
model.eval()
# Get the 1000-dimensional model output
outputs = model(image)
print(outputs[0].shape)

#only the x and y axis are used since depth is only 1 
#by taking the mean of the depth map, the result of genuineness is found       
with torch.no_grad():
        score = torch.mean(outputs[0], axis=(1,2))
        print(score)
#decide whether fake or genuine using score found   
if score >= 0.6:
    result = "genuine"
    print("genuine")
else:
    result = "fake"
    print("fake")