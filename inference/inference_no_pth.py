# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 12:49:17 2021

@author: Admin
"""

#from video photo, find face mtccn, draw square on it, then write the score on top
#torch inference 
from mtcnn_cv2 import MTCNN
import torch, json
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import cv2
import models.CDCNs as CDCNs
import matplotlib.pyplot as plt
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture("face.mp4")
model = CDCNs.CDCNpp()#(pretrained=True)
model.cuda()
model.eval()
while True:
    #read image from the video
    ret, img = cap.read()
    #create MTCNN detector to find faces
    detector = MTCNN()
    #change colour space to rgb for the mtcnn
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #finally the resulting faces properties are assigned to a variable 
    result = detector.detect_faces(image_rgb)
    #if there is a face present to the following:
    if len(result) > 0:
        #extract keypoint and bounding box information from the result variable
        keypoints = result[0]['keypoints']
        bounding_box = result[0]['box']
        #assign bounding box infromation to proper variables for ease of use                    
        x1 = bounding_box[0]
        y1 = bounding_box[1]
        x2 = bounding_box[0]+bounding_box[2]
        y2 = bounding_box[1] + bounding_box[3]    
        #draw a rectangle on the picture to show the faces place          
        cv2.rectangle(img, (x1, y1), (x2, y2),(50,255,0),2)
    else: 
        continue
    #crop the image to include only the face section
    image_cr = img[y1:y2,x1:x2]
    data_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    #image = cv2.imread(test_image)
    image = image_cr.transpose(2,1,0)
    # Now apply the transformation, expand the batch dimension, and send the image to the GPU
    image = torch.from_numpy(image).float().unsqueeze(0).cuda(0)
    #get the outputs
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
    #for writing geniune or fake at the bottom of screen
    cv2.putText(img, str(result), (x2 - 85, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(img, str(round(float(score),4)), (x2-85, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    #for showing the resulting photos and depth map
    cv2.imshow("mtcnn face",img)
    with torch.no_grad():
        depth_bin_img = outputs[0].cpu().numpy()
        new = depth_bin_img.transpose(2,1,0)
        cv2.imshow("depth map", new)
    #torch.cuda.empty_cache()
    if cv2.waitKey(1) &0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()

