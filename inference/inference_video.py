# -*- coding: utf-8 -*-
"""
Created on Fri Sept 03 12:49:17 2021

@author: Admin
"""
#from video photo, find face mtccn, draw square on it, then write the score on top
from mtcnn_cv2 import MTCNN
import torch, json
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import cv2
import models.CDCNs as CDCNs
import matplotlib.pyplot as plt
from utils.utils import read_cfg, get_optimizer, get_device, build_network
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#path to the trained model
PATH = "C:/Users/Admin/OneDrive/face_anti_spoofing/experiments/output/CDCNpp_nuaa_360.pth"
cap = cv2.VideoCapture("face.mp4")

#this section is for loading the model
model = CDCNs.CDCNpp()
model_main = torch.load(PATH)
#then using load_state_dict the parameters are loaded, strict is added 
#to prevent the error because of deleting part of the model
model.load_state_dict(model_main['state_dict'], strict=False)
#cuda is not necessary and eval makes it worse so commented
#model.cuda()
#model.eval()

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
    
    #transform the face and transpose for it to be usable in the from_numpy
    data_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    image = image_cr.transpose(2,1,0)
    # Now apply the transformation, expand the batch dimension, and send the image to the GPU
    image = torch.from_numpy(image).float().unsqueeze(0)#.cuda(0)
    #after the image is in the proper form, it is loaded to the CDCNN model for output
    outputs = model(image)
    
    #torch.no_grad() is used to prevent errors from clashes 
    #and only output's [0] tuple part used out of the 6 since that 
    #part is the depth map which is the wanted output.
    with torch.no_grad():
            #only the x and y axis are used since depth is only 1 
            #by taking the mean of the depth map, the result of genuineness is found
            score = torch.mean(outputs[0], axis=(1,2))
            #for finding the pixel number
            #score = torch.sum(outputs[0] > 0.05) 
            print(score)
            
    #if the resulting score is bigger than 0.6 it is genuine otherwise it is fake     
    if score >= 0.2: # or 150 if going from pixel number
        result = "genuine"
        print("genuine")
    else:
        result = "fake"
        print("fake")
        
    #to print genuine or not at the bottom of the window
    cv2.putText(img, str(result), (x2 - 85, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(img, str(round(float(score),4)), (x2-85, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    #this parts are for showing the rseults, the image and the depth map
    cv2.imshow("mtcnn face",img)
    plt.imshow(img)
    #depth map part, changes output to image shape and after transpose, suitable
    #to be shown with cv2
    with torch.no_grad():
        depth_bin_img = outputs[0].cpu().numpy()
        new = depth_bin_img.transpose(2,1,0)
        cv2.imshow("depth map", new)
        
    if cv2.waitKey(1) &0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()










