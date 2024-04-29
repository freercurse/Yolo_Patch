import os
import numpy as np
import cv2 as cv

import torch
from torchvision.transforms import v2
from ultralytics import YOLO
import matplotlib.pyplot as plt

from captum.attr import visualization as viz
from captum.attr import Occlusion

model = YOLO('model/yolov8n.pt')
model.to('cuda')

def model_wrapper(input):    
    results, _ = model.model(input)
    
    return torch.argmax(results)

occlusion = Occlusion(model_wrapper)

temp_pics = []
input_pics = []

transform_normalize = v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

def transform_img(img_num):     

    img = cv.resize(temp_pics[img_num],(128,128))
    
    
    tranposed = np.transpose(img,(2,0,1))   

    input = torch.tensor(tranposed, device=('cuda')).float()    

    normalised = transform_normalize(input)         

    input = normalised.unsqueeze(0)      
    
    explain(input, tranposed)


def explain(img, tranposed): 

    torch.cuda.empty_cache()

    normalised = np.zeros_like(tranposed)

    attributions_occ = occlusion.attribute(img,                                                               
                                    strides = (3, 6, 6),                                                                       
                                    sliding_window_shapes=(3, 13, 13),
                                    baselines=0)  
     
    img_mat = attributions_occ.squeeze().cpu().detach().numpy()   
    

    max = np.max(img_mat)
    min = np.min(img_mat)

    for i in range(img_mat.shape[0]):
        for j in range(img_mat.shape[1]):
            normalised[i][j] = img_mat[i][j] - min * (255/(max-min))
        
    
    img_trans = np.transpose(normalised, (1,2,0))
    img_orig = np.transpose(tranposed, (1,2,0))

    show_img(img_orig, img_trans)
    
    
def show_img(img1, img2):
   
    fig = plt.figure(figsize=(10, 7)) 

    fig.add_subplot(1,2,1)

    plt.imshow(img1)    
    fig.add_subplot(1,2,2)
    plt.imshow(img2)

    plt.show()  
    

for pic in os.listdir('pics/val2017'):
    temp = cv.imread('pics/val2017/' + pic, cv.COLOR_BAYER_BGGR2RGB)
    
    temp_pics.append(temp)

for i in range(len(temp_pics)):
    transform_img(i)

    cv.waitKey(0)



