import os
import numpy as np
import cv2 as cv

import torch
from torchvision.transforms import v2
from ultralytics import YOLO

from captum.attr import visualization as viz
from captum.attr import Occlusion

model = YOLO('model/yolov8n.pt')
model.to('cuda')

def model_wrapper(input):    
    tensor, device = model.model(input)
    
    return tensor

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
    
    explain(input, img_num)


def explain(img, num):     


    torch.cuda.empty_cache()

    attributions_occ = occlusion.attribute(img,                                        
                                    strides = (3, 7, 7),                                        
                                    sliding_window_shapes=(3, 15, 15),
                                    baselines=0)   
    
    print(attributions_occ.cpu().detach().numpy().shape)
    
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(),(1,2,0)),
                                    np.transpose(img.squeeze().cpu().detach().numpy(),(1,2,0)),
                                    ["original_image", "heat_map"],
                                    ["all", "positive"],
                                    show_colorbar=True,
                                    outlier_perc=2)            
    

for pic in os.listdir('pics/val2017'):
    temp = cv.imread('pics/val2017/' + pic)
    
    temp_pics.append(temp)


transform_img(0)


