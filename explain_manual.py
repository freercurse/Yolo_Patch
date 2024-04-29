import logging

import numpy as np
import cv2 as cv

import torchvision.transforms as transforms
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.WARNING)

model = YOLO('model/yolov8n.pt')
model.to('cuda')



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

def preprocess_image(image):
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  
    return tensor

def start_explanation(image):    
    stride = 8
    process = [8, 10, 16, 24 , 32, 48]

    explanations = []

    for i, sizes in enumerate(process):
        print("Starting Explaination ", i ," Process")
        ex = explain(image, sizes, stride)
        explanations.append(ex)
        print("Completed Explaination ", i ," Process")
    
    combined = np.min(np.stack(explanations), axis=0)

    normalized_saliency_map = (combined - combined.min()) / (combined.max() - combined.min())
    normalized_saliency_map = (normalized_saliency_map * 255).astype(np.uint8)

    cv.imshow("Saliency Map", combined)
    cv.waitKey(0)
    cv.destroyAllWindows()




def explain(image, window_size, stride):   
    height, width, _ = image.shape
    
    scores = np.zeros((height, width))
    
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            
            occluded_image = occlude_image(image, y, x, window_size)
            
            occluded_tensor = preprocess_image(occluded_image)
            
            with torch.no_grad():
                results = model(occluded_tensor, classes=[11])                
                for preds in results:
                    for box in preds.boxes:                        
                        conf = box.conf.cpu().numpy()
                        scores[y:y + window_size, x:x + window_size] = conf

    
    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    
    saliency_map = (normalized_scores * 255).astype(np.uint8)

    return saliency_map
    

def occlude_image(image, y, x, window_size):
   
    occluded_image = image.copy()    
    occluded_image[y:y + window_size, x:x + window_size] = 0

    return occluded_image

temp = cv.imread('pics/val2017/000000000724.jpg')
start_explanation(temp)
