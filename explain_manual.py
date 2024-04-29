import os
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('model/yolov8n.pt')
model.to('cuda')

window_size = 16
stride = 8

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

def preprocess_image(image):
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  
    return tensor

def explain(image):   
    height, width, _ = image.shape
    
    scores = np.zeros((height, width))
    
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            
            occluded_image = occlude_image(image, y, x, window_size)
            
            occluded_tensor = preprocess_image(occluded_image)
            
            with torch.no_grad():
                results = model(occluded_tensor, classes=[67])                
                for preds in results:
                    for box in preds.boxes:                        
                        conf = box.conf.cpu().numpy()
                        scores[y:y + window_size, x:x + window_size] = conf

    
    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    
    saliency_map = (normalized_scores * 255).astype(np.float32)
    
    cv.imshow("Saliency Map", saliency_map)
    cv.waitKey(0)
    cv.destroyAllWindows()

def occlude_image(image, y, x, window_size):    
    occluded_image = image.copy()
    
    occluded_image[y:y + window_size, x:x + window_size] = 0

    return occluded_image

temp = cv.imread('pics/val2017/000000001296.jpg')
explain(temp)
