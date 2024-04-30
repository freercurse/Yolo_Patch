import logging
import os
import random

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.WARNING)

model = YOLO('model/yolov8n.pt')
model.to('cuda')

Patch_threshold = 36
images = []

#ATTACKED IMAGE INFERANCE

def infer_attacked(attacked_image, image):

    Compare_images = [attacked_image, image]
    
    results = model.predict(Compare_images, classes=range(0,79))    

    for i, preds in enumerate(results):
        print("Class Prection - Image ", i ,":")
        print(preds.boxes.cls)
        print("Class Confidence - Image", i ,":")
        print(preds.boxes.conf)

#PATCH PLACEMENT
def place_random(patch):
    random_file = random.randint(0, len(images)-1)
    image = images[random_file] 
    image_copy = np.copy(image)

    patch_pixels = []
    creation_counter = 0

    temp = np.zeros_like(image)       

    random_placement_y = random.randint(0 + Patch_threshold , image.shape[0] - Patch_threshold )
    random_placement_x = random.randint(0 + Patch_threshold , image.shape[1] - Patch_threshold )    

    cv.circle(temp, (random_placement_x, random_placement_y), Patch_threshold,(255,255,255), -1)     

    for x in range(0, patch.shape[0] -1):
        for y in range(0, patch.shape[1] -1):            
            if(patch[x][y][0] > 10):
                patch_pixels.append(patch[x][y])

    for x in range(0, temp.shape[0]-1):
        for y in range(0, temp.shape[1] -1):            
            if(temp[x][y][0] > 100):
                image[x][y] = patch_pixels[creation_counter]
                creation_counter += 1 

    return (image, image_copy)    

#PATCH GENERATION

def remove_border(map):    
    for x in range(0, map.shape[0] ):
        for y in range(0, map.shape[1] ):            
            if(map[x][y] <= 10):
                map[x][y] = 255

    return map

def start_patching(map, image): 
    unbordered = remove_border(map)
    _, _, min, _ =cv.minMaxLoc(unbordered)

    temp = np.zeros_like(image)
    patch = np.zeros_like(image)

    cv.circle(temp, min, Patch_threshold,(255,255,255), -1)    

    for x in range(0, patch.shape[0] -1):
        for y in range(0, patch.shape[1] -1):            
            if(temp[x][y][0] > 100):
                patch[x][y] = image[x][y]  

    return patch


# IMAGE PREPROCESS 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

def preprocess_image(image):
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  
    return tensor

# EXPLANATION  PROCESSING

def start_explanation(image):    
    stride = 8
    process = [8, 10, 16, 24 , 32, 48]

    explanations = []

    for i, sizes in enumerate(process):
        print("Starting Pass ", i ," - in Process")

        ex = explain(image, sizes, stride)
        explanations.append(ex)

        print("Pass ", i ," Completed")
    
    print("Completed Explanation")
    combined = np.min(np.stack(explanations), axis=0)

    normalized_saliency_map = (combined - combined.min()) / (combined.max() - combined.min())
    normalized_saliency_map = (normalized_saliency_map * 255).astype(np.uint8)

    return normalized_saliency_map

def explain(image, window_size, stride):   
    height, width, _ = image.shape
    
    scores = np.zeros((height, width))
    
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            
            occluded_image = occlude_image(image, y, x, window_size)
            
            occluded_tensor = preprocess_image(occluded_image)
            
            with torch.no_grad():
                results = model.predict(occluded_tensor, classes=[11])                
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

## IMAGE LOADING
folder = 'pics/val2017'

for filename in os.listdir(folder):
    img = cv.imread(os.path.join(folder,filename))
    if img is not None:
            images.append(img)

#PROGRAM CYCLING

#Start with an image to extract a patch from
temp = cv.imread('pics/val2017/000000000724.jpg')

#Perform an occlusion based explanation to extract a saliency map
map = start_explanation(temp)

#use the saliency to extract the patch a circular patch from the chosen image
patch = start_patching(map, temp)

#generate a random image and a random location for that patch 
attack, copy = place_random(patch)

#run the attacked image through the model to evaluate the effectiveness
eval = infer_attacked(attack, copy)

#SHOW IMAGE

fig = plt.figure(figsize=(10, 7)) 

fig.add_subplot(1,2,1)
plt.imshow(attack)    

fig.add_subplot(1,2,2)
plt.imshow(copy)

plt.show()  

    

