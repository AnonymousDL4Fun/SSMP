#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 4 20:16:37 2023

@author: Annonymus
"""

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1e20
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path


"""
Script to generate Combined Tissue + Tumor masks (single channel)

Output:
    Non tissue region = 0
    Tissue mask = 255
    Tumor mask = 120
"""

def img2label(img: np.ndarray, threshold: float = 20) -> np.ndarray:
    """
        Function to create binary mask for RGB image
        
        Returns:
            mask: numpy.ndarray - binary mask of input image
    """
    im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elm = im_hsv[:, :, 1]
    _, mask = cv2.threshold(elm, threshold, 255, cv2.THRESH_BINARY)
    return mask


# Specify WSI directory
image_paths = glob('./WSIs/*.jpg')[:]
# Specify 
tumor_paths = [f.replace('WSIs', 'Masks').replace('.jpg', '_Tumor_mask.png') for f in image_paths]

save_dir = './Tissue_Mask/'
Path(save_dir).mkdir(exist_ok=True, parent=True)

for n, p in enumerate(tqdm(image_paths[:])):
    f_name = p.split('/')[-1]
    tumor_m = np.array(Image.open(tumor_paths[n]).convert('L'))

    im = np.array(Image.open(p).resize((int(0.25*tumor_m.shape[0]), int(0.25*tumor_m.shape[1]))))
    
    
    tissue_mask = img2label(im, threshold=10)
    tissue_mask = np.array(Image.fromarray(tissue_mask).resize((int(tumor_m.shape[1]), int(tumor_m.shape[0])), Image.NEAREST))
    
    combined_mask = tissue_mask
    combined_mask[tumor_m==255] = 120
    
    combined_mask = Image.fromarray(combined_mask.astype(np.uint8))
    combined_mask.save(f'{save_dir}/{f_name}')