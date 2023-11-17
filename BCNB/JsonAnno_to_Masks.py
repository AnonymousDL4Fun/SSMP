#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 20:05:11 2022

@author: Anonymus
"""

from pathlib import Path
import json
from glob import glob
import math
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1e20
from tqdm import tqdm

"""
Following script converts the tumor annotations provided in JSON format
into binary image mask.

Output:
    Non tissue region = 0
    Tumor mask = 255
"""

scale_factor = 1
evaluation_json = {}

file_paths = glob("./WSIs/*.json")
wsi_dir = './WSIs/'
save_dir = './Masks/'

for n, file_path in enumerate(tqdm(file_paths[:])):

    file_name = Path(file_path).stem
    
    # Reading data from the xml file
    with open(f'{file_path}', 'r') as f:
        annos = json.load(f)
        
    # Read WSI to create dummy mask
    #----------------------------------
    f_name = glob(f"{wsi_dir}/{file_name}.jpg")[0]

    # Read WSI Image
    wsi = np.array(Image.open(f_name))
    height_img, width_img = wsi.shape[0], wsi.shape[1]  
    
    mask = np.zeros((height_img, width_img))
    #----------------------------------
    
    # Create folder to save masks
    Path(f"{save_dir}/{file_name}").mkdir(parents=True, exist_ok=True)
    
    data = annos['positive']
    
    for line in data:           
        x_roi, y_roi = [p[0] for p in line['vertices']], [p[1] for p in line['vertices']]
        
        contours = np.array((x_roi, y_roi)).T
        mask = cv2.fillPoly(mask, [contours], (255, 0, 0))
        
    
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.save(f"{save_dir}/{file_name}_Tumor_mask.png")       
