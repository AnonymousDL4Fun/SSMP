#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:48:06 2022

@author: aashayt
"""

from torchvision import transforms
from PIL import Image
import numpy as np
from glob import glob
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

Total_classes = 2
# Set the Image size to generate
size = 512


# Specify split [Training, Validation]
split = 'Validation'

# Specify the correct paths for Normal and Tumor patches. (Dummy dataset paths given)
Normal_dir = f'./dummy_dataset/Camelyon/{split}/Normal/'
Tumor_dir = f'./dummy_dataset/Camelyon/{split}/Tumor/'

# Specify the mask paths
Masks_dir = './dummy_dataset/DigestPath2019_sample_masks/'

# Specify folder locations to save the simulated RGB and Mask images
im_save_dir = f'./dummy_dataset/Camelyon/{split}/SSMP/Combined/'
mask_save_dir = f'./dummy_dataset/Camelyon/{split}/SSMP/Combined_labels/'

# Create save directories
Path(im_save_dir).mkdir(exist_ok=True, parents=True)
Path(mask_save_dir).mkdir(exist_ok=True, parents=True)

normal_paths = glob(f"{Normal_dir}/*.jpg")[:]
tumor_paths = glob(f"{Tumor_dir}/*.jpg")[:]

mask_paths =  glob(f"{Masks_dir}/*.png", recursive=True)

# Set max number of images to generate (Following values used in our experiments)
if split=='Training':
    max_images = 12000
if split=='Validation':
    max_images = 4000
    
counter=0
for idx in tqdm(range(0,40000)):
    
    # Select random image from normal and tumor sets
    im_0 = np.array(Image.open(normal_paths[np.random.randint(0,len(normal_paths))]))
    im_1 = np.array(Image.open(tumor_paths[np.random.randint(0,len(tumor_paths))]))
    
    # For benchmark dataset 1 (From digestpath2019)
    imgs = {0: im_0, 1: im_1}
    
    # Select random mask
    mask = Image.open(mask_paths[np.random.randint(0,len(mask_paths)-1)])
    
    # Apply standard transforms to increase variability
    mask = transforms.RandomHorizontalFlip()(mask)
    mask = transforms.RandomVerticalFlip()(mask)
    mask = transforms.RandomResizedCrop((size,size), scale=(0.5, 10),
                                        ratio=(0.5,1.5),
                                        interpolation=transforms.InterpolationMode.NEAREST)(mask)
    
    mask = np.array(mask)
    if len(np.unique(mask))>1:
        cl_1 = np.random.randint(0, Total_classes)
        cl_2 = np.random.randint(0, Total_classes)
        
        if cl_1 != cl_2: # Both selected classes should not be the same
        
            combined = np.zeros((size,size,3))
            combined[:,:,0] = (1-mask)*imgs[cl_1][:size,:size,0] + mask*imgs[cl_2][:size,:size,0]
            combined[:,:,1] = (1-mask)*imgs[cl_1][:size,:size,1] + mask*imgs[cl_2][:size,:size,1]
            combined[:,:,2] = (1-mask)*imgs[cl_1][:size,:size,2] + mask*imgs[cl_2][:size,:size,2]
            
            tmp_mask = mask.copy()
            tmp_mask[mask==0] = cl_1
            tmp_mask[mask==1] = cl_2
            mask = tmp_mask
            
            combined = Image.fromarray(combined.astype(np.uint8))
            mask = Image.fromarray(mask.astype(np.uint8))
            
            combined.save(f"{im_save_dir}/im_{counter:06d}.png")
            mask.save(f"{mask_save_dir}/im_{counter:06d}.png")
            
            counter += 1

    if counter==max_images:
        break