# SSMP: Simulated Segmentation with Masked Patches

## Schematic
![SSMP Schematic](./figures/SSMP_schematic.drawio.png)

## Algorithm
<img src="./figures/SSMP_Algorithm.png" width="600">

## Dataset and Experiment details

Please check python scrips in BCNB and Camelyon folders. Dataset are accessible through the following links

Camelyon 16 and 17: https://camelyon17.grand-challenge.org/

Breast Cancer Core-Needle Biopsy (BCNB) : https://bupt-ai-cz.github.io/BCNB/


Data splits for few-shot experiments are provided in a text file inside each dataset folder.

Create a simulated segmentation dataset using the following command

python create_ssmp_dataset.py 

## Visualization of Tumor localization performance on Camelyon 16
Model was trained on 5 WSIs from Camelyon 17

![Camelyon visualization](./figures/Camlyon_results.png)

## Visualization of Tumor localization performance on MSKCC 
Model was trained on 5 WSIs from Camelyon 17

![MSKCC visualization](./figures/MSKCC_visualization.png)

## Visualization of Tumor localization performance on Breast Cancer Needle Core Biopsies (BCNB) 
Model was trained on 15 WSIs

![BCNB visualization](./figures/BCNB_visualization.png)
