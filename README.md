# TSA_Segmentation
Duke ECE590 - Segmentation group for Kaggle Passanger Safety Screen 

# Package Dependence
Python package 
 - SimpleElastix
 ---- https://simpleelastix.readthedocs.io/GettingStarted.html

# File Introduction
Batch_binary_warp
- Batch process
- 1. warp atlas image to target image and create the transform
- 2. use the transform to warp atlas label to get target label

Batch_binary_visualization
- Batch process
- covert the result label to better values for 3D propogation 
