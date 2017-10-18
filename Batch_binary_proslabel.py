from __future__ import division, print_function
import SimpleITK as sitk
import numpy as np
import numpy.ma as ma
import pdb
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import os
from scipy import misc
import scipy as sp
from scipy import signal

dir_fixedimg_png = "./Batch_2D_imgs/" # directory of fixed image(.png), the one you wanna get labels for
dir_fixedimg_nii = "./Batch_2D_niis/" # directory of fixed image(.nii)
dir_warplabel = "./Batch_2D_warp_labels/" # directory of the labels generated
dir_warpimg = "./Batch_2D_warp_niis/" # directory of the warpped image, which is generated from atlas image to be as close as fixed image
dir_figure = "./Batch_Figure_overlay/" # plot the figure of the comparison
dir_binary = "./Batch_2D_binary/"
dir_otsu = "./Batch_Otsu/"

# obtain file name list
filelist = os.listdir(os.getcwd()+"/Batch_2D_warp_labels/")
filelist = [x.split('.')[0] for x in filelist]
affine = np.diag([1, 2, 3, 1])
fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize = (24,12))

# extract values of all the labels
num_label = 16
labelValues = np.zeros((num_label+1)) #including 0 for un-colored area
# replace the original label values with segmentation index
segValues = np.multiply(np.array(range(1,num_label+1)),0.05)

for filename in filelist:
    # convert .png into .nii, which is the input format of the warpping class
    filename = filename[0:-6]
    # filename = "07d04f2ba71419b0d7228f2c50c14318"

    img_png = misc.imread(dir_fixedimg_png+filename+".png")
    bodymask = nib.load(dir_fixedimg_nii+filename+".nii")
    bodymask = np.rot90(bodymask.get_data(),1)
    resultlabel = nib.load(dir_warplabel+filename+"_label.nii")
    resultlabel = np.fliplr(np.rot90(resultlabel.get_data(),-1))

    # extract values of all the labels
    unique, counts = np.unique(resultlabel, return_counts=True)
    stats = np.stack((unique,counts)).T
    stats = stats[stats[:,1].argsort()[::-1]]
    labelValues[0:num_label] = stats[1:num_label+1,0]
    labelValues.sort()
    labelValues = labelValues[labelValues>0]

    #create mask for pixels to color, fill all pixels with 0 in colorMask
    #map the result label values into segmentation values, 0.05, 0.1, ...
    colorMask = np.zeros(resultlabel.shape) # pixels from the interpolated areas
    backGroundWarp = ~(resultlabel==1) # background from warpping
    backGroundMask = ~(resultlabel==1) # background from body intensity threshold
    backGroundMask[bodymask == 1] = True
    # pdb.set_trace()
    for index in range(0,num_label):
        colorMask[resultlabel == labelValues[index]] = 1
        resultlabel[resultlabel == labelValues[index]] = segValues[index]

    ##### use mask numpy array and bit shift to fill the empty pixels (pixels with interpolated values)
    ##### using nearest neigbor principle
    resultlabel = ma.masked_array(resultlabel,colorMask ==0)

    flag = True
    while np.any(flag): # break when it is all false
        for shift in (-1,1):
            for axis in (0,1):
                result_shifted = np.roll(resultlabel,shift=shift,axis=axis)
                idx = ~result_shifted.mask * resultlabel.mask
                resultlabel[idx] = result_shifted[idx]

                flag = resultlabel.mask * backGroundMask

    finalMask = np.logical_or(backGroundMask,backGroundWarp) #create a powerful mask
    resultlabel[finalMask == False] = 1
    ##########################cheating here####################################################3
    patch = resultlabel[1:100,:]
    patch[patch<0.7] = 1
    resultlabel[1:100,:] = patch

    ax[0].imshow(img_png,aspect="auto",cmap="gray")
    # ax[0].imshow(resultlabel,aspect="auto",cmap="gray")
    ax[1].imshow(resultlabel,aspect="auto",cmap="Accent")
    ax[2].imshow(resultlabel,aspect="auto",cmap="Accent")
    ax[2].imshow(img_png, aspect="auto",cmap='gray', alpha=0.4)
    # ax[2].imshow(bodymask,aspect="auto",cmap="gray")
    ax[1].set_title(filename,fontsize = 20)
    fig.savefig(dir_figure+filename+'.png')
    # plt.show()
