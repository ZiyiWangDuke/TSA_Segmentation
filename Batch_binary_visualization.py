from __future__ import division, print_function
import SimpleITK as sitk
import numpy as np
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

for filename in filelist:
    # convert .png into .nii, which is the input format of the warpping class
    filename = filename[0:-6]
    img_png = misc.imread(dir_fixedimg_png+filename+".png")
    resultlabel = nib.load(dir_warplabel+filename+"_label.nii")
    resultlabel = np.fliplr(np.rot90(resultlabel.get_data(),-1))

    # round up and med filter
    resultlabel = np.multiply(np.round(np.divide(resultlabel,0.05)),0.05)
    resultlabel = sp.signal.medfilt(resultlabel,[7,7])

    resultlabel[resultlabel == 0.2] = 0.9
    resultlabel[resultlabel == 0.8] = 0.2
    resultlabel[resultlabel == 0.9] = 0.8

    resultlabel[resultlabel>0.8]= 1
    resultlabel[resultlabel<0.05]= 1
    # unique, counts = np.unique(resultlabel, return_counts=True)
    # pdb.set_trace()

    ax[0].imshow(img_png,aspect="auto",cmap="gray")
    ax[1].imshow(resultlabel,aspect="auto",cmap="Accent")
    ax[2].imshow(resultlabel,aspect="auto",cmap="Accent")
    ax[2].imshow(img_png, aspect="auto",cmap='gray', alpha=0.4)
    ax[1].set_title(filename,fontsize = 20)
    fig.savefig(dir_figure+filename+'_ori.png')
    # plt.show()
