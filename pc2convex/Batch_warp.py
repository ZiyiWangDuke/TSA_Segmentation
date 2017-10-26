## This is a batch processing script
# Ziyi Wang, Oct.05.2017

from __future__ import division, print_function
import SimpleITK as sitk
import numpy as np
import pdb
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import os
from scipy import misc

dir_fixedimg_png = "./Batch_2D_imgs/" # directory of fixed image(.png), the one you wanna get labels for
dir_fixedimg_nii = "./Batch_2D_niis/" # directory of fixed image(.nii)
dir_warplabel = "./Batch_2D_warp_labels/" # directory of the labels generated
dir_warpimg = "./Batch_2D_warp_niis/" # directory of the warpped image, which is generated from atlas image to be as close as fixed image
dir_figure = "./Batch_Figure/" # plot the figure of the comparison

# obtain file name list
filelist = os.listdir(os.getcwd()+"/Batch_2D_imgs/")
filelist = [x.split('.')[0] for x in filelist]
affine = np.diag([1, 2, 3, 1])

for filename in filelist:
    # convert .png into .nii, which is the input format of the warpping class
    img_png = misc.imread(dir_fixedimg_png+filename+".png")
    # read in png file and save as nii
    array_img = nib.Nifti1Image(np.rot90(img_png,3), affine)
    nib.save(array_img,dir_fixedimg_nii+filename+".nii")

    # set up elastix filter for the warpping
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.ReadImage(dir_fixedimg_nii+filename+".nii")) # fixed image
    elastixImageFilter.SetMovingImage(sitk.ReadImage("fixedImage.nii")) # atlas image

    # set up parameters for the warpping, we use affine first and then use bspline interpolation for non-rigid warpping
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.Execute()

    # save warpped image
    sitk.WriteImage(elastixImageFilter.GetResultImage(),dir_warpimg+filename+"_warp.nii")

    # apply the same transform to the label image
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    transformixImageFilter.SetMovingImage(sitk.ReadImage("fixedlabel.nii"))
    transformixImageFilter.Execute()
    sitk.WriteImage(transformixImageFilter.GetResultImage(),dir_warplabel+filename+"_label.nii")

    ##******************** view registered image (not important)
    resultlabel = nib.load(dir_warplabel+filename+"_label.nii")
    resultlabel = resultlabel.get_data()
    resultlabel[resultlabel <= 0] = 1
    resultlabel[resultlabel>1] = 1
    atlaslabel = nib.load("fixedlabel.nii")
    atlaslabel = atlaslabel.get_data()
    resultImage = nib.load(dir_warpimg+filename+"_warp.nii")
    resultImage = resultImage.get_data()
    fixedImage = nib.load(dir_fixedimg_nii+filename+".nii")
    fixedImage = fixedImage.get_data()

    fig, ax = plt.subplots(1,4, sharex=True, sharey=True,figsize=(12,5))
    # plt.set_cmap(gray)
    ax[0].imshow(np.rot90(fixedImage),aspect="auto", cmap='gray')
    ax[0].set_title('fixedImage', fontsize=20)
    ax[1].imshow(np.rot90(resultImage),aspect="auto", cmap='gray')
    ax[1].set_title('resultImage', fontsize=20)
    ax[2].imshow(np.rot90(atlaslabel),aspect="auto", cmap='Accent')
    ax[2].set_title('atlaslabel', fontsize=20)
    ax[3].imshow(np.rot90(resultlabel),aspect="auto", cmap='Accent')
    ax[3].set_title('resultlabel', fontsize=20)
    # plt.show(block=False)
    fig.savefig(dir_figure+filename+'.png')
    # pdb.set_trace()
