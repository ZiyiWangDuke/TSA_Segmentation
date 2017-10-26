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
import scipy as sp
from scipy import signal

def otsu_threshold(im):

    pixel_counts = [np.sum(im == i) for i in range(256)]

    s_max = (0,-10)
    ss = []
    for threshold in range(256):

        # update
        w_0 = sum(pixel_counts[:threshold])
        w_1 = sum(pixel_counts[threshold:])

        mu_0 = sum([i * pixel_counts[i] for i in range(0,threshold)]) / w_0 if w_0 > 0 else 0
        mu_1 = sum([i * pixel_counts[i] for i in range(threshold, 256)]) / w_1 if w_1 > 0 else 0

        # calculate
        s = w_0 * w_1 * (mu_0 - mu_1) ** 2
        ss.append(s)

        if s > s_max[1]:
            s_max = (threshold, s)

    return s_max[0]

dir_fixedimg_png = "./Batch_2D_imgs/" # directory of fixed image(.png), the one you wanna get labels for
dir_fixedimg_nii = "./Batch_2D_niis/" # directory of fixed image(.nii)
dir_warplabel = "./Batch_2D_warp_labels/" # directory of the labels generated
dir_warpimg = "./Batch_2D_warp_niis/" # directory of the warpped image, which is generated from atlas image to be as close as fixed image
dir_figure = "./Batch_Figure_binary/" # plot the figure of the comparison
dir_binary = "./Batch_2D_binary/"

ShortImage = "./Templatess/ShortMask.nii"
TallImage = "./Templatess/TallMask.nii"
ShortLabel = "./Templatess/ShortLabel.nii"
TallLabel = "./Templatess/TallLabel.nii"

# obtain file name list
filelist = os.listdir(os.getcwd()+"/Batch_2D_imgs/")
filelist = [x.split('.')[0] for x in filelist]
affine = np.diag([1, 2, 3, 1])

fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize = (24,12))
for filename in filelist:
    # convert .png into .nii, which is the input format of the warpping class
    img_png = misc.imread(dir_fixedimg_png+filename+".png")

    # Image binary conversion
    imgfilet = img_png
    imgfilet = sp.signal.medfilt(imgfilet,[5,5])
    thre = otsu_threshold(imgfilet)
    mask = imgfilet
    bord1 = 250
    bord2 = 600
    mask[0:80,:] = mask[0:80,:]>thre*0.8
    mask[80:bord1,:] = mask[80:bord1,:]>thre*0.38
    mask[bord1:bord2,:] = mask[bord1:bord2,:]>thre
    mask[bord2:660,:] = mask[bord2:660,:]>thre*0.8
    maskf = sp.signal.medfilt(mask,[15,15])
    maskf = maskf>0.5

    # maskf = sp.ndimage.binary_fill_holes(maskf, structure=np.ones((30,30)))
    maskf = sp.ndimage.binary_closing(maskf, structure=np.ones((10,10)))

    # calculate the height of this sunject
    height = np.sum(maskf,axis=1)
    height = height>1
    height = np.sum(height)
    # the atlas image has height 602, the fixed image has height 552
    flag = height<(602+552)/2

    # read in png file and save as nii
    array_img = nib.Nifti1Image(np.rot90(1.0*maskf,3), affine)
    nib.save(array_img,dir_fixedimg_nii+filename+".nii")

    # set up elastix filter for the warpping
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.ReadImage(dir_fixedimg_nii+filename+".nii")) # fixed image

    if flag: # for short person
        elastixImageFilter.SetMovingImage(sitk.ReadImage(ShortImage)) # atlas image
    else:  # for tall person
        elastixImageFilter.SetMovingImage(sitk.ReadImage(TallImage)) # atlas image

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
    if flag:
        transformixImageFilter.SetMovingImage(sitk.ReadImage(ShortLabel)) # fixed label
    else:
        transformixImageFilter.SetMovingImage(sitk.ReadImage(TallLabel)) # atlas label

    transformixImageFilter.Execute()
    sitk.WriteImage(transformixImageFilter.GetResultImage(),dir_warplabel+filename+"_label.nii")

    ##******************** view registered image (not important)
    resultlabel = nib.load(dir_warplabel+filename+"_label.nii")
    resultlabel = resultlabel.get_data()
    resultlabel[resultlabel>1] = 1
    resultlabel = np.multiply(1.0*resultlabel,np.rot90(1.0*maskf,3))
    resultlabel[resultlabel <= 0] = 1

    # resultlabel = sp.signal.medfilt(resultlabel,[5,5])
    resultlabel = np.multiply(np.round(np.divide(resultlabel,0.05)),0.05)
    resultlabel[resultlabel>0.98]= 1
    resultlabel[resultlabel<0.02]= 1

    resultlabel = np.rot90(resultlabel,1)
    # resultlabel = np.fliplr(resultlabel)
    # resultlabel = sp.signal.medfilt(resultlabel,[10,10])
    # rotate 180 and save it back
    array_img = nib.Nifti1Image(np.rot90(1.0*resultlabel,2), affine)
    nib.save(array_img,dir_warplabel+filename+"_label.nii")

    # visualize result labels
    ax[0].imshow(img_png,aspect="auto",cmap="gray")
    ax[1].imshow(resultlabel,aspect="auto",cmap="Accent")
    ax[2].imshow(resultlabel,aspect="auto",cmap="Accent")
    ax[2].imshow(img_png, aspect="auto",cmap='gray', alpha=0.4)
    ax[1].set_title(filename,fontsize = 20)
    fig.savefig(dir_figure+filename+'_ori.png')
    # plt.show()

    #plot comparison figure
    # if flag:
    #     atlaslabel = nib.load(ShortLabel)
    # else:
    #     atlaslabel = nib.load(TallLabel)
    # atlaslabel = atlaslabel.get_data()
    # resultImage = nib.load(dir_warpimg+filename+"_warp.nii")
    # resultImage = resultImage.get_data()
    # fixedImage = nib.load(dir_fixedimg_nii+filename+".nii")
    # fixedImage = fixedImage.get_data()
    #
    # fig, ax = plt.subplots(1,4, sharex=True, sharey=True,figsize=(12,5))
    # # plt.set_cmap(gray)
    # ax[0].imshow(np.rot90(fixedImage),aspect="auto", cmap='gray')
    # ax[0].set_title('fixedImage', fontsize=20)
    # ax[1].imshow(np.rot90(resultImage),aspect="auto", cmap='gray')
    # ax[1].set_title('resultImage', fontsize=20)
    # ax[2].imshow(np.rot90(atlaslabel),aspect="auto", cmap='Accent')
    # ax[2].set_title('atlaslabel', fontsize=20)
    # ax[3].imshow(np.rot90(resultlabel),aspect="auto", cmap='Accent')
    # ax[3].set_title('resultlabel', fontsize=20)
    # plt.show()
    # # fig.savefig(dir_figure+filename+'.png')
    # pdb.set_trace()
