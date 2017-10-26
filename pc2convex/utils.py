import numpy as np
from body_scan import BodyScan
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from time import  time
import random
import os
import config

def get_name(fpath):
    '''
    get file name without extension from fpath
    :param fpath:
    :return:
    A file name string
    '''
    _, fname = os.path.split(fpath)
    fname, _ = os.path.splitext(fname)
    return fname

def read_data(fpath):
    '''
    read a3d data from fpaht
    :param fpath: file path
    :return:
    An numpy n-dimensional array of 512x512x660
    '''
    tic = time()
    bs = BodyScan(fpath)
    data, _ = bs.read_img_data()
    toc = time()
    print('read {} costs {}s'.format(fpath, toc-tic))
    return data

def compress_data(data, n=2):
    '''
    compress numpy array by down sampling
    :param data: numpy ndarray
    :param n: compression rate
    :return: a numpy ndarray each dimension is 1/n of the original one
    '''
    return data[::n, ::n, ::n]

def ostu3d(data, min_th=0.1, aug=1.2):
    '''
    apply ostu filter to each vertial slice of a 3D tensor
    :param data: input 3D tensor
    :param min_th: the minimal threshold
    :param aug: strengthen the threshold to further reduce noise
    :return: a binary 3D tensor with the size of data
    '''
    out = np.zeros_like(data)
    th = []
    for i in range(data.shape[2]):
        thresh = threshold_otsu(data[:,:,i])
        out[:,:,i] = data[:,:,i] > max(thresh*aug, min_th)
        th.append(thresh)
    return out, th

def pc2img(data, threshold=0.1, axis=0):
    img = np.amax(data, axis)
    max_ = np.amax(img)
    min_ = np.amin(img)
    img = np.clip((img - min_) / (max_ - min_) - threshold, 0, None)
    img = img.transpose()
    img = np.flipud(img)
    return img

def get_points(image, thresh=0):
    indx = np.where(image > thresh)
    return np.vstack(indx).T.astype(np.float16)

def get_height(bata):
    '''
    get height of the figure
    :param bata: binary 3D tensor
    :return: the height in pixel, which is usually the position of hand
    '''
    pts = get_points(bata)
    return pts[np.argsort(pts[:,2])][-len(pts)//100, 2]

def seg_vis2d(pts, labels, data, savefig=True, fname=None):
    '''
    visualize the segmentation using 2d projection
    :param pts: coordinate of a point on human body
    :param labels: labels for segmentation
    :param data: 3D tensor to draw projections
    :return: 1 for success
    '''

    if len(pts) > 5000:
        indx = random.sample(range(len(pts)), 5000)
        pts = pts[indx]
        labels = np.array(labels)[indx]
    pt_front = np.array([[p[0], p[2], labels[i]] for i,p in enumerate(pts) if labels[i] not in (-1, 17)], dtype=np.int16)
    pt_side  = np.array([[p[1], p[2], labels[i]] for i,p in enumerate(pts) if (labels[i] not in (-1, 1, 2, 3, 4, 8, 9, 11, 13, 15))], dtype=np.int16)
    fig, axes= plt.subplots(ncols=2)
    for i, pt in enumerate(pt_front):
        axes[0].scatter(pt[0], pt[1], c=config.colors_plt[pt[2]], s=2)
    for i, pt in enumerate(pt_side):
        axes[1].scatter(pt[0], pt[1], c=config.colors_plt[pt[2]], s=2)
    for ax in axes:
        ax.set_xlim([0, data.shape[0]])
        ax.set_ylim([0, data.shape[2]])
    if savefig:
        assert fname
        plt.savefig('./outputs/' + fname + '.png')
    else:
        plt.show()

def lrf3d(tensor, threshold, nsampling):
    if tensor.size <= nsampling:
        return tensor, 1
    out = np.zeros_like(tensor)
    dense = np.sum(tensor) / tensor.size
    flag = 0
    if  dense > threshold:
        x, y, z = np.where(tensor > 0)
        index = random.sample(range(len(x)), int(nsampling / dense))
        for i in index:
            out[x[i], y[i], z[i]] = True
            flag = 1
    return out, flag

def sampling3d(data, sz=4, sd=4, threshold=0.2, nsampling=4):
    tic = time()
    out = np.zeros_like(data)
    flags = np.zeros([data.shape[0]//sd, data.shape[1]//sd, data.shape[2]//sd])
    for i in range((data.shape[0]-sz) // sd):
        for j in range((data.shape[1]-sz) // sd):
            for k in range((data.shape[2]-sz) // sd):
                out[i*sd:i*sd+sz,j*sd:j*sd+sz,k*sd:k*sd+sz], flag = lrf3d(data[i*sd:i*sd+sz,j*sd:j*sd+sz,k*sd:k*sd+sz], threshold=0.2, nsampling=nsampling)
                flags[i,j,k] = flag
    toc = time()
    print('sampling takes {} sec'.format(toc - tic))
    return out, flags

if __name__ == '__main__':
    fpath = './a3d/a3d/06323e0c225d04e325d70d6adc0240ef.a3d'
    data = read_data(fpath)
    data = compress_data(data, 4)
    out,_ = ostu3d(data)
    h = get_height(out)
    print('height: {}'.format(h))

    plt.imshow(out[:,:,int(h/2)])
    plt.show()

    sampled, flags = sampling3d(out, sz=4, sd=4, nsampling=2)
    plt.imshow(sampled[:,:,int(h/2)])
    plt.show()

    img = pc2img(sampled, axis=1)
    plt.imshow(img)
    plt.show()
