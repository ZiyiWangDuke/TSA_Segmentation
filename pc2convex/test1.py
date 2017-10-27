'''
The script generate the flags, interpolated bounds, 2D projections
to test the performance of convex pose on more cases
'''

import convex
import utils
import segmentation
import pose
import config
import matplotlib.pyplot as plt
import numpy as np
import os
from smooth import interp
from glob import glob
from time import time
import pickle

fpaths = glob('/ext2/seg/PSRC/Data/stage1/a3d/*a3d')
ratio = 8
n_layers = 30
ifplot=False
output_dir = './outputs2'

if os.path.exists(output_dir):
    pass
else:
    os.mkdir(output_dir)

for fpath in fpaths:
    fname = utils.get_name(fpath)
    data_ = utils.read_data(fpath)
    data = utils.compress_data(data_, ratio)
    bita, _ = utils.ostu3d(data) # binary data
    layers = np.hstack([np.linspace(0, bita.shape[2] // 2, n_layers // 3, dtype=np.int16),
                        np.linspace(bita.shape[2] // 2 + 1, bita.shape[2] - 1, n_layers // 3 * 2, dtype=np.int16)])
    x, y, z, b, d, flags = pose.frame_gen(bita, layers, ratio=ratio, penalty=1, min_thresh=15, verbose=False, if_plot_frame=False)
    height = utils.get_height(bita)
    last_layer = int(height * n_layers // bita.shape[2])  # indx of the highest slice
    flags = segmentation.find_chin(bita, flags)

    ll, rl, bd, la, hd, ra = segmentation.frame_parser(x, y, z, b, d, flags)

    flags = segmentation.flag_parser(flags, ll, rl, bd, la, hd, ra, last_layer)
    flags = segmentation.find_chest_th(bita, bd, flags)
    segs = [ll, rl, bd, la, hd, ra]
    segs = [interp(x) for x in segs]

    tic = time()
    # with raw data
    bita_, _ = utils.ostu3d(data_)
    pts_ = utils.get_points(bita_, thresh=0)
    labels_ = segmentation.get_labels(pts_, flags, segs, ratio=ratio)
    print('labling takes {}s.'.format(time() - tic))

    # for vis
    # tic = time()
    # pts = utils.get_points(bita, thresh=0)
    # labels = segmentation.get_labels(pts, flags, segs, ratio=1)

    pts, labels = segmentation.get_labels_from_data(data_, flags, segs, ratio=1)

    if ifplot:
        utils.seg_vis2d(pts, labels, bita, savefig=True, fname= fname)
        print('visualization takes {}s'.format(time() - tic))

    segs_ = [np.array([np.hstack([pt, data_[pt.astype(np.int16)], labels_[j], data_[pt.astype(np.int16)]]) for j, pt in enumerate(pts_) if labels_[j] == i]) for i in range(18)]
    np.save('./outputs/'+fname+'_segs.npy', segs_)

    # np.save('./outputs/'+fname+'_pts.npy', pts)
    # np.save('./outputs/' + fname + '_labels.npy', labels)
    # with open('./outputs/' + fname + '.txt', 'wb') as fid:
    #     pickle.dump(flags, fid)
    # break



