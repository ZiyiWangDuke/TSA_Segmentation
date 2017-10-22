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

fpaths = glob('./a3d/a3d/*a3d')
ratio = 8
n_layers = 30
ifplot=True


if os.path.exists('./outputs'):
    pass
else:
    os.mkdir('./outputs')

for fpath in fpaths:
    fname = utils.get_name(fpath)
    data_ = utils.read_data(fpath)
    data = utils.compress_data(data_, ratio)
    bita, _ = utils.ostu3d(data) # binary data
    layers = np.hstack([np.linspace(0, bita.shape[2] // 2, n_layers // 3, dtype=np.int16),
                        np.linspace(bita.shape[2] // 2 + 1, bita.shape[2] - 1, n_layers // 3 * 2, dtype=np.int16)])
    x, y, z, b, d, flags = segmentation.frame_gen(bita, layers, ratio=ratio, penalty=1, min_thresh=15, verbose=False)
    height = utils.get_height(bita)
    last_layer = int(height * n_layers // bita.shape[2])  # indx of the highest slice

    ll, rl, bd, la, hd, ra = segmentation.frame_parser(x, y, z, b, d, flags)
    flags = segmentation.flag_parser(flags, ll, rl, bd, la, hd, ra, last_layer)
    segs = [ll, rl, bd, la, hd, ra]
    segs = [interp(x) for x in segs]

    # with raw data
    bita_ = utils.ostu3d(data)
    pts = utils.get_points(bita, thresh=0)
    labels = get_labels(pts, flags, ratio=ratio)

    if ifplot:
        utils.seg_vis2d(pts, labels, savefig=True, fname=fname)

    # ignore the
    segs_ = [np.array([pt for j, pt in enumerate(pts) if labels[j] == i]) for i in range(18)]

    with open('./output/'+fname+'.npy', 'w') as fid:
        np.save(fid, segs_)

    break




