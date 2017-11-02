import utils
import os
from glob import glob
from skimage.io import imsave

fpaths = glob('./a3d/a3d/*a3d')
output_dir = 'projections'
for fpath in fpaths:
    fname = utils.get_name(fpath)
    data = utils.read_data(fpath)
    img = utils.pc2img(data, threshold=0.1, axis=1)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    imsave(os.path.join(output_dir, fname + '.png'), img)
    exit()
