from utils import *
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN
import config

def cost(cen, pts, ratio, radius_penalty):
    a, b, c, d = cen
    dis = 0
    for x,y in pts:
        margin = np.sqrt((x-a-256//ratio)**2/b**2 + (y-c-256//ratio)**2/d**2) - 1
        margin = margin# * 1.0 ** np.sign(margin)
        dis += margin / (b**2 + d** 2) ** radius_penalty
    return dis

def cost2(cen, pts, ratio, radius_penalty):
    a, b, c, d, e = cen
    dis = 0
    for x,y in pts:
        if x < 256//ratio:
            margin = np.sqrt((x+a-256//ratio - e)**2/b**2 + (y-c-256//ratio)**2/d**2) - 1
            # margin = np.exp(margin)
            dis += margin / (b**2 + d** 2) ** radius_penalty / 2
        if x > 256//ratio:
            margin = np.sqrt((x-a-256//ratio - e)**2/b**2 + (y-c-256//ratio)**2/d**2) - 1
            # margin = np.exp(margin)
            dis += margin / (b**2 + d** 2) ** radius_penalty / 2
    return dis

def cost3(cen, pts, ratio, radius_penalty):
    # a - disp from boundary of head to boundary of arm
    # b - r of arm
    # c - y disp of arm
    # d - r of head
    # e - y disp of head
    # f - x disp of head

    a, b, c, d, e, f = cen
    dis = 0
    for x,y in pts:
        if x < 256//ratio -f - (a+b+d)/2:
            margin = np.sqrt((x + a + d + b - 256//ratio - f)**2/b**2 + (y-c-256//ratio)**2/b**2) - 1
            # margin = np.exp(margin)
            dis += margin / ((b**2 * 2) ** radius_penalty * 2 + (d**2 * 2) ** radius_penalty)
        elif x > 256//ratio -f + (a+b+d)/2:
            margin = np.sqrt((x - a - d - b - 256//ratio - f)**2/b**2 + (y-c-256//ratio)**2/b**2) - 1
            # margin = np.exp(margin)
            dis += margin / ((b**2 * 2) ** radius_penalty * 2 + (d**2 * 2) ** radius_penalty)
        else:
            margin = np.sqrt((x-256//ratio-f)**2/d**2 + (y-e-256//ratio)**2/d**2) - 1
            # margin = np.exp(margin)
            dis += margin / ((b**2 * 2) ** radius_penalty + (d**2 * 2) ** radius_penalty)
    return dis

def softmin(x):
    """Compute softmin values for each sets of scores in x."""
    e_x = np.exp(np.negative(x) - np.min(x))
    return e_x / e_x.sum()


def plot_ecllipse(x, pts, ratio):
    x1, x2, x3 = x
    # plot results
    fig = plt.figure(0, figsize=[12, 18])
    # if 1 in hypo:
    ec = Ellipse(xy=[x1[0]+256//ratio, x1[2]+256//ratio], width=x1[1]*2, height=x1[3]*2, angle=0)
    ax1 = fig.add_subplot(131, aspect='equal')
    ax1.scatter(pts[:,0], pts[:,1], marker='o')
    ax1.add_artist(ec)
    ec.set_clip_box(ax1.bbox)
    ec.set_alpha(0.3)
    ec.set_facecolor([1, 0, 0])
    ax1.set_xlim(0, 512//ratio)
    ax1.set_ylim(0,512//ratio)
    #if 2 in hypo:
    ec1 = Ellipse(xy=[ x2[0]+256//ratio,  x2[2]+256//ratio], width=x2[1]*2, height=x2[3]*2, angle=0)
    ec2 = Ellipse(xy=[-x2[0]+256//ratio, x2[2]+256//ratio], width=x2[1]*2, height=x2[3]*2, angle=0)
    ax2 = fig.add_subplot(132, aspect='equal')
    ax2.scatter(pts[:,0], pts[:,1], marker='o')
    for e in [ec1, ec2]:
        ax2.add_artist(e)
        e.set_clip_box(ax2.bbox)
        e.set_alpha(0.3)
        e.set_facecolor([1, 0, 0])
        ax2.set_xlim(0, 512//ratio)
        ax2.set_ylim(0,512//ratio)
    # if 3 in hypo:
    ec1 = Ellipse(xy=[ 256//ratio+x3[5],  x3[4]+256//ratio], width=x3[3]*2, height=x3[3]*2, angle=0)
    ec2 = Ellipse(xy=[-x3[0]-x3[1]-x3[3]+256//ratio+x3[5], x3[2]+256//ratio], width=x3[1]*2, height=x3[1]*2, angle=0)
    ec3 = Ellipse(xy=[ x3[0]+x3[1]+x3[3]+256//ratio+x3[5],  x3[2]+256//ratio], width=x3[1]*2, height=x3[1]*2, angle=0)
    ax3 = fig.add_subplot(133, aspect='equal')
    ax3.scatter(pts[:,0], pts[:,1], marker='o')
    for e in [ec1, ec2, ec3]:
        ax3.add_artist(e)
        e.set_clip_box(ax3.bbox)
        e.set_alpha(0.3)
        e.set_facecolor([1, 0, 0])
        ax3.set_xlim(0, 512//ratio)
        ax3.set_ylim(0, 512//ratio)
    plt.show()

def cluster_gen(pts, ratio=4, if_plot=False, radius_penalty=1):
    '''
    centroids, sizes = cluster_gen_with_hypo(pts, n_layer, height, last_n_cluster, n_cluster=0, if_plot=False, verbose=False, radius_penalty=1)
    '''
    res = []
    x1 = [0, 150//ratio, 0//ratio, 120//ratio]
    x2 = [150//ratio, 100//ratio, 0//ratio, 100//ratio, 0//ratio]
    x3 = [10//ratio, 60//ratio, 0//ratio, 70//ratio, 0//ratio, 0//ratio]
    if len(config.last_geom):
        if config.last_n_cluster == 1:
            x1 = config.last_geom[0]
            x3[5] = config.last_geom[0][0]
            x3[4] = config.last_geom[0][2]
        elif config.last_n_cluster == 2:
            x2 = config.last_geom[1]
        elif config.last_n_cluster == 3:
            x3 = config.last_geom[2]

    bnds1 = [(-60//ratio, 60//ratio), (50//ratio, 200//ratio), (-60//ratio, 60//ratio), (50//ratio, 200//ratio)]
    res.append(optimize.minimize(cost, x1, args=(pts, ratio, radius_penalty,), method='L-BFGS-B', bounds=bnds1))

    bnds2 = [(0//ratio, 256//ratio), (10//ratio, 150//ratio), (-100//ratio, 100//ratio), (20//ratio, 150//ratio), (-60//ratio, 60//ratio)]
    res.append(optimize.minimize(cost2, x2, args=(pts, ratio, radius_penalty,), method='L-BFGS-B', bounds=bnds2))

    bnds3 = [(-20//ratio, 200//ratio), (40//ratio, 120//ratio), (-60//ratio, 60//ratio), (60//ratio, 80//ratio), (-60//ratio, 60//ratio), (-60//ratio, 60//ratio)]
    res.append(optimize.minimize(cost3, x3, args=(pts, ratio, radius_penalty,), method='L-BFGS-B', bounds=bnds3))


    x = [r.x for r in res]
    scores = [r.fun for r in res]
    probs = softmin(scores)
    if if_plot:
        plot_ecllipse(x, pts, ratio)
    return x, probs


if __name__ =='__main__':
    pass