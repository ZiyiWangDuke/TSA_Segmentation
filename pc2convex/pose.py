from convex import *
from utils import *

def hypo_gen(layer, height, last_n_clusters, reverse=False):
    if layer < (height / 2):
        if last_n_clusters == 2 and layer < height / 2:
            return [1, 2]# [3]
        else:
            return [1]#, [2,3]
    elif layer < height / 3 * 2:
            return [1]
    else:
        if last_n_clusters == 1:
            return [1, 3]#, [2]
        if last_n_clusters == 3:
            return [2, 3]#, [1]
        else:
            return [2]#, [1,3]

def cluster_parser(x, probs, hypo, ratio, flags):
    probs_ = [p if i+1 in hypo else 0 for i, p in enumerate(probs)]
    n_cluster = np.argmax(probs_) + 1
    if flags['counts'] < config.min_count:
        print('count < than min_count')
        n_cluster = flags['last_n_cluster']
    if n_cluster == 1:
        return [(x[0][0] + 256//ratio, x[0][2] + 256//ratio)], [(x[0][1], x[0][3])]
    elif n_cluster == 2:
        return [(x[1][0] + 256//ratio + x[1][4], x[1][2] + 256//ratio), (-x[1][0] + 256//ratio + x[1][4], x[1][2] + 256//ratio)], [(x[1][1], x[1][3]), (x[1][1], x[1][3])]
    elif n_cluster == 3:
        return [(x[2][0] + x[2][1] + x[2][3] + 256//ratio + x[2][5], x[2][2] + 256//ratio), (256//ratio + x[2][5], x[2][2] + 256//ratio), (-x[2][0] - x[2][1] - x[2][3] + 256//ratio + x[2][5], x[2][4] + 256//ratio)], [(x[2][1], x[2][1]), (x[2][3], x[2][3]), (x[2][1], x[2][1])]

# correct the hypothesis about the number of cluster using geometry properties.
# this is sensitive to noise
def hypo_correction(pts, prob, geom, hypo, ratio):
    center = geom[1][4] + 256 / ratio

    # case1: the two cluster are close
    if geom[1][0] <= geom[1][1] * 1.1:
        if 1 in hypo and 2 in hypo and prob[1] < prob[0]:
            if np.ptp(pts[np.argsort(np.abs(pts[:, 0] - center))[:len(pts) // 6]][:, 1]) < np.ptp(pts[:,1]) * config.weak_link:
                hypo.remove(1)
                print('There is a weak connection in the center thus 1 is removed')
        elif 1 in hypo and 2 in hypo and prob[1] > prob[0] :
            if np.ptp(pts[np.argsort(np.abs(pts[:, 0] - center))[:len(pts) // 6]][:, 1]) > np.ptp(pts[:,1]) * config.strong_link:
                hypo.remove(2)
                print('There is a strong connection in the center thus 2 is removed')

    # case2: the side cluster is minor
    if 2 not in hypo and 3 in hypo and prob[1] > prob[2] and prob[2] > prob[0]:
        hypo.remove(3)
        print('side cluster is trivial thus 1 is appended to hypo')

    # case3: unbalanced cluster
    if 1 in hypo and 3 in hypo and prob[2] > prob[0]:
        if np.amax(pts[:, 0]) < geom[2][0] + geom[2][1] * 1.2 + geom[2][3]:
            hypo.remove(3)
            print('unbalanced side cluster')

    return hypo

def frame_gen(data, layers, ratio, penalty=1, min_thresh=10, if_plot_frame=True,  verbose=True):
    x = []
    y = []
    z = []
    b = []
    d = []
    n_clusters = []
    height = get_height(data)
    flags = {'hip':0, 'shoulder':0, 'neck':0, 'top':0, 'counts':0, 'height':height, 'last_n_cluster':2, 'chin':0, 'disp': np.mean(np.diff(layers))}
    tic = time()
    geom = []
    for i in layers:
        image = data[:,:,i]
        pts = get_points(image)
        if len(pts) < min_thresh:
            continue
        hypo = hypo_gen(i, height, flags['last_n_cluster'])
        geom, prob = cluster_gen(pts, ratio, if_plot=False, radius_penalty=penalty)


        hypo = hypo_correction(pts, prob, geom, hypo, ratio)
        # hypo correction
        centroids, sizes = cluster_parser(geom, prob, hypo, ratio, flags)
        config.last_n_cluster = len(centroids)
        config.last_geom = geom

        # clear the data:
        # for pt in pts:
        #     out_flags = [(pt[0] - cen[0])**2 / sz[0]**2 + (pt[1] - cen[1])**2 / sz[1]**2 > 1.1 for cen, sz in zip(centroids, sizes)]
        #     if np.sum(out_flags) == len(centroids):
        #         data[pt[0], pt[1], i] = 0
        if verbose:
            print(i, len(centroids), hypo, prob)

        if flags['last_n_cluster'] == len(centroids):
            flags['counts'] += 1
        else:
            flags['counts'] = 1

        # when n_cluster changes
        if flags['last_n_cluster']==2 and len(centroids)==1:
            flags['hip'] = i
        elif flags['last_n_cluster']==1 and len(centroids)==3:
            flags['shoulder'] = i
        elif flags['last_n_cluster']==3 and len(centroids)==2:
            flags['top'] = i

        flags['last_n_cluster'] = len(centroids)
        for cen, sz in zip(centroids, sizes):
            x.append(cen[0])
            y.append(cen[1])
            z.append(i)
            n_clusters.append(len(centroids))
            b.append(sz[0])
            d.append(sz[1])
    toc = time()
    print('frame generated in {} s'.format(toc-tic))
    if if_plot_frame:
        fig, axes = plt.subplots(ncols=2, figsize=(12,6))
        ax1, ax2 = axes
        img1 = pc2img(data, axis=1)
        img2 = pc2img(data, axis=0)
        rgb_img1 = np.swapaxes(np.swapaxes(np.stack((img1,)*3), 0, 2), 0, 1)
        rgb_img2 = np.swapaxes(np.swapaxes(np.stack((img2,)*3), 0, 2), 0, 1)
        ax1.imshow(np.flipud(rgb_img1), origin='lower')
        ax2.imshow(np.flipud(rgb_img2), origin='lower')
        for i in range(len(x)):
            ax1.plot(x[i], z[i], 'o', linewidth=5, color='r')
            ax1.plot(x[i]+b[i], z[i], 'o', linewidth=1, color='g')
            ax1.plot(x[i]-b[i], z[i], 'o', linewidth=1, color='g')

            ax2.plot(y[i], z[i], 'o', linewidth=5, color='r')
            ax2.plot(y[i]+d[i], z[i], 'o', linewidth=1, color='g')
            ax2.plot(y[i]-d[i], z[i], 'o', linewidth=1, color='g')
        plt.show()

    return x, y, z, b, d, flags

def save_pts(fpath, X, Y, Z, ratio):
    with open(fpath, 'w') as fid:
        for x, y, z in zip(X, Y, Z):
            fid.write('{}, {}, {}'.format(x*ratio, y*ratio, z*ratio))
    return

def frame_parser(x, y, z, b, d, flags):
    # put joints into different classes
    z_set, z_counts = np.unique(z, return_counts=True)
    idx = np.add.accumulate(z_counts) - 1
    # take advantage of human structure
    legflag = True
    bodyflag = True
    headflag = True
    ll = []
    rl = []
    bd = []
    la = []
    hd = []
    ra = []

    for z_, c, i in zip(z_set, z_counts, idx):
        if c == 2:
            pts = np.array([(x[i - j], y[i - j], z_, b[i - j], d[i - j]) for j in range(2)])
            pts = pts[pts[:, 0].argsort()]
            if z_ <=flags['hip']:
                ll.append(pts[0])
                rl.append(pts[1])
            else:
                la.append(pts[0])
                ra.append(pts[1])
        elif c == 1:
                bd.append((x[i], y[i], z_, b[i], d[i]))
        elif c== 3:
            pts = np.array([(x[i - j], y[i - j], z_, b[i - j], d[i - j]) for j in range(3)])
            pts = pts[pts[:, 0].argsort()]
            if z_ < flags['chin']:
                la.append(pts[0])
                bd.append(pts[1])
                ra.append(pts[2])
            else:
                la.append(pts[0])
                hd.append(pts[1])
                ra.append(pts[2])
        else:
            print('Error: wrong number of centroids')
    ll = np.array(ll)
    rl = np.array(rl)
    bd = np.array(bd)
    la = np.array(la)
    hd = np.array(hd)
    ra = np.array(ra)
    return ll, rl, bd, la, hd, ra

if __name__ =='__main__':
    pass