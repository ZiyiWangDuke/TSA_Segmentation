from pose import *
from sklearn.decomposition import pca

import matplotlib.pyplot as plt

# find the neck using connected component in the vertical slice.
# find the neck using connected component in the vertical slice.
def find_chin(data, flags, eps=2, min_samples=3, ratio=8):
    img = np.flipud(pc2img(data, axis= 1))
    for indx in range(flags['shoulder'], int(flags['height'])):
        slice = img[indx, :]
        pts = get_points(slice.reshape([-1,1]))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
        _, counts = np.unique(db.labels_, return_counts=True)
        n_cluster = np.sum(counts >= min_samples)
        if n_cluster > 2 and flags['chin'] == 0:
            flags['chin'] = indx
    return flags

def find_chest_th(data, bd, flags):
    # find the principle axis of the chest
    s = []
    for i in bd[:, 2]:
        if i >= flags['chest'][2]:
            pts = get_points(data[:, :, int(i)], thresh=0)
            u_, s_, v_ = np.linalg.svd(pts.astype(np.float32))
            s.append(s_)
    s = np.mean(s, axis=0)
    print(s)
    flags['chest_theta'] = np.arctan2(-s[1], s[0])
    return flags

def flag_parser(flags, ll, rl, bd, la, hd, ra, last_layer):
    center1 = la[len(la)//2-2:len(la)//2+2]
    center2 = ra[len(ra)//2-2:len(ra)//2+2]
    flags['elbow_left'] = center1[np.argsort(center1[:,0])][ 0, 2]
    flags['elbow_right'] = center2[np.argsort(center2[:,0])][-1, 2]
    flags['feet']  = ((ll[len(ll)*1//3, 0] + rl[len(ll)*1//3, 0])/2, ll[len(ll)*1//3, 2], np.amax(ll[:len(ll)*1//3, 3]), np.amax(ll[:len(ll)*1//3, 4]))
    flags['lower_leg']  = ((ll[len(ll)*1//2, 0] + rl[len(ll)*1//2, 0])/2, ll[len(ll)*1//2, 2], np.amax(ll[:len(ll)*1//2, 3]), np.amax(ll[:len(ll)*1//2, 4]))
    flags['thigh_left']  = (ll[len(ll)*5//6, 0]+ll[len(ll)*5//6, 3]/3, ll[len(ll)*5//6, 2])
    flags['thigh_right'] = (rl[len(ll)*5//6, 0]-rl[len(ll)*5//6, 3]/3, rl[len(ll)*5//6, 2])
    flags['waist_left'] = (bd[last_layer//2 - len(rl), 0] - ll[len(ll)//2, 3], bd[last_layer//2-len(ll), 2])
    flags['waist_right']= (bd[last_layer//2 - len(rl), 0] + rl[len(ll)//2, 3], bd[last_layer//2-len(ll), 2])
    flags['waist'] = (bd[last_layer//2-len(ll), 0], flags['height']/2)
    flags['neck'] = (hd[0, 0], hd[0, 1], hd[0, 2])
    flags['chest'] = (bd[-len(hd), 0], bd[-len(hd), 1], flags['chin'] - 7.0 / 6.0* (flags['top'] - flags['chin']))
    flags['shoulder_left_lower']  = (bd[-len(hd), 0]-bd[-len(hd), 3], bd[-len(hd), 2])
    flags['shoulder_right_lower'] = (bd[-len(hd), 0]+bd[-len(hd), 3], bd[-len(hd), 2])
    flags['shoulder_left_higher'] = (hd[0,0]-hd[0,3], hd[0,2])
    flags['shoulder_right_higher']= (hd[0,0]+hd[0,3], hd[0,2])
    flags['head'] = np.mean(hd, axis=0)
    flags['top'] = (hd[-1, 0], flags['top'])#hd[-1,2])

    return flags

def inconvex(pt, slices, tolerance=1.05):
    px, py, pz = pt
    if pz < np.amin(slices[:,2]):
        indx = 0
    elif pz > np.amax(slices[:,2]):
        indx = -1
    else:
        indx = np.where(slices[:,2]==pz)[0]
        if not len(indx):
            return False
        indx = indx[0]
    x, y, z, b, d = slices[indx]
    return ((px-x)/b)**2 + ((py-y)/d)**2 < tolerance


def get_labels(pts, flags, segs, ratio=4):
    labels = []
    height = np.amax(pts[:,2])
    ll, rl, bd, la, ra, hd = segs
    for pt in pts:
        pt =  np.divide(pt, ratio).astype(np.int16)
        x, y , z = pt
        if z <= flags['feet'][1]:
            if inconvex(pt, ll):
                labels.append(15)
            elif inconvex(pt, rl):
                labels.append(16)
            else:
                labels.append(-1)

        elif z <= flags['lower_leg'][1]:
            if inconvex(pt, ll):
                labels.append(13)
            elif inconvex(pt, rl):
                labels.append(14)
            else:
                labels.append(-1)

        elif z <= flags['thigh_left'][1]:
            if inconvex(pt, ll):
                labels.append(11)
            elif inconvex(pt, rl):
                labels.append(12)
            else:
                labels.append(-1)

        elif z <= flags['waist'][1]:
            if inconvex(pt, ll) or inconvex(pt, rl) or inconvex(pt, bd):
                th1 = np.arctan2(flags['waist_left'][1] - flags['thigh_left'][1],
                                 flags['waist_left'][0] - flags['thigh_left'][0])
                th2 = np.arctan2(flags['waist_right'][1] - flags['thigh_right'][1],
                                 flags['waist_right'][0] - flags['thigh_right'][0])
                theta1 = np.arctan2(z - flags['thigh_left'][1], x - flags['thigh_left'][0])
                theta2 = np.arctan2(z - flags['thigh_right'][1], x - flags['thigh_right'][0])
                # on leg
                if z <= np.amax(ll[:,2]):
                    if theta1 >= th1 and inconvex(pt, ll):
                        labels.append(8)
                    elif theta2 <= th2 and inconvex(pt, rl):
                        labels.append(10)
                    elif inconvex(pt, ll) or inconvex(pt, rl):
                        labels.append(9)
                    else:
                        labels.append(-1)
                elif inconvex(pt, bd) or inconvex(pt, ll) or inconvex(pt, rl):
                    if theta1 >= th1:
                        labels.append(8)
                    elif theta2 <= th2:
                        labels.append(10)
                    else:
                        labels.append(9)
            else:
                labels.append(-1)

        elif z <= flags['chest'][2]:
            if x < flags['chest'][0] and inconvex(pt, bd):
                labels.append(6)
            elif inconvex(pt, bd):
                labels.append(7)
            else:
                labels.append(-1)

        elif z <= flags['chin']:
            if inconvex(pt, bd):
                th1 = np.arctan2(flags['shoulder_left_higher'][1] - flags['shoulder_left_lower'][1],
                                 flags['shoulder_left_higher'][0] - flags['shoulder_left_lower'][0])
                th2 = np.arctan2(flags['shoulder_right_higher'][1] - flags['shoulder_right_lower'][1],
                                 flags['shoulder_right_higher'][0] - flags['shoulder_right_lower'][0])
                theta1 = np.arctan2(z - flags['shoulder_left_lower'][1], x - flags['shoulder_left_lower'][0])
                theta2 = np.arctan2(z - flags['shoulder_right_lower'][1], x - flags['shoulder_right_lower'][0])
                if theta1 > th1:
                    labels.append(1)
                elif 0 <= theta2 and theta2 < th2:
                    labels.append(3)
                else:
                    vec = np.subtract(flags['neck'], flags['chest'])
                    cen = np.add(flags['chest'], vec * (z - flags['chest'][2]) / vec[2])
                    theta = np.arctan2(y - cen[1], x - cen[0])
                    if flags['chest_theta'] < theta and theta < flags['chest_theta'] + np.pi:
                        labels.append(17)
                    else:
                        labels.append(5)
            else:
                labels.append(-1)

        elif z <= (flags['top'][1] * 1.5 + height * 0.5) / 2:

            if inconvex(pt, la):
                if z < flags['elbow_left']:
                    labels.append(1)
                else:
                    labels.append(2)
            elif inconvex(pt, ra):
                if z < flags['elbow_right']:
                    labels.append(3)
                else:
                    labels.append(4)
            elif inconvex(pt, hd):
                labels.append(0)
            else:
                labels.append(-1)
        else:
            if inconvex(pt, la):
                if z < flags['elbow_left']:
                    labels.append(1)
                else:
                    labels.append(2)
            elif inconvex(pt, ra):
                if z < flags['elbow_right']:
                    labels.append(3)
                else:
                    labels.append(4)
            else:
                labels.append(-1)

    return labels


def get_labels_from_data(data, flags,
                         segs, ratio):
    x = np.arange(512)
    y = np.arange(512)
    z = np.arange(660)
    xv, yv, zv = np.meshgrid(x, y, z)
    pts = np.array([xv.ravel(), yv.ravel(), zv.ravel()]).T
    labels = get_labels(pts, flags, segs, ratio=ratio)
    return pts, labels

if __name__ =='__main__':
    pass