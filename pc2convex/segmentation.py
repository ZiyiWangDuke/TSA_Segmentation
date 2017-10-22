from pose import *
from sklearn.decomposition import pca

import matplotlib.pyplot as plt

# find the neck using connected component in the vertical slice.
# find the neck using connected component in the vertical slice.
def find_chin(data, flags, eps=2, min_samples=3, ratio=8):
    img = np.flipud(pc2img(data, axis=1))
    for indx in range(flags['shoulder'], int(flags['height'])):
        slice = img[indx, :]
        pts = get_points(slice.reshape([-1,1]))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
        _, counts = np.unique(db.labels_, return_counts=True)
        n_cluster = np.sum(counts >= min_samples)
        if n_cluster > 2 and flags['chin'] == 0:
            flags['chin'] = indx
        if flags['chin'] != 0 and n_cluster == 2:
            pts1 = np.array([pts[i,0] for i, l in enumerate(db.labels_) if l==0])
            pts2 = np.array([pts[i,0] for i, l in enumerate(db.labels_) if l==1])
            if abs(max(pts1) - min(pts2)) > 80 /ratio or abs(min(pts1) - max(pts2)) < 80 / ratio:
                flags['top'] = indx
                break
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

def inconvex(pt, slices, tolerance=1.1):
    px, py, pz = pt
    indx = np.where(slices[:,2]==pz)[0][0]
    x, y, z, b, d = slices[indx]
    return ((px-x)/b)**2 + ((py-y)/d)**2 < tolerance


def get_labels(pts, flags, segs, ratio=4):
    labels = []
    height = np.amax(pts[:,2])
    ll, rl, bd, la, ra, hd = segs
    for pt in pts:
        x, y, z = np.divide(pt, ratio)
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
            if x < (flags['thigh_left'][0] + flags['thigh_right'][0]) / 2:
                labels.append(11)
            else:
                labels.append(12)

        elif z <= flags['waist'][1]:
            th1 = np.arctan2(flags['waist_left'][1] - flags['thigh_left'][1],
                             flags['waist_left'][0] - flags['thigh_left'][0])
            th2 = np.arctan2(flags['waist_right'][1] - flags['thigh_right'][1],
                             flags['waist_right'][0] - flags['thigh_right'][0])
            theta1 = np.arctan2(z - flags['thigh_left'][1], x - flags['thigh_left'][0])
            theta2 = np.arctan2(z - flags['thigh_right'][1], x - flags['thigh_right'][0])
            if theta1 > th1:
                labels.append(8)
            elif theta2 < th2:
                labels.append(10)
            else:
                labels.append(9)
        elif z <= flags['chest'][2]:
            if x < flags['chest'][0]:
                labels.append(6)
            else:
                labels.append(7)
        elif z <= flags['chin']:
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
        elif z <= (flags['top'][1] * 1.5 + height * 0.5) / 2:
            if x < flags['head'][0] - flags['head'][3]:
                if z < flags['elbow_left']:
                    labels.append(1)
                else:
                    labels.append(2)
            elif x > flags['head'][0] + flags['head'][3]:
                if z < flags['elbow_right']:
                    labels.append(3)
                else:
                    labels.append(4)
            else:
                labels.append(0)
        else:
            if x < flags['top'][0]:
                if z < flags['elbow_left']:
                    labels.append(1)
                else:
                    labels.append(2)
            else:
                if z < flags['elbow_right']:
                    labels.append(3)
                else:
                    labels.append(4)

    return labels

if __name__ =='__main__':
    pass