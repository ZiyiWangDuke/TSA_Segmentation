from scipy.interpolate import interp1d
import numpy as np

def interpz(x, y):
    '''
    1d interpolate along z axis.
    :return:
    x, y, z, b, d the same length as data.shape[2]
    '''
    x = np.array(x)
    y = np.array(y)
    f = interp1d(x, y)
    x_new = np.arange(np.amin(x), np.amax(x))
    return f(x_new)

def interp(x):
    out = []
    z = x[:,2]
    for i in range(x.shape[1]):
            out.append(interpz(z, x[:,i]))
    return np.array(out).T
