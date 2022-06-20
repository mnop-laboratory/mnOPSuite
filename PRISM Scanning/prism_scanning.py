import numpy as np
from scipy.optimize import leastsq

def phase_level(parr,px,py,p0):

    parr=np.array(parr)
    sx,sy=parr.shape
    idx,idy=np.ogrid[0:1:sx*1j,
                     0:1:sy*1j]

    poffset=p0+idx*px+idy*py

    pcorr=(parr+poffset + np.pi) % (2*np.pi) - np.pi

    return pcorr

level_params=[0,0,0]

def auto_phase_level(poffset, *level_params0,**kwargs):

    global level_params

    ignore_region = np.isclose(poffset, 0)
    while len(level_params0) < 3:
        level_params0 = list(level_params0) + [0]
    level_params0=np.array(level_params0)

    def to_minimize(args):
        pnew = phase_level(poffset, *args)
        pnew[ignore_region] = 0

        return np.abs(pnew).flatten()

    level_params = leastsq(to_minimize, level_params0,**kwargs)[0]

    pflattened = phase_level(poffset, *level_params)

    return pflattened

def get_level_params(x):
    global level_params
    return level_params
