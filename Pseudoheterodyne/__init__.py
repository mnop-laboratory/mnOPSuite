import numpy as np
from scipy.optimize import minimize

p=[0]

def fit_phase_leakage(leakage_arr,amps,n=5):

    amps=np.array(amps)
    leakage_arr=np.array(leakage_arr)
    amps = amps[np.isfinite(leakage_arr)]
    leakage_arr = leakage_arr[np.isfinite(leakage_arr)]

    global p,best_amp
    if n>len(amps)-1: n=len(amps)-1
    p=np.polyfit(x=amps,y=leakage_arr,deg=n)

    fit = np.polyval(p,amps)

    amps_dense = np.linspace(np.min(amps),np.max(amps),10*len(amps))
    vals = np.polyval(p,amps_dense)
    best_amp = amps_dense[np.argmin(vals)]

    return fit

def get_best_amp(amps):

    global best_amp

    return best_amp


