import numpy as np
from scipy.optimize import minimize

p=[0]

def fit_phase_leakage(leakage_arr,amps,n=5):

    amps=np.array(amps)
    leakage_arr=np.array(leakage_arr)
    amps = amps[np.isfinite(leakage_arr)]
    leakage_arr = leakage_arr[np.isfinite(leakage_arr)]

    global p
    if n>len(amps)-1: n=len(amps)-1
    p=np.polyfit(x=amps,y=leakage_arr,deg=n)

    fit = np.polyval(p,amps)

    return fit

def get_best_amp(amps):

    global p
    to_minimize=lambda amp: np.polyval(p,amp)

    bounds0=(np.min(amps),np.max(amps))
    result = minimize(to_minimize,
                      x0=np.mean(amps),
                      bounds=(bounds0,))

    return result.x


