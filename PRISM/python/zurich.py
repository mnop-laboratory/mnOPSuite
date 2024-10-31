import numpy as np
from scipy.special import jv,jn_zeros

def get_compromise_V_and_XYgains(V_ideal,wl,wl0,gain0=1,search_n_zeros=50):

    gamma_star = 2.63
    gammas_n = jn_zeros(0,search_n_zeros)
    ind = np.argmin(np.abs(gammas_n - gamma_star*wl/wl0))
    gamma_n = gammas_n[ind]

    gamma_compromise = gamma_n * wl0/wl
    V_compromise = V_ideal * gamma_compromise/gamma_star

    ampY = jv(1,gamma_compromise)
    ampX = jv(2,gamma_compromise)
    desired_amp = jv(1,gamma_star) # would be the same as `jv(2,gamma_star)`

    Xgain = desired_amp/ampX * gain0
    Ygain = desired_amp/ampY * gain0

    return np.array([V_compromise,Xgain,Ygain])