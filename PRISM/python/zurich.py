import numpy as np
from scipy.special import jv,jn_zeros

def get_compromise_V_and_XYgains_old(V_ideal,wl,wl0,gain0=1,search_n_zeros=50):

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

def get_compromise_V_and_XYgains(Vosc_ideal,Vnode1_HeNe,Vnode2_HeNe,
                                 gain0=1,search_n_zeros=50):

    Vosc_HeNe = (Vnode1_HeNe+Vnode2_HeNe)/2
    dVosc_HeNe = Vnode2_HeNe-Vnode1_HeNe

    Voscs_HeNe = Vosc_HeNe + dVosc_HeNe * np.arange(-search_n_zeros // 2,
                                                    search_n_zeros // 2)

    # print('wish:',Vosc_ideal)
    ind = np.argmin(np.abs(Voscs_HeNe - Vosc_ideal))
    Vosc_compromise = Voscs_HeNe[ind]

    gamma_star = 2.63
    gamma_compromise = Vosc_compromise / Vosc_ideal * gamma_star

    ampY = jv(1, gamma_compromise)
    ampX = jv(2, gamma_compromise)
    desired_amp = jv(1, gamma_star)  # would be the same as `jv(2,gamma_star)`

    Xgain = desired_amp / ampX * gain0
    Ygain = desired_amp / ampY * gain0

    return np.array([Vosc_compromise,Xgain,Ygain])