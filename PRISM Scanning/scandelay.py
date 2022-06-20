import numpy as np
from scipy.interpolate import interp1d
import common
from common import numerics as num
from common.baseclasses import AWA



def bin_average(a, Nx, delay_calibration_factor=1):

    from scipy.stats import binned_statistic

    a = np.array(a).copy()
    y,x = list(a)
    x=x*delay_calibration_factor
    xmin=np.min(x)
    xmax=np.max(x)

    stat = binned_statistic(x, y,
                            statistic='mean',
                            bins=Nx,
                            range=(xmin, xmax))

    ynew = stat.statistic
    xnew = stat.bin_edges[:-1]

    keep = np.isfinite(ynew)
    xnew = xnew[keep]
    ynew = ynew[keep]

    xnew,ynew=zip(*sorted(zip(xnew,ynew)))

    return np.array([ynew, xnew])


def weighted_average(a1, a2, w1, w2):
    from scipy.interpolate import interp1d

    a1 = np.array(a1).copy()
    y1, x1 = list(a1)
    x1, y1 = zip(*sorted(zip(x1, y1)))
    y1 = np.array(y1)

    a2 = np.array(a2).copy()
    y2, x2 = list(a2)
    x2, y2 = zip(*sorted(zip(x2, y2)))
    y2 = np.array(y2)

    # interpolate second dataset to abscissa of first
    y2interp = interp1d(x2, y2, assume_sorted=True, bounds_error=False,\
                        fill_value='extrapolate',kind='cubic')(x1)

    # weighted sum
    yavg = (w1 * y1 + w2 * y2interp) / (w1 + w2)

    return np.array([yavg, x1])


def fit_envelope(f,sabs,fmin=400,expand_envelope=1):

    from scipy.optimize import minimize,leastsq

    global result
    sub=f>fmin
    fsub=f[sub]
    sabssub=sabs[sub]
    ind=np.argmax(sabssub)
    A=sabssub[ind]
    f0 = fsub[ind]
    df = f0/10
    x0=np.array([A,f0,df])

    def model(f,A,f0,df):
        return A*np.exp(-(f-f0)**2/(2*df**2))

    def to_minimize(x):

        A,f0,df=x
        delta = sabssub-model(fsub,A,f0,df)
        return delta

    result=leastsq(to_minimize,x0)

    envelope = model(f,*result[0])
    #envelope/=envelope.max() #The overall value will not be meaningful, normalize to 1 before applying
    envelope = envelope**(1/expand_envelope**2)

    return envelope

def fourier_xform(a,tsubtract=0,envelope=True):

    v,t=np.array(a)
    c=3e8 #speed of light in m/s
    t=t-tsubtract #subtract a time correction in ps
    t*=1e-12 #ps in seconds
    d=t*c*2 #distance in m
    d*=1e2 #distance in cm

    v = v-np.mean(v)
    w = np.blackman(len(v))
    wmax_ind = np.argmax(w)
    centerd = np.sum(d * v ** 2) / np.sum(v ** 2)
    imax_ind = np.argmin((centerd - d) ** 2)
    w = np.roll(w, imax_ind - wmax_ind, axis=0)

    v *= w
    v = AWA(v, axes=[d])
    s = num.Spectrum(v, axis=0)

    # This applies knowledge that d does not start at zero, but min value is meaningful
    f = s.axes[0]
    pcorr = -2 * np.pi * (np.min(d)) * f
    s *= np.exp(1j * pcorr)

    posf=(f>0)
    pvf=np.angle(s)[posf]
    sabs=np.abs(s)[posf]
    f=f[posf]

    if envelope: env=fit_envelope(f,sabs)
    else: env=[1]*len(f)

    # this will apply the knowledge that distance is not zero at first index of spectrum
    pcorr = -2 * np.pi * (np.min(d)) * f
    pvf += pcorr
    pvf = (pvf+np.pi)%(2*np.pi)-np.pi

    assert len(f)==len(sabs)==len(pvf)==len(env)

    return np.array([f,sabs,pvf,env])

def subtract_phase_slope(f,s,phase_slope):

    s=s*np.exp(-1j*f*phase_slope) #positive phase slope means removal of positive time delay

    return s

def accumulate_spectra(spectra, phase_slope=0, apply_envelope=True):
    # `Spectra` is expected to be a set of stacked row vectors in groups of four,
    # each group is indexed by row as:
    # 0: frequencies
    # 1: spectrum amplitude
    # 2: spectrum phase
    # 3: spectrum envelope

    spectra = np.array(spectra)
    assert len(spectra) % 4 == 0, 'Input spectra must come as stacked groups of 4 row vectors'

    from scipy.interpolate import interp1d

    s_accumulated = 0
    s_accumulated_abs = 0
    Nspectra = len(spectra) // 4
    for i in range(Nspectra):

        f = spectra[4 * i]
        sabs = spectra[4 * i + 1]
        sphase = spectra[4 * i + 2]
        env = spectra[4 * i + 3]
        s = sabs * np.exp(1j * sphase)

        # In case we have invalid entries (zeros), remove them
        where_valid = np.isfinite(f) * (f > 0)
        f = f[where_valid]
        s = s[where_valid]
        env = env[where_valid]

        if i == 0:
            f0 = f
        else:
            # interpolate to original frequencies
            s = interp1d(f, s, assume_sorted=True, bounds_error=False, \
                         fill_value=0, kind='cubic')(f0)
        if apply_envelope:
            env = interp1d(f, env, assume_sorted=True, bounds_error=False, \
                           fill_value=0, kind='cubic')(f0)
            s = s * (env / env.max())  # normalize the envelope to unity

        s_accumulated += s
        s_accumulated_abs += np.abs(s)

    s_accumulated /= Nspectra
    s_accumulated_abs /= Nspectra

    # Subtract a phase slope
    s_accumulated=subtract_phase_slope(f0,s_accumulated,phase_slope)

    s_accum_phase = np.angle(s_accumulated)

    return np.array([f0, s_accumulated_abs, s_accum_phase])

def accumulate_spectra_normalized(spectra, spectra_ref, phase_slope=0, apply_envelope=True):
    # `Spectra` is expected to be a set of stacked row vectors in groups of four,
    # each group is indexed by row as:
    # 0: frequencies
    # 1: spectrum amplitude
    # 2: spectrum phase
    # 3: spectrum envelope
    #
    # Same for reference spectra

    # We need to apply envelopes from reference to analyte spectra!
    # Spoof it: make a new version of analyte spectra with envelope
    # replaced by reference envelope

    spectra = np.array(spectra)
    spectra_ref = np.array(spectra_ref)

    if apply_envelope:
        assert len(spectra)==len(spectra_ref),\
            "Applying envelope requires same number of accumulations for both analyte and reference"
        assert len(spectra)%4==0

        assert len(spectra) % 4 == 0, 'Input spectra must come as stacked groups of 4 row vectors'

    from scipy.interpolate import interp1d

    s_accumulated = 0
    s_accumulated_abs = 0
    s_accumulated_ref = 0
    s_accumulated_abs_ref = 0
    Nspectra = len(spectra) // 4
    for i in range(Nspectra):

        ## Analyte spectrum components
        f = spectra[4 * i]
        sabs = spectra[4 * i + 1]
        sphase = spectra[4 * i + 2]
        s = sabs * np.exp(1j * sphase)

        # In case we have invalid entries (zeros), remove them
        where_valid = np.isfinite(f) * (f > 0)
        f = f[where_valid]
        s = s[where_valid]

        if i == 0:
            f0 = f
        else:
            # interpolate to original frequencies
            s = interp1d(f, s, assume_sorted=True, bounds_error=False, \
                         fill_value=0, kind='cubic')(f0)

        ## Reference spectrum components
        f_ref = spectra_ref[4 * i]
        sabs_ref = spectra_ref[4 * i + 1]
        sphase_ref = spectra_ref[4 * i + 2]
        env = spectra_ref[4 * i + 3]
        s_ref = sabs_ref * np.exp(1j * sphase_ref)

        # In case we have invalid entries (zeros), remove them
        where_valid = np.isfinite(f_ref) * (f_ref > 0)
        f_ref = f_ref[where_valid]
        s_ref = s_ref[where_valid]
        env = env[where_valid]

        s_ref = interp1d(f_ref, s_ref, assume_sorted=True, bounds_error=False, \
                     fill_value=0, kind='cubic')(f0)

        if apply_envelope:
            env = interp1d(f_ref, env, assume_sorted=True, bounds_error=False, \
                           fill_value=0, kind='cubic')(f0)
            env_norm = env/env.max() # normalize the envelope to unity
            s_ref = s_ref * env_norm
            s = s * env_norm

        s_accumulated += s
        s_accumulated_abs += np.abs(s)
        s_accumulated_ref += s_ref
        s_accumulated_abs_ref += np.abs(s_ref)

    s_norm = s_accumulated / s_accumulated_ref
    s_norm_abs = s_accumulated_abs / s_accumulated_abs_ref
    # having divided by a reference possibly with zeros, remove those entries
    where_valid = (s_accumulated_abs_ref > s_accumulated_abs_ref.max()/20)
    s_norm=s_norm[where_valid]
    s_norm_abs=s_norm_abs[where_valid]
    f0 = f0[where_valid]

    # Subtract a phase slope
    s_norm = subtract_phase_slope(f0, s_norm, phase_slope)

    s_norm_phase = np.angle(s_norm)

    return np.array([f0, s_norm_abs, s_norm_phase])