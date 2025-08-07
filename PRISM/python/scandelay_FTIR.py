import os
import numpy as np
import traceback
import scipy
import pickle
from scipy.interpolate import interp1d
from common import numerics as num
from common.baseclasses import AWA
from common import numerical_recipes as numrec
from scipy.stats import binned_statistic
from scipy.optimize import leastsq,minimize

basedir = os.path.dirname(__file__)
diagnostic_dir = os.path.join(basedir,'..','Diagnostic')

dX=None #This will imply the same df throughout, until (say), `normalize_spectra`
Nbins=None

interp_kwargs=dict(bounds_error=False,
                   fill_value=0,
                   kind='cubic')

def bin_average(a, Nx, delay_calibration_factor=1):

    a = np.array(a).copy()
    y,x = list(a)
    x=x*delay_calibration_factor

    # If we are using `aligning` VIs, then the interferogram duration (frequency resolution)
    #  will be set for the duration of execution right here at the first execution of this function
    global dX
    xmin=np.min(x)
    if dX is None: xmax=xmin+dX
    else:
        xmax=np.max(x)
        dX = xmax-xmin

    stat = binned_statistic(x, y,
                            statistic='mean',
                            bins=Nx,
                            range=(xmin, xmax))

    ynew = stat.statistic
    xnew = stat.bin_edges[:-1]

    keep = np.isfinite(ynew) #If we had no value found within a bin, that bin is false
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

def flatten_interferogram(x,y,flattening_order=20):

    while flattening_order>0:
        try:
            y -= np.polyval(np.polyfit(x=x, y=y, deg=flattening_order), x)
            return y
        except:
            flattening_order -= 1

    return y

best_delay=None

def get_best_delay():

    global best_delay
    if best_delay is None: return 0
    return best_delay

def model_xs(indices,params):

    offset,period=params[:2]
    xs = offset
    amps_phases=params[2:]

    N=(len(amps_phases))//2 #harmonics to include
    for i in range(N):
        n=i+1
        amp,phase=amps_phases[2*i:2*(i+1)]
        xs += amp*np.cos(2*np.pi*n*indices/period+phase)

    return xs

xaxis=None


# @TODO: 2025.07.22 - changes needed for multi-channel
#               Currently, `intfgs_arr` is 2D with shape: (2*Ncycles, Nxs)
#               What we want:  (Ncycles,1+Nchannels,Nxs)
#               Apparently we also need to put the x-axis as the *first* channel
#               --> So, higher level changes need include:
#                   * Make incoming `intfgs_arr` to be size (Ncycles,1+Nchannels,Nxs)
#                   * Put time/x-axis as first channel
def align_interferograms_new(intfgs_arr, delay_calibration_factor=1,
                                shift0=None, optimize_shift=True, shift_range=15,
                                flattening_order=5, noise=0,
                                fit_xs=True, fit_xs_order=1, smooth_xs = 100, maxfev=100,
                             shift_channel=0):

    global dX,best_delay,xaxis,all_xs0,all_xs_fitted
    global intfg_mutual_fwd,intfg_mutual_bwd
    if best_delay is None: best_delay=shift0

    intfgs_arr = np.array(intfgs_arr)
    Ncycles,Nchannels,Nts = intfgs_arr.shape
    Nchannels -= 1 #Because the zeroth channel should be x-axis

    # --- Process data
    all_xs = [] # a list of 1D arrays, later will be averaged to a single 1D array
    all_ys = [] # a list of 2D arrays, each (Nchannels,Nts), later will be averaged to a single 2D array
    for i in range(Ncycles):
        # i=i+10
        xs = intfgs_arr[i,0]
        ys = intfgs_arr[i,1:]
        all_ys.append(ys)
        all_xs.append(xs)

    # We commit to aligning fwd/bwd, but aligning every intfg sample is too much
    #    So we through out the intfg-to-intfg x-axis variation here
    #    We assume intfg-to-intfg x-axis variations will average out when we add the interferograms
    all_xs = np.mean(np.array(all_xs),axis=0) # now a single 1D array
    all_ys = np.mean(np.array(all_ys),axis=0) # Now we have a 2D array of size (Nchannels,Nts)

    if fit_xs:
        Nsamples = len(all_xs)
        indices = np.arange(Nsamples)
        offset = np.mean(all_xs)
        period = float(Nsamples)
        amps = [0] * fit_xs_order
        amps[0] = np.ptp(all_xs) / 2
        phases = [0] * fit_xs_order
        params0 = [offset, period]
        for amp, phase in zip(amps, phases): params0 = params0 + [amp, phase]
        params, _ = numrec.ParameterFit(indices, all_xs, model_xs, params0, maxfev=maxfev)
        amps_phases = np.array(params[2:]).reshape((fit_xs_order,2))
        print('Fitted sinusoid parameters for X coordinate:')
        for i,(amp,phase) in enumerate(amps_phases):
            print('Harmonic %i: amp=%1.2G, phase=%1.2f'%(i+1,amp,phase%(2*np.pi)))
        all_xs0 = all_xs
        all_xs = model_xs(indices, params)
        all_xs_fitted = all_xs

    # Remove a polynomial background from each channel data
    for chan_ys in all_ys:
        chan_ys -= np.polyval(np.polyfit(x=all_xs, y=chan_ys, deg=flattening_order), all_xs)

    # --- Get fwd/bwd interferograms
    idx_turn = np.argmax(all_xs) #We assume max is in midway point, BEFORE applying calibration factor
    all_xs *= delay_calibration_factor
    all_xs_fwd = all_xs[idx_turn:]
    all_xs_bwd = all_xs[:idx_turn]
    all_ys_fwd = all_ys[:,idx_turn:]
    all_ys_bwd = all_ys[:,:idx_turn]

    #--- Determine and update the global x-range, if needed
    x1 = np.min(all_xs)
    x2 = np.max(all_xs)
    Nbins = len(all_xs)//2
    if xaxis is None:
        print('Re-using pre-existing x-axis for interferograms...')
        xaxis = np.linspace(x1, x2, Nbins)

    def shifted_intfgs(shift):

        all_xs_rolled = np.roll(all_xs, shift, axis=0)
        result = binned_statistic(all_xs_rolled, all_ys, bins=Nbins) # Assume `all_ys` is a sequence of intfgs
        xnew = result.bin_edges[:-1]
        intfgs_new = result.statistic # This will be a sequence of intfgs

        keep = np.prod(np.isfinite(intfgs_new),axis=0) # all channels (along axis=0) must be simultaneously finite
        keep = keep.astype(bool) # So it is not confused with index values
        intfgs_new = intfgs_new[:,keep]
        xnew = xnew[keep]

        return xnew, intfgs_new

    #--- If we don't need to optimize fwd/bwd shift, return what we need
    if not optimize_shift:
        best_delay = shift0
        x, intfgs = shifted_intfgs(shift0)

        intfg_interp = interp1d(x=x, y=intfgs,
                                axis=-1,
                                **interp_kwargs) # Interpolator is happy to apply along axis to a sequence of arrays in `y`
        intfgs_new = intfg_interp(xaxis)

        return np.vstack([xaxis,intfgs_new])

    #--- Now proceed to find a global time-delay shift of `all_ys` relative to `all_xs`.
    # Once it is found, we'll `roll` the channels `all_ys` along the time axis.
    # Binning the `all_ys` with respect to `all_xs` will yield intfgs aligned to a uniformly increasing x-axis.
    # These latter two steps are both accomplished by `shifted_intfgs`.

    #--- Get fwd / bwd interferograms and their mutual overlap
    result = binned_statistic(all_xs_fwd,
                              all_ys_fwd,
                              bins=Nbins)
    x_fwd = result.bin_edges[:-1]
    intfgs_fwd = result.statistic
    where_keep=np.prod(np.isfinite(intfgs_fwd),axis=0) # all channels (along axis=0) must be simultaneously finite
    where_keep = where_keep.astype(bool) # Before we use it, need boolean
    intfgs_fwd=intfgs_fwd[:,where_keep]
    x_fwd = x_fwd[where_keep] # Should be uniformly increasing
    intfgs_fwd = AWA(intfgs_fwd, axes=[None,x_fwd]) # First axis is channels

    result = binned_statistic(all_xs_bwd,
                              all_ys_bwd,
                              bins=Nbins)
    x_bwd = result.bin_edges[:-1]
    intfgs_bwd = result.statistic
    where_keep=np.prod(np.isfinite(intfgs_bwd),axis=0) # all channels (along axis=0) must be simultaneously finite
    where_keep = where_keep.astype(bool) # Before we use it, need boolean
    intfgs_bwd=intfgs_bwd[:,where_keep]
    x_bwd = x_bwd[where_keep] # Should be uniformly increasing
    intfgs_bwd = AWA(intfgs_bwd, axes=[None,x_bwd]) # First axis is channels

    xmin = np.max((np.min(x_fwd), np.min(x_bwd)))
    xmax = np.min((np.max(x_fwd), np.max(x_bwd)))
    xmutual = np.linspace(xmin, xmax, Nbins)
    #raise ValueError
    intfgs_mutual_fwd = intfgs_fwd.interpolate_axis(xmutual, axis=-1,**interp_kwargs)
    intfgs_mutual_bwd = intfgs_bwd.interpolate_axis(xmutual, axis=-1,**interp_kwargs)

    #--- Define some helper functions for identifying x-coordinate of interferograms
    def get_dx(intfg_mutual_fwd, intfg_mutual_bwd, exp=4): # Here we must supply a single intfg channel as the basis for uncovering dx.
        # This is very clever way to estimate the x-axis shift of two nominally identical interferograms.
        # If there is an offset, then the relative phase will have the form:
        #       p = 2*pi * dx *freqs
        # In this case, dx takes the form of n=1 coefficient in a polynomial fit to p with respect to freqs.

        sf = num.Spectrum(intfg_mutual_fwd-np.mean(intfg_mutual_fwd), axis=0)
        sb = num.Spectrum(intfg_mutual_bwd-np.mean(intfg_mutual_bwd), axis=0)
        spow = np.abs(sf) ** exp + np.abs(sb) ** exp
        keep = (spow >= spow.max() / 10) * (sf.axes[0] > 0)
        norm = sf / sb
        norm = numrec.smooth(norm, window_len=5, axis=0)
        norm = norm[keep]
        spow = spow[keep]
        f, p = norm.axes[0], np.unwrap(np.angle(norm))

        p = np.polyfit(x=f, y=p, w=spow, deg=1)
        dx = -p[0] / (2 * np.pi)

        return dx  # This is the amount we need to now shift fwd with respect to bwd.

    def get_x0(intfg_mutual_fwd, intfg_mutual_bwd, exp=2):

        x = intfg_mutual_fwd.axes[0]

        intfg_fwd = intfg_mutual_fwd
        intfg_bwd = intfg_mutual_bwd
        w = np.abs(intfg_fwd) ** exp + np.abs(intfg_bwd) ** exp

        x0 = np.sum(x * w) / np.sum(w)

        return x0

    #--- Find average characteristic index of x0_fwd vs x0_bwd = shift0
    if not shift0:
        # Use the specified shift channel, and perform analysis there.
        intfg_mutual_fwd = intfgs_mutual_fwd[shift_channel]
        intfg_mutual_bwd = intfgs_mutual_bwd[shift_channel]

        x0 = get_x0(intfg_mutual_fwd, intfg_mutual_bwd, exp=4)
        print('Interferograms center (avg fwd/bwd) x0:', x0)
        dx = get_dx(intfg_mutual_fwd, intfg_mutual_bwd, exp=4)
        print('Fwd/bwd dx separation:', dx)
        x0_fwd = x0 + dx / 2
        x0_bwd = x0 - dx / 2

        n2 = np.argmin(np.abs(all_xs_fwd - x0_fwd)) # Index of fwd intfg peak
        n1 = np.argmin(np.abs(all_xs_fwd - x0_bwd)) # Index of bwd intfg peak
        dn=np.round((n2 - n1) / 2.) # how much we should shift bwd relative to fwd

        shift0 = int(dn)
        print('Initially optimal shift:', shift0)

    # -- Optimize around shift0
    shifts = np.arange(-shift_range, shift_range, 1) + shift0
    sums = []
    for shift in shifts:
        x, intfgs = shifted_intfgs(shift)
        sums.append(np.sum(intfgs[shift_channel] ** 2)) # Again judge shifts based on the specified channel
    shift0 = shifts[np.argmax(sums)]

    print('Finally optimal shift:', shift0)
    best_delay = shift0
    x, intfgs = shifted_intfgs(shift0)

    intfgs_interp = interp1d(x=x,y=intfgs,
                            axis=-1,
                            **interp_kwargs)
    intfgs_new = intfgs_interp(xaxis)

    result = np.vstack((xaxis,intfgs_new))

    # Here we have changed the return signature compared to the `stable` version, so that time-values are first
    return result


#--- Wrapper
def align_interferograms_wrapper(intfgs_arr, delay_calibration_factor=1,
                                 shift0=0,optimize_shift=True,shift_range=15,
                                 flattening_order=20,noise=0):

    try:

        result = align_interferograms_new(intfgs_arr, delay_calibration_factor=delay_calibration_factor,
                                             shift0=shift0, optimize_shift=optimize_shift, shift_range=shift_range,
                                             flattening_order=flattening_order, noise=noise)

        #--- Dump the problematic interferograms
        error_file = os.path.join(diagnostic_dir,'align_interferograms_output.pickle')
        with open(error_file,'wb') as f:
            pickle.dump(result,f)

        #return result
        return result + np.zeros(result.shape)# + 0 #0*np.random.randn(*result.shape) # No idea why, but the latter is needed to prevent an "unknown python / labview error"

    except:

        #--- Dump the error
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'align_interferograms.err')
        with open(error_file,'w') as f: f.write(error_text)

        #--- Dump the problematic interferograms
        error_file = os.path.join(diagnostic_dir,'align_interferograms_input.pickle')
        with open(error_file,'wb') as f:
            pickle.dump(intfgs_arr,f)

        raise

def test_align_interferograms():
    # Result will be provided as global variable.

    global result

    try:
        file = os.path.join(diagnostic_dir, 'interferograms.txt')
        intfgs_arr = np.loadtxt(file)
        result = align_interferograms_new(intfgs_arr, delay_calibration_factor=-2.5,
                                          shift0=-10, optimize_shift=True, shift_range=10,
                                          flattening_order=10, noise=0)
        return str(1)

    except:
        # s=str(np.__version__)+','+str(scipy.__version__)
        s = str(np.__file__)
        s = str(traceback.format_exc())
        return s

def spectral_envelope(f,A,f0,df,expand_envelope=1):

    df = expand_envelope*df
    e = np.exp(-(f-f0)**2/(2*df**2))

    return A*e #maximum value will always be `A`

def fit_envelope(f,sabs):

    global result
    ind=np.argmax(sabs)
    f0 = f[ind]
    df = f0/4
    A=sabs[ind]
    # This is a trick to make sure if we have any frequency-halved "ghost" spectral parts, the highest frequency gaussian with amplitude `A2` wins
    f0_2 = 2*f0
    ind2 = np.argmin(np.abs(f-f0_2))
    A_2 = sabs[ind2]
    if A_2/A > .2: # What should this threshold be?  If it is too low, then 2*omega peak may be grabbed instead
        f0=f0_2
        A=A_2
        print('Picking higher peak!')
    x0=np.array([A,f0,df])

    def to_minimize(x):

        A,f0,df=x
        delta = sabs-spectral_envelope(f,A,f0,df,
                                          expand_envelope=1)
        return delta

    envelope_params = leastsq(to_minimize,x0,maxfev=100)[0]

    envelope = spectral_envelope(f,*envelope_params)
    #The overall value will not be meaningful, so be sure to normalize to 1 later, before applying

    return envelope,envelope_params

fmax = 3000 # in wavenumbers
def fourier_xform_new(intfgs_data,tsubtract=0,envelope=True,gain=1,fmin=500,fmax=fmax,window=True,convert_to_wn=True):

    intfgs_data = np.array(intfgs_data)

    t = intfgs_data[0]; Nts = len(t)
    v = intfgs_data[1:] # A 2D array of size (Nchannels,Nts)
    Nchannels = len(v)

    t=t-tsubtract #subtract a time correction in ps
    if convert_to_wn:
        c=3e8 #speed of light in m/s
        t*=1e-12 #ps in seconds
        d=t*c*2 #distance in m
        d*=1e2 #distance in cm
    else: d=t

    v = v-np.mean(v,axis=-1)[:,np.newaxis] # Subract any DC offset away from all channels
    v *= gain

    if window:
        w = np.blackman(Nts)[np.newaxis,:]
        # wmax_ind = np.argmax(w) #Let's roll window to position of maximum weight in interferogram
        # centerd = np.sum(d * v ** 2) / np.sum(v ** 2)
        # imax_ind = np.argmin((centerd - d) ** 2)
        # w = np.roll(w, imax_ind - wmax_ind, axis=0)
        # w = 1 #We are disabling the windowing for now
        pow0 = np.sum(v**2,axis=-1)
        v *= w
        pownew = np.sum(v**2,axis=-1)
        norm = np.sqrt(pow0/pownew)[:,np.newaxis]
        v*=norm # This normalization is being applied channel-wise

    # Roll signals so that d=0 is at the beginning (corresponds to a phase of zero)
    # Added: 2023.11.19
    # Removed: 2023.11.22, in favor of putting x0 phase info into num.Spectrum
    #Noff = np.argmin(np.abs(d))
    #v = np.roll(v,-Noff,axis=0)
    #d -= np.min(d) #Make all abscissa values positive

    global vshifted
    vshifted = v
    v = AWA(v, axes=[None,d], axis_names=['Channel','Time (ps)'])
    s = num.Spectrum(v, axis=-1) # First axis remains `channels` index, last index will be frequency
    f = s.axes[-1]

    posf=(f>0)
    if fmax is not None: posf*=(f<fmax)
    if fmin is not None: posf*=(f>fmin)
    phase=np.angle(s)[:, posf] # Preserve first axis as channels
    phase = (phase+np.pi)%(2*np.pi)-np.pi #This modulo's the phase to closest interval near 0
    sabs=np.abs(s)[:, posf]
    f=f[posf]

    if envelope: env,env_params = fit_envelope(f,sabs[0]) # Fit envelope to zeroth channel
    else: env=[1]*len(f); env_params=[0]
    Nf=len(f); Ne=len(env_params)
    if Ne<Nf: env_params=np.append(env_params,[0]*(Nf-Ne))

    assert len(f)==sabs.shape[-1]==phase.shape[-1]==len(env)

    # Stack up the result array
    result = np.vstack([f,env,env_params])
    for n in range(Nchannels):
        s = sabs[n]
        p = phase[n]
        result = np.vstack((result,s,p))

    # The output will now be:
    #   [frequencies,
    #    envelope,
    #    envelope parameters,
    #    spectrum absolute value (channel 1),
    #    spectrum phase (channel 1),
    #    spectrum absolute value (channel 2),
    #    ... etc. for all channels]
    return result

#--- Wrapper
def fourier_xform_wrapper(intfgs_data,tsubtract=0,envelope=True,gain=1,fmin=500,fmax=fmax,window=True,convert_to_wn=True):

    try:
        result = fourier_xform_new(intfgs_data,tsubtract=tsubtract,
                                   envelope=envelope,gain=gain,
                                   fmin=fmin,fmax=fmax,
                                   window=window,convert_to_wn=convert_to_wn)
        return result

    except:

        #--- Dump the error
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'fourier_xform.err')
        with open(error_file,'w') as f: f.write(error_text)

        #--- Dump the problematic interferograms
        error_file = os.path.join(diagnostic_dir,'fourier_xform_input.err.pickle')
        with open(error_file,'wb') as f:
            pickle.dump(intfgs_data,f)

        raise

class SpectralProcessor(object):

    ################################
    #- Frequency calibration methods
    ################################
    @classmethod
    def apply_frequency_calibration(cls,factor,f0,
                                      ss, ss_interp_ref,
                                      env_params,
                                    apply_envelope=True,expand_envelope=1):

        spectra_cal = []
        spectra_BB_cal = []
        for i in range(len(ss)):
            spectrum = ss[i]
            if apply_envelope:
                env = spectral_envelope(f0 * factor, *env_params[i],
                                        expand_envelope=expand_envelope)
                env /= env.max()
            else: env=1
            spectrum_ref = ss_interp_ref[i](f0 * factor)
            spectra_cal.append(env * spectrum)
            spectra_BB_cal.append(env * spectrum_ref)

        return spectra_cal, spectra_BB_cal

    @classmethod
    def calibration_to_minimize(cls,params,
                                f0, ss, ss_interp_ref, env_params,
                                apply_envelope=True,expand_envelope=1,
                                valid_thresh=.01):
        factor, = params

        spectra_cal, spectra_BB_cal = cls.apply_frequency_calibration(factor,f0,
                                                                    ss, ss_interp_ref, env_params,
                                                                    apply_envelope=apply_envelope,
                                                                    expand_envelope=expand_envelope)
        f, s_abs_norm = cls.normalize_spectrum(f0, spectra_cal,
                                               f0, spectra_BB_cal,
                                               valid_thresh=valid_thresh)
        d1 = np.gradient(s_abs_norm) / s_abs_norm

        return np.mean(np.abs(d1) ** (1 / 2))  # minimize first derivative

    ##################################
    # - Phase alignment methods
    ##################################
    @staticmethod
    def phase_displace(f, s, p):

        return np.exp(1j * f * p) * s

    @staticmethod
    def get_intfg(fs,scomplex):

        assert scomplex.any(),'Input must not be empty!'

        # Stack negative frequencies, because we don't have them
        interp_pos=interp1d(x=fs,y=scomplex,
                        axis=0, fill_value=0,
                        kind='linear', bounds_error=False) #types beyond 'linear' type could give problematic extrapolations
        interp_neg=interp1d(x=-fs[::-1],y=scomplex.conj()[::-1],
                            axis=0, fill_value=0,
                            kind='linear', bounds_error=False) #types beyond 'linear' type could give problematic extrapolations

        # Get desired frequency channels for properly interpolated FFT
        df = np.diff(fs)[0]
        Npts = 2*int(fs.max()/df)
        fs_fft = np.fft.fftfreq(Npts, d=1 / (2 * fs.max()) ) #Assert `fs` goes up to Nyquist, so that `d` is sampling interval

        # Generate proper FFT from interpolation
        s_fft_pos = interp_pos(fs_fft[fs_fft>=0])
        s_fft_neg = interp_neg(fs_fft[fs_fft<0])
        s_fft = np.append(s_fft_pos,s_fft_neg)

        intfg = np.fft.ifft(s_fft).real
        Dx = 1 / np.diff(fs)[0]
        xs = np.linspace(-Dx/2, Dx/2, len(intfg))
        intfg = np.roll(intfg,Npts//2,axis=0) #Roll forward by half, because our x=0 is in the middle, not at the beginning

        return xs,intfg

    @classmethod
    def level_phase(cls,f, s, order=1, manual_offset=0, return_leveler=False,
                    weighted=True, subtract_baseline=False):

        from scipy.linalg import LinAlgError
        assert np.all(np.isfinite(s)) and len(f)==len(s)

        if s.any() and weighted: w = np.abs(s) ** 2
        else: w=None

        leveler = np.ones(s.shape,dtype=complex)
        if order and s.any(): #If `s` is empty, we don't want to compute bogus interferogram
            #Begin by de-shifting interferogram
            x,intfg = cls.get_intfg(f,s)
            x0 = np.sum(x*intfg**2)/np.sum(intfg**2)
            cls.x0 = x0
            leveler *= np.exp(2*np.pi*1j*f*x0) #plane waves are in a basis `exp(-1j*x...)`, so we are shifting by `-x0`

            #Loop until we can flatten the phase no further
            attempts=0; max_attempts=10
            while True:
                p = np.unwrap(np.angle( s*leveler ))
                try: m = list(np.polyfit(x=f, y=p, deg=order, w=w))
                except LinAlgError: break #Many reasons why this might fail out
                if order == 1 and \
                    not subtract_baseline:  m[-1]=0 #no overall phase offsets, if we are restricting to physical offsets
                pcorr = -np.polyval(m, f)
                leveler *= np.exp(1j*pcorr) #Update leveler
                if np.isclose(np.sum(m[:-1]),0)\
                        or attempts==max_attempts: break #If we can update non-constant offsets no further, break
                attempts += 1

        if manual_offset:
            f0 = np.sum(np.abs(f)*np.abs(s)**2)/np.sum(np.abs(s)**2) # Spectrum-weighted average frequency
            pcorr = manual_offset * f/f0
            leveler *= np.exp(1j*pcorr)

        if return_leveler: return leveler
        else: return s*leveler

    @classmethod
    def get_phase(cls, f, s,
                  level=True,
                  **kwargs):

        if level:
            s=cls.level_phase(f,s,**kwargs)

        p = np.unwrap(np.angle(s))

        #Remove as many factors of 2*pi so that greatest intensity region is closest to zero
        if s.any():
            pavg = np.sum(p*np.abs(s)**2) / np.sum(np.abs(s)**2)
            p -= 2*np.pi * np.round(pavg/(2*np.pi))

        return p.real

    @classmethod
    def get_phases_in_sequence(cls, f, spectra,
                               threshold=1e-1,
                               level=True, order=6,
                               **kwargs):
        """" The purpose of this function is not to process spectra, but rather to gather
        the phases of spectra for display together in a sequence.

        The idea is partly to see how the phases "line up" from individual spectra.
        To produce a sequence of phases where this can be evaluated, this function
         figures out a global curve that levels the combined phase spectrum.  It
         propagates this leveling curve to all the individual spectra.  It subtracts
         a suitable number of `2*pi`s such that each individual spectrum lies somewhere
         on the global curve.

         So, the function is merely cosmetic in that it applies
         1) a global background correction, and
         2) piecewise factors of 2*pi."""

        # Get the zone of validity for all spectra
        where_goods = [ (np.abs(s) > np.abs(s).max() * threshold) * np.isfinite(s) \
                        for s in spectra ]
        where_all_good = np.sum(where_goods,axis=0).astype(bool) # This is an "or"

        # Get global leveler
        stot = np.sum(spectra,axis=0)
        x,intfg = cls.get_intfg(f,stot)
        x0 = np.sum(x*intfg**2)/np.sum(intfg**2)
        global_leveler = np.exp(2*np.pi*1j*f*x0)

        # Get phase backbone, with extra leveler, unwrapped in the relevant energy range
        phase_backbone = cls.get_phase(f, stot*global_leveler,
                                       manual_offset=0, level=True,order=order) # Don't level yet, we want raw phase
        global_leveler *= cls.level_phase(f,stot*global_leveler,manual_offset=0, order=order, return_leveler=True) # update global leveler with further leveling that gave us this phase backbone
        phase_backbone[where_all_good] = np.unwrap(phase_backbone[where_all_good])
        phase_backbone[~where_all_good] = np.nan

        #return [phase_backbone]

        # Go through spectra and add as many factors of 2*pi as necessary to match backbone
        phases = []
        for i,(s,where_good) in enumerate(zip(spectra,where_goods)):
            phase = cls.get_phase(f, s*global_leveler, manual_offset=0,
                                  level=False,
                                  **kwargs)  # only leveling will be that from the global leveler, then offsets by 2*pi
            phase[~where_good] = np.nan
            #phase[where_good] = np.unwrap(phase[where_good])

            # Weighted average phase in this spectrum range
            pavg = np.sum(phase[where_good]*np.abs(s[where_good])**2) \
                    / np.sum(np.abs(s[where_good])**2)

            # Weighted average backbone phase in this spectrum range
            pavg_backbone = np.sum(phase_backbone[where_good] * np.abs(stot[where_good]) ** 2) \
                             / np.sum(np.abs(stot[where_good]) ** 2)

            # Compare the two and subtract the number of pis by which we differ from backbone
            n2pis = np.round( (pavg - pavg_backbone) / (2*np.pi) )
            if np.isnan(pavg) or np.isnan(pavg_backbone): raise ValueError
            phases.append(phase - 2*np.pi*n2pis)

        return phases

    @classmethod
    def get_phases_old(cls,f,spectra,
                      threshold=1e-2,
                      ps=None,
                      level=True,order=2,
                      **kwargs):
        """Return the phases corresponding to `spectra`, shifting each by an
        appropriate factor of `2*pi` so that they are quasi-contiguous.
        Spectra are assumed to have monotonically increasing center frequencies.

        Set `threshold` to decide the noise threshold above which spectral phase is computed.

        Choose `level=True` to apply an overall polynomial leveling of order `order`
        to all spectra together."""

        phases=[]
        for i,s in enumerate(spectra):
            if ps is not None:
                s=cls.phase_displace(f,s,ps[i])
            phase=cls.get_phase(f, s, manual_offset=0,
                                level=False, **kwargs) #level phase has to be false, we want the overall phase!
            thresh=np.abs(s).max()*threshold
            phase[np.abs(s) <= thresh] = np.nan
            phases.append(phase)

        for i in range(len(phases)-1):
            phase0=phases[i]
            phase=phases[i+1]
            coincide = np.isfinite(phase0)*np.isfinite(phase)
            if not coincide.any(): break
            d = np.mean(phase0[coincide]-phase[coincide])
            npi = np.round(d/(2*np.pi))
            offset = npi * 2*np.pi
            #offset = d
            phase += offset

        if not level: return phases

        attempts = 0
        max_attempts = 10
        while attempts<max_attempts:
            phase_avg = np.nanmean(phases, axis=0)
            keep = np.isfinite(phase_avg)

            poly = np.polyfit(f[keep],np.unwrap(phase_avg[keep]),deg=order)
            background = np.polyval(poly,f)
            for i in range(len(phases)): phases[i]-=background

            if np.sum(background) == 0: break
            attempts +=1

        return phases

    @classmethod
    def interpolate_spectrum(cls,s,f1,f2,order=1):
        #Regular interpolation of a spectrum, but first removing overall phase slope, which can be smeared by interpolation

        if np.all(f1==f2): return s #No need for interpolation if frequencies are identical

        # Find overall phase slope
        leveler = cls.level_phase(f1,s,order=order,manual_offset=0,
                                  subtract_baseline=True,return_leveler=True)

        result = interp1d(x=f1,y=s*leveler, **interp_kwargs)(f2)

        leveler = interp1d(x=f1,y=leveler,**interp_kwargs)(f2)

        result /= leveler #re-impose overall phase slope

        #print(len(f2))
        result[~np.isfinite(result)]=0
        #print(len(result))

        return result

    @staticmethod
    def summed_spectrum(spectra, abs=False):

        if abs:
            spectrum = np.sum([np.abs(s) for s in spectra], axis=0)
            spectrum = spectrum.real
        else:
            spectrum = np.sum([s for s in spectra], axis=0)

        spectrum[~np.isfinite(spectrum)] = 0

        return spectrum

    @classmethod
    def impose_threshold(cls,f_mutual,spectrum,spectrum_ref,valid_thresh=.01):

        use_previous = (valid_thresh == 'last') and hasattr(cls,'thresholded_range')
        if use_previous and len(cls.thresholded_range) == len(f_mutual):
            thresholded_range = cls.thresholded_range
        else:
            thresh = valid_thresh * np.abs(spectrum_ref[np.isfinite(spectrum_ref)]).max()
            thresholded_range = np.abs(spectrum_ref) >= thresh
            # We do not want to threshold based and sample spectrum values - presume that even zeros are valid data
            #thresh = valid_thresh * np.abs(spectrum[np.isfinite(spectrum)]).max()
            #thresholded_range *= np.abs(spectrum) >= thresh
            cls.thresholded_range = thresholded_range

        f_mutual = f_mutual[thresholded_range]
        spectrum = spectrum[thresholded_range]
        spectrum_ref = spectrum_ref[thresholded_range]

        return f_mutual, spectrum, spectrum_ref

    @classmethod
    def normalize_spectrum(cls, f_s, spectrum,
                           f_r, spectrum_ref,
                           valid_thresh=.01):

        f_mutual = np.unique(np.append(f_s,f_r)) #Join and sort the frequencies
        Navg = np.mean((len(f_s),len(f_r)))
        decimation = int(np.round(len(f_mutual) / Navg))
        if decimation>1:
            from scipy.signal import decimate
            print('Normalization: decimating by %i to common frequency axis'%decimation)
            f_mutual0 = f_mutual
            f_mutual = decimate(f_mutual,q=decimation,n=1) #down-sample by a factor of 2
            print(len(f_mutual0),'-->',len(f_mutual))

        spectrum = cls.interpolate_spectrum(spectrum,f_s,f_mutual,order=4)
        spectrum_ref = cls.interpolate_spectrum(spectrum_ref,f_r,f_mutual,order=4)

        #Limit output to range where reference & input spectra have weight above threshold
        f_mutual,spectrum,spectrum_ref = cls.impose_threshold(f_mutual,spectrum,spectrum_ref,
                                                              valid_thresh=valid_thresh)
        snorm = (spectrum / spectrum_ref)

        return f_mutual, snorm

    window_order=1

    # This is deprecated, no obvious way to make it both fast and reliable
    @classmethod
    def windowed_spectrum(cls,f,scomplex,window=np.blackman,window_order=None):

        if window is True:
            window = np.blackman

        if window_order is None:
            window_order = cls.window_order

        x,intfg = cls.get_intfg(f,scomplex)

        w = window(len(intfg))**window_order
        # Do not permit variability of window due to uncertainty in peak intensity
        xcenter = np.sum(intfg ** 2 * x) / np.sum(intfg ** 2)
        w = np.roll(w, np.argmin(x - xcenter) - np.argmax(w), axis=0) # Roll window center to intfg maximum"""
        w=1
        intfgw = intfg * w

        ## Convert back to spectrum
        sout = num.Spectrum(AWA(intfgw,axes=[x]), axis=0)
        sout = sout.interpolate_axis(f, axis=0, **interp_kwargs)

        # Do not let window change overall power
        Pactual = np.sum(np.abs(sout)**2)
        Pdesired = np.sum(np.abs(scomplex)**2)
        sout *= np.sqrt( Pdesired/Pactual )

        return np.array(sout,dtype=complex)

    @classmethod
    def smoothed_spectrum(cls,f,scomplex,smoothing=4,order=2):

        leveler = cls.level_phase(f,scomplex,order=order,return_leveler=True,
                                  weighted=True,subtract_baseline=True)

        scomplexl = scomplex*leveler

        scomplexl = numrec.smooth(scomplexl,window_len=smoothing,axis=0)

        return scomplexl / leveler

    def phase_aligned_spectrum(self, f, spectra,
                               phase_alignment_exponent=None):

        if phase_alignment_exponent is None: phase_alignment_exponent=self.phase_alignment_exponent
        #self.s0s=[] #Debugging - check the operands for phase alignments

        def aligned_spectrum(s0, s, leveler0=1):

            #Determine and remove average phase of stot, since it could be wrapping like crazy
            #leveler = self.level_phase(f, s0 , order=1, manual_offset=0, return_leveler=True,
            #                           weighted=True,subtract_baseline=True)

            # Debugging - check the leveling
            #self.s0s.append(num.Spectrum(AWA(s0 * leveler, axes=[f], axis_names=['X Frequency']), axis=0))

            phase0 = self.get_phase(f, s0, manual_offset=0,
                                    level=False)
            phase = self.get_phase(f, s * leveler0, manual_offset=0,
                                   level=False) #make a guess that we should level new phase to same as previous
            coincidence = (np.abs(s0) * np.abs(s)) ** phase_alignment_exponent
            coincidence /= coincidence.max() #Just to be a standardized distribution

            if not coincidence.any(): return s,0

            # 2023.11.24: Notebook entitled "PRISM SPECTROSCOPY DIAGNOSTICS.IPYNB" demonstrates that
            # new method with offset is superior by 2x compared to old method
            use_new=True
            if use_new:  #This doesn't work very well so far, optimization overshoots?
                S = s*leveler0
                def residual(args):
                    delta, p0 = args
                    S_aligned = self.phase_displace(f,S,delta) # p is phase per unit frequency
                    #S_aligned *= np.exp(1j*p0)
                    return np.abs(s0-S_aligned)*coincidence
                from scipy.optimize import leastsq
                delta,p0 = leastsq(residual,(0,0),factor=1e-3)[0]
                #p0=0

            else:
                norm = np.sum(coincidence)
                fcenter = np.sum(f * coincidence) / norm
                pdiff = np.sum((phase0 - phase) * coincidence) / norm
                #pdiff -= np.round(pdiff/(2*np.pi))*2*np.pi #Add as many factors of 2*pi as minimizes `pdiff`

                # We are assuming that a physical shift of the spectrum of order 2*pi is unlikely,
                # otherwise, any attempt match adjacent phase spectra will be hopeless anyway!

                # This is by how much we need to roll next spectral phase forwards
                delta = pdiff/fcenter
                p0=0

            leveler = leveler0 * np.exp(1j * f * delta) * np.exp(1j*p0)
            spectrum_aligned = s * leveler

            return spectrum_aligned, leveler

        #Find the spectrum with greatest weight
        weights = [np.sum(np.abs(spectrum)**2)
                   for spectrum in spectra]
        ind0 = np.argmax(weights)
        print('Primary index for phase alignment:',ind0)

        self.phase_levelers= [1] * len(spectra)
        spectra_aligned=[0]*len(spectra)
        spectra_aligned[ind0] = spectrum_cmp = spectra[ind0]
        total_count=1

        #Count backward, then forward
        index_list = np.append( np.arange(ind0-1,-1,-1),
                                np.arange(ind0+1,len(spectra),1) )
        latest_leveler = 1
        for i in index_list:
            spectrum_aligned, latest_leveler = aligned_spectrum(spectrum_cmp,
                                                                     spectra[i],
                                                                     latest_leveler)
            spectra_aligned[i] = spectrum_aligned
            self.phase_levelers[i] = latest_leveler
            if i==ind0+1: latest_leveler = 1 #Re-set if we're now comparing to `ind0`
            total_count+=1

            #When we hit the first, re-set our reference spectrum
            if i==0: spectrum_cmp=spectra[ind0]
            else: spectrum_cmp = spectrum_aligned

        assert total_count==len(spectra)

        return spectra_aligned

    Nrows_env = 2
    N_env_params = 3 # number of parameters to reconstruct the envelope

    @classmethod
    def align_and_envelope_spectra(cls, spectra, spectra_BB,
                                   channel = 1, channel_BB = 1,
                                   apply_envelope=True, envelope_width=1,
                                   BB_phase=False,
                                   smoothing=None, window=np.blackman,
                                   **kwargs):

        spectra = np.array(spectra)
        spectra_BB = np.array(spectra_BB)

        # We expect both `spectra` and `spectra_BB` to have shape:
        # (Naccumulations, 1+Nrows_env + Nchannels, Nfreqs)

        assert len(spectra) == len(spectra_BB), \
            "Applying envelope requires same number of accumulations for both analyte and reference"
        Naccumulations = spectra.shape[0]

        Nchannels = (spectra.shape[1] - (1 + cls.Nrows_env))//2
        assert 1 <= channel and channel <= Nchannels, \
            '`channel`="%i" should be between 1 and %i.'%(channel,Nchannels)

        Nchannels_BB = (spectra_BB.shape[1] - (1 + cls.Nrows_env))//2
        assert 1 <= channel_BB and channel_BB <= Nchannels_BB, \
            '`channel_BB`="%i" should be between 1 and %i.'%(channel_BB,Nchannels_BB)

        # Establish some mutual frequency axis
        f0 = None

        # Collect all the spectra and the envelopes
        ss = []
        ss_BB = []
        env_sets = []
        for n_accum in range(Naccumulations):

            ##----- Reference spectrum components
            f_BB,env,env_set = spectra_BB[n_accum,:1 + cls.Nrows_env] # we will only pay attention to first channel of BB
            channel_BB_ind = 1 + cls.Nrows_env + 2 * (channel_BB - 1)
            sabs_BB, sphase_BB = spectra_BB[n_accum,channel_BB_ind:channel_BB_ind+2]
            env_set = env_set[:cls.N_env_params]
            s_BB = sabs_BB * np.exp(1j * sphase_BB)

            # In case we have invalid entries (zeros), remove them
            where_valid = np.isfinite(f_BB) * (f_BB > 0)
            if not where_valid.any():
                print('Bright-beam spectrum at accumulation number %i is inferred empty!' % (n_accum+1) )
                continue
            f_BB = f_BB[where_valid]
            s_BB = s_BB[where_valid]

            # Establish some mutual frequency axis
            if f0 is None: f0=f_BB #In this way, even if analyte spectra are empty, we have a definite frequency axis

            # Compute envelope
            if apply_envelope:
                env = spectral_envelope(f0, *env_set,
                                        expand_envelope = envelope_width)
                env /= env.max()

            # Touch up the spectrum in all the ways
            # Smooth before applying envelope
            if window:
                s_BB = cls.windowed_spectrum(f_BB, s_BB, window=window)
            if smoothing:
                s_BB = cls.smoothed_spectrum(f_BB, s_BB, smoothing=smoothing)
            s_BB = cls.interpolate_spectrum(s_BB, f_BB, f0)
            if apply_envelope: s_BB *= env

            # Discard phase if we don't want it
            if not BB_phase:
                s_BB = np.abs(s_BB)

            ##-------- Analyte spectrum components
            # We will ignore the 'envelope' parts, these are taken from BB
            f = spectra[n_accum][0] # First row at this accumulation is the frequencies row
            channel_ind = 1 + cls.Nrows_env + 2 * (channel - 1)
            sabs, sphase = spectra[n_accum,channel_ind:channel_ind+2]
            s = sabs * np.exp(1j * sphase)

            # In case we have invalid entries (zeros), remove them
            where_valid = np.isfinite(f) * (f > 0)
            if not where_valid.any():
                print('Analyte spectrum at accumulation number %i is inferred empty!' % (n_accum+1) )
                continue
            f = f[where_valid]
            s = s[where_valid]

            # Touch up the spectrum in all the ways
            # Smooth before applying envelope
            if window:
                s = cls.windowed_spectrum(f, s, window=window)
            if smoothing:
                s = cls.smoothed_spectrum(f, s, smoothing=smoothing)
            s = cls.interpolate_spectrum(s, f, f0)
            if apply_envelope: s *= env

            # We're done
            env_sets.append(env_set)
            ss.append(s)
            ss_BB.append(s_BB)

        # If we didn't process any reference spectra (at least) then we have an error
        if f0 is None: raise ValueError('All spectral data were supplied empty!')

        # Sort by center frequency
        fcenters = np.array([env_set[1] for env_set in env_sets])
        ordering = np.argsort(fcenters)
        ss = np.array( [ss[idx] for idx in ordering] )
        ss_BB = np.array( [ss_BB[idx] for idx in ordering] )

        # Add a global leveler.  It's a stand-in, maybe we'll do something with it later.
        cls.global_leveler = np.ones(f0.shape)

        #Spectra could all be empty, though f0 will not be
        return f0, ss, ss_BB

    @classmethod
    def accumulate_spectra(cls, spectra,
                           channel=1,
                           apply_envelope=True,
                           expand_envelope=1,
                           **kwargs):

        # Piggyback on top of `align_and_envelope_spectra` to get the aligned spectra.
        # The BB is only used to get envelope.  So just get envelope from same spectra, if we use it.
        f0, ss, ss_BB = cls.align_and_envelope_spectra(spectra, spectra_BB=spectra,
                                                       channel = channel, channel_BB = channel,
                                                       apply_envelope=apply_envelope, envelope_width=expand_envelope,
                                                       BB_phase=False,
                                                       smoothing=None, window=np.blackman,
                                                       **kwargs)

        spectrum_abs = cls.summed_spectrum(ss, abs=True)
        spectrum_phase = cls.get_phase(f0,
                                       cls.summed_spectrum(ss, abs=False),
                                       level=False)  # We could add option for leveling phase

        return np.array([f0, spectrum_abs, spectrum_phase], dtype=float)

    ###########
    #- User API
    ###########

    def __init__(self,
                 sample_spectra,
                 sample_BB_spectra,
                 ref_spectra,
                 ref_BB_spectra,
                 phase_alignment_exponent=2):

        self.phase_alignment_exponent=phase_alignment_exponent

        sample_spectra = np.array(sample_spectra)
        assert sample_spectra.ndim == 3, \
            "`sample_spectra` should be an array with three dimensions: (N accumulations, %i + N channels, N frequencies"%(1+self.Nrows_env)
        assert len(sample_spectra) == len(sample_BB_spectra), \
            "`sample_spectra` and `sample_BB_spectra` should have the same number of accumulations."

        ref_spectra = np.array(ref_spectra)
        assert ref_spectra.ndim == 3, \
            "`ref_spectra` should be an array with three dimensions: (N accumulations, %i + N channels, N frequencies"%(1+self.Nrows_env)
        assert len(ref_spectra) == len(ref_BB_spectra), \
            "`ref_spectra` and `ref_BB_spectra` should have the same number of accumulations."

        #assert sample_spectra.shape[1] == ref_spectra.shape[1], \
        #    "`sample_spectra` and `ref_spectra` must have the same number of channels!"

        self.sample_spectra = np.array(sample_spectra)
        self.sample_BB_spectra = np.array(sample_BB_spectra)
        self.ref_spectra = np.array(ref_spectra)
        self.ref_BB_spectra = np.array(ref_BB_spectra)

    def select_spectrum(self,idx,target='sample',window=None):
        #Spectra should share same frequency axes, but it is not guaranteed,
        # so interpolate them to same frquency axis

        options = ['sample','sample_BB','reference','reference_BB']
        assert target in options,'`target` must be one of %s'%options

        if target==options[0]: spectra=self.sample_spectra
        elif target==options[1]: spectra=self.sample_BB_spectra
        elif target==options[2]: spectra=self.ref_spectra
        elif target==options[3]: spectra=self.ref_spectra_BB

        f,s,p = spectra[self.Nrows*idx : self.Nrows*idx+3]
        s = s*np.exp(1j*p)
        if window: s = self.windowed_spectrum(f, s, window=window)

        s = AWA(s,axes=[f],axis_names=['X Frequency'])
        return num.Spectrum(s,axis=0)

    def process_spectrum(self, target='sample',
                         channel=1, channel_BB=1,
                         apply_envelope=True,
                         envelope_width=1,
                         valid_thresh=.01,
                         smoothing=None,
                         window=None,
                         align_phase=True,
                         BB_normalize=False,
                         BB_phase=False,
                         view_phase_alignment=False,
                         view_phase_alignment_leveling=6):
        """This normalizes sample or reference spectra (based on `target`)
        to their respective bright beam spectra, while enveloping,
        optimizing phase alignment, and windowing spectra, if desired."""

        assert target=='sample' or target=='reference'
        if target=='sample':
            all_spectra = self.sample_spectra
            all_spectra_BB = self.sample_BB_spectra
        else:
            all_spectra = self.ref_spectra
            all_spectra_BB = self.ref_BB_spectra

        print('Aligning and enveloping spectra...')
        f0, spectra, spectra_BB = self.align_and_envelope_spectra(all_spectra, all_spectra_BB,
                                                                  channel=channel, channel_BB=channel_BB,
                                                                  apply_envelope=apply_envelope,
                                                                  envelope_width=envelope_width,
                                                                  smoothing=smoothing,window=window,
                                                                  BB_phase=BB_phase,
                                                                  valid_thresh=valid_thresh)

        # Apply correction factor (as from detector calibration) if provided
        if hasattr(self,'%s_factors'%target):
            print('ya')
            factors=getattr(self,'%s_factors'%target)
            for s,factor in zip(spectra,factors): s *= factor

        # Store the processed spectra and their attributes
        if target == 'sample':
            self.sample_spectra_processed = spectra
            self.sample_BB_spectra_processed = spectra_BB
            self.sample_frequencies = f0
            self.sample_leveler = self.global_leveler
        else:
            self.ref_spectra_processed = spectra
            self.ref_BB_spectra_processed = spectra_BB
            self.ref_frequencies = f0
            self.ref_leveler = self.global_leveler

        if not len(spectra):
            print('Analyte spectra were empty of data!  Returning spectra of zeros...')
            s_empty = np.zeros(f0.shape,dtype=float)
            return f0,s_empty,s_empty

        # Align phase only if commanded, store uncorrected version as `spectra0` for comparison
        if align_phase:
            print('Aligning phase...')
            spectra0 = spectra
            spectra = self.phase_aligned_spectrum(f0, spectra0)

            if view_phase_alignment:
                from matplotlib import pyplot as plt
                plt.figure(figsize=(10,10))

                # Plot the spectral power recovered by phase correction
                plt.subplot(211)
                a = self.summed_spectrum(spectra0, abs=True)
                b = np.abs(self.summed_spectrum(spectra0, abs=False))
                c = np.abs(self.summed_spectrum(spectra, abs=False))
                plt.plot(f0, a, label='Phase discarded')
                plt.plot(f0, b, label='Phase unaligned')
                plt.plot(f0, c, label='Phase aligned')
                plt.ylabel('Power spectral density')
                plt.xlabel('Frequency')
                plt.legend(loc='best')
                plt.twinx()
                plt.plot(f0, a - c, color='r')
                plt.ylabel('Spectral density lost to phase',
                           rotation=270,labelpad=20,color='r')
                plt.title('Target = %s'%target)

                # Now plot directly the corrected phases
                plt.subplot(212)
                phases0 = self.get_phases_in_sequence(f0, spectra0,
                                                      threshold=.1,
                                                      ps=None,
                                                      level=True, order=view_phase_alignment_leveling)
                phases = self.get_phases_in_sequence(f0, spectra,
                                                     threshold=.1,
                                                     ps=None,
                                                     level=True, order=view_phase_alignment_leveling)

                label0='Uncorrected'; label='Corrected'
                for i in range(len(phases0)):
                    l,=plt.plot(f0,phases0[i],ls='--',label=label0)
                    plt.plot(f0,phases[i],color=l.get_color(),label=label)
                    label0=label=None
                plt.xlabel('Frequency')
                plt.ylabel('Phase ')
                plt.legend(loc='best')
                plt.title('Target = %s'%target)

                #plt.subplots_adjust(hspace=.3)
                plt.tight_layout()

        if BB_normalize:
            spectrum_summed = self.summed_spectrum(spectra, abs=False)
            spectrum_BB_summed = self.summed_spectrum(spectra_BB, abs=False)

            spectrum_abs_summed = self.summed_spectrum(spectra, abs=True)
            spectrum_BB_abs_summed = self.summed_spectrum(spectra_BB, abs=True)

            # BB phase will already be discarded unless earlier `BB_phase=True`
            f, spectrum_abs = self.normalize_spectrum(f0, spectrum_abs_summed,
                                                      f0, spectrum_BB_abs_summed,
                                                        valid_thresh=valid_thresh)
            f, spectrum = self.normalize_spectrum(f0, spectrum_summed,
                                                  f0, spectrum_BB_summed,
                                                  valid_thresh='last') #Here we will use the previous threshold to get identical frequency channels
        else:
            spectrum_abs = self.summed_spectrum(spectra, abs=True)
            spectrum = self.summed_spectrum(spectra, abs=False)
            f=f0

        return f,spectrum_abs,spectrum

    # @TODO:  The API of this spectral processor now needs to be invoked at higher level
    #           with three additional arguments to support channels:
    #           channel=..., channel_ref=..., channel_BB=...
    def __call__(self,
                 channel=1, channel_ref=1, channel_BB=1,
                 optimize_BB=False,
                 align_phase=False,
                 apply_envelope=True,
                 envelope_width=1,
                 valid_thresh=.01,
                 smoothing=None,window=None,
                 view_phase_alignment=False,
                 view_phase_alignment_leveling=6,
                 BB_normalize=False,BB_phase=False,
                 recompute_reference=True,
                 **kwargs):
        """This normalizes sample and reference spectra to their
        respective bright-beam spectra, and then normalizes these
        two BB-normalized spectra to each other."""

        #--- First get reference spectrum
        #See if we can get away with not recomputing reference spectrum
        try:
            if recompute_reference: raise AttributeError
            self.f_ref,self.ref_spectrum_abs,self.ref_spectrum
        except AttributeError:
            print('Processing reference spectra...')
            self.f_ref,self.ref_spectrum_abs,self.ref_spectrum\
                                            = self.process_spectrum(target='reference',
                                                                    channel=channel_ref, channel_BB=channel_BB,
                                                                    apply_envelope=apply_envelope,
                                                                    envelope_width=envelope_width,
                                                                    valid_thresh=valid_thresh,
                                                                    smoothing=None,window=window,
                                                                    align_phase=align_phase,
                                                                    view_phase_alignment=view_phase_alignment,
                                                                    view_phase_alignment_leveling=view_phase_alignment_leveling,
                                                                    BB_normalize=BB_normalize,BB_phase=BB_phase,
                                                                    **kwargs)

            # `process_spectrum` can return zeros, if provided spectra were empty of data.
            # That is not allowed to happen here.
            assert self.ref_spectrum_abs.any(), 'Reference spectra must be non-empty!'

            #Smooth magnitude spectrum before normalization
            if smoothing and smoothing>1:
                self.ref_spectrum_abs = self.smoothed_spectrum(self.f_ref,self.ref_spectrum_abs,
                                                               smoothing,order=1) #magnitude spectrum, no leveling required

        #--- Now get sample spectrum
        print('Processing sample spectra...')
        self.f_sample,self.sample_spectrum_abs,self.sample_spectrum\
                                            =self.process_spectrum(target='sample',
                                                                   channel=channel, channel_BB=channel_BB,
                                                                   apply_envelope=apply_envelope,
                                                                   envelope_width=envelope_width,
                                                                   valid_thresh=valid_thresh,
                                                                   smoothing=None,window=window,
                                                                   align_phase=align_phase,
                                                                   view_phase_alignment=view_phase_alignment,
                                                                   view_phase_alignment_leveling=view_phase_alignment_leveling,
                                                                   BB_normalize=BB_normalize,BB_phase=BB_phase,
                                                                   **kwargs)

        #If sample spectrum is empty, fill it with zeros so normalization won't break
        if not self.sample_spectrum_abs.any():
            self.f_sample = self.f_ref
            self.sample_spectrum_abs = np.zeros(self.f_ref.shape)
            self.sample_spectrum = np.zeros(self.f_ref.shape)

        # Smooth magnitude spectrum before normalization
        if smoothing and smoothing > 1:
            self.sample_spectrum_abs = self.smoothed_spectrum(self.f_sample, self.sample_spectrum_abs,
                                                              smoothing,
                                                              order=1)  # magnitude spectrum, no leveling required

        self.f_norm_abs, self.norm_spectrum_abs = self.normalize_spectrum(self.f_sample, self.sample_spectrum_abs,
                                                                          self.f_ref, self.ref_spectrum_abs,
                                                                          valid_thresh=valid_thresh)
        self.f_norm, self.norm_spectrum = self.normalize_spectrum(self.f_sample, self.sample_spectrum,
                                                                  self.f_ref, self.ref_spectrum,
                                                                  valid_thresh='last') #re-use last thresholding

        self.norm_spectrum_abs = self.interpolate_spectrum(self.norm_spectrum_abs,
                                                           self.f_norm_abs, self.f_norm).real

        # Smooth complex spectrum only after normalization
        if smoothing and smoothing > 1:
            self.norm_spectrum = self.smoothed_spectrum(self.f_norm, self.norm_spectrum,
                                                        smoothing,
                                                        order=4)  # High-order leveling (then de-leveling) required for accurate smoothing

        return self.f_norm, self.norm_spectrum_abs, self.norm_spectrum


def accumulate_spectra(spectra, apply_envelope=True, expand_envelope=1, channel=1):

    try:
        return SpectralProcessor.accumulate_spectra(spectra,
                                                    channel=channel,
                                                    apply_envelope=apply_envelope,
                                                    expand_envelope=expand_envelope)

    except:
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'accumulate_spectra.err')
        with open(error_file,'w') as f:
            f.write(error_text)

        raise

def average_spectrum_block(specblock1,specblock2):

    #unpack spectrum blocks according to our known organization
    Nchann1 = (len(specblock1)-3)//2
    f1,e1,eparams1=specblock1[0:3]
    s1s = [specblock1[3+2*n]*np.exp(1j*specblock1[3+2*n+1])
           for n in range(Nchann1)]

    Nchann2 = (len(specblock2)-3)//2
    assert Nchann1 == Nchann2
    f2,e2,eparams1=specblock2[0:3]
    s2s = [specblock2[3+2*n]*np.exp(1j*specblock2[3+2*n+1])
           for n in range(Nchann2)]

    # In the oddball case that frequency axes are different, interpolate the former to the latter
    f=f2
    if not np.all(f1==f):
        s1s = [SpectralProcessor.interpolate_spectrum(s1, f1, f) for s1 in s1s]
        e1 = SpectralProcessor.interpolate_spectrum(e1, f1, f)
        eparams1 = eparams1[eparams1!=0]
        eparams1 = np.append(eparams1, [0]*(len(f)-len(eparams1))) #make sure envelope parameters have right size for stacking

    # Make arrays
    s1s = np.array(s1s)
    s2s = np.array(s2s)

    # Get average spectral amplitude
    sabsavgs = np.sqrt(np.abs(s1s) ** 2 + np.abs(s2s) ** 2) / np.sqrt(2)

    # Get and apply levelers, to get constituent phases
    leveling_chann = 0 # We can't use all channels so use zeroth
    s1 = s1s[leveling_chann]
    leveler1 = SpectralProcessor.level_phase(f, s1 * e1, order=1, manual_offset=0, return_leveler=True,
                                             weighted=True, subtract_baseline=True) #consider the enveloped spectrum when determining how to level phase
    s1s_leveled = s1s * leveler1[np.newaxis,:]

    s2 = s2s[leveling_chann]
    leveler2 = SpectralProcessor.level_phase(f, s2 * e2, order=1, manual_offset=0, return_leveler=True,
                                             weighted=True, subtract_baseline=True)
    s2s_leveled = s2s * leveler2[np.newaxis,:]

    # Get average phase
    pavgs = np.angle(s1s_leveled+s2s_leveled)

    # Determine what "in-between" leveler should be reversed
    p1lev = np.unwrap(np.angle(leveler1))
    p2lev = np.unwrap(np.angle(leveler2))
    W1 = np.sum(np.abs(s1))
    W2 = np.sum(np.abs(s2))
    p3lev = (W1 * p1lev + W2 * p2lev) / (W1+W2)
    pavgs -= p3lev[np.newaxis,:] # subtract leveler phase

    result = np.vstack( (f,e1,eparams1) )
    for n in range(Nchann1):
        sabsavg = sabsavgs[n]
        pavg = pavgs[n]
        result = np.vstack( (result,sabsavg,pavg) )

    return result

def heal_linescan(linescan):
    # We must assume our incoming data is shaped like:
    # (Npoints,Naccum,Nchann+..,Nfreqs)

    print('Healing linescan!')

    linescan = np.array(linescan,dtype=float)

    isempty = lambda arr: np.isclose(arr,0).all()

    import copy
    healed_linescan = copy.copy(linescan)

    # Loop over pixels
    for i in range(len(healed_linescan)):

        # We need nonzero data from 'nextnext' and preferably also 'next' pixels
        # So we cannot conceivably heal the last two pixels of data, sadly!
        if i <= len(healed_linescan) - 3:
            Nspectra = len(healed_linescan[i])

            for j in range(Nspectra): #Iterate over accumulations at this pixel
                shere = healed_linescan[i,j]

                if isempty(shere):
                    print('Spectrum at pixel %i, accumulation %i is empty!'%(i,j))
                    snext = healed_linescan[i + 1,j]
                    snextnext = healed_linescan[i + 2,j]

                    # Fill hole with data from next pixel, and fill next pixel with `next-nextnext` average
                    if not isempty(snext):
                        print("We're taking data from pixel %i!"%(i+1))
                        healed_linescan[i,j] = snext  # Assume data for `here` landed at next pixel
                        if not isempty(snextnext):
                            print("And we're filling in pixel %i with an average of %i and %i!"%(i+1,i+1,i+2))
                            # We are sending two spectra to `average_spectrum_block`, each are multi-channel spectra (2D arrays)
                            healed_linescan[i + 1,j] = average_spectrum_block(snext,snextnext) # Fill (false) next pixel spectrum with an average
                            #healed_linescan[i + 1][j1:j2] = (snext + snextnext) / 2  # Fill (false) next pixel spectrum with an average
                    elif not isempty(snextnext):
                        print("We're taking data from pixel %i!"%(i+2))
                        healed_linescan[i,j] = snextnext  # Assume data for `here` landed two pixels away

    return np.array(healed_linescan,dtype=float)

def heal_linescan_recursive(linescan):

    print('Recursively healing linescan!')

    linescan=np.array(linescan,dtype=float)

    linescan_healed = linescan
    linescan_healed_prev = np.zeros(linescan.shape)
    iteration=1
    while not np.isclose(linescan_healed, linescan_healed_prev).all():
        print('Healing iteration %i...'%iteration)
        linescan_healed_prev = linescan_healed
        linescan_healed = heal_linescan(linescan_healed_prev)
        iteration+=1

    return np.array(linescan_healed,dtype=float)

def BB_referenced_spectrum(spectra,spectra_BB,
                          apply_envelope=True, envelope_width=1,
                           smoothing=None,align_phase=False,
                           valid_thresh=.01,
                             channel=1, channel_BB=1):

    try:
        SP=SpectralProcessor(spectra, spectra_BB,
                             spectra_BB, spectra_BB)

        # `BB_normalize=False` is key, to compute only the envelope from BB,
        # and then to normalize spectra directly to BB as though it were reference
        f, spectrum_abs, spectrum = SP(apply_envelope=apply_envelope, envelope_width=envelope_width,
                                       channel=channel,channel_ref=channel_BB,channel_BB=channel_BB,
                                     smoothing=smoothing,
                                     valid_thresh=valid_thresh,
                                     window=False,
                                     BB_normalize=False,
                                     align_phase=align_phase,
                                     view_phase_alignment=False)

        phase = SP.get_phase(f, spectrum, level=True)

        return np.array([f,
                         spectrum_abs.real,
                         phase.real],dtype=float)
    except:
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'BB_referenced_spectrum.err')
        with open(error_file,'w') as f:
            f.write(error_text)

        raise

def BB_referenced_linescan(linescan,spectra_BB,
                          apply_envelope=True, envelope_width=1,
                           smoothing=None,align_phase=False,
                           valid_thresh=.01,
                            channel=1, channel_BB=1):

    try:

        fmutual = None
        SP = None
        spectra_abs = []; phases=[]
        for spectra in linescan:

            if SP is None:
                SP=SpectralProcessor(spectra, spectra_BB,
                                     spectra_BB, spectra_BB)
            else:
                SP.sample_spectra = spectra #re-set sample spectra, only

            # `BB_normalize=False` is key, to use only the envelope from BB, but overall normalize spectra directly to BB
            f, spectrum_abs, spectrum = SP(apply_envelope=apply_envelope, envelope_width=envelope_width,
                                           channel=channel,channel_ref=channel_BB,channel_BB=channel_BB,
                                         smoothing=smoothing,
                                         valid_thresh=valid_thresh,
                                         BB_normalize=False,
                                         align_phase=align_phase,
                                         view_phase_alignment=False,
                                         recompute_reference=False) #`recompute_reference=False` saves us half our effort
            phase = SP.get_phase(f, spectrum, level=True)

            if fmutual is None:  fmutual = f
            else:
                phase = interp1d(f,phase,**interp_kwargs)(fmutual)
                spectrum_abs = interp1d(f,spectrum_abs,**interp_kwargs)(fmutual)

            spectra_abs.append(spectrum_abs)
            phases.append(phase)

        #Put everything together as three pages in a 3D array
        # 1) N copies of frequency axis (could in general have been distinct axes)
        # 2) N abs spectra
        # 3) N phase spectra
        fmutuals = np.array([fmutual]*len(linescan))
        spectra_abs = np.array(spectra_abs)
        phases = np.array(phases)

        return np.array([fmutuals.real,
                         spectra_abs.real,
                         phases.real],dtype=float)
    except:
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'BB_referenced_linescan.err')
        with open(error_file,'w') as f:
            f.write(error_text)

        raise

def standardize_spectrum_shape(spectrum):

    # Let us assume data may be coming in as shaped:
    #   1) (Naccumulations x 5, Nfreq)  <-- Ancient but previous 'standard' format
    #   2) (Naccumulations, 5, Nfreq)  <-- Regular but nonstandard format; but we now like it as effectively Nchannels=1
    #   3) (Naccumulations, 3+2*Nchannels, Nfreq)  <-- New and desired format
    # And no matter what, the output will be shaped like 3)

    spectrum = np.array(spectrum)
    assert spectrum.ndim in (2, 3), '`spectrum` array must be 2D or 3D!'

    if spectrum.ndim == 3:
        Nchannels2 = spectrum.shape[1] - 1 - SpectralProcessor.Nrows_env
        assert (Nchannels2 % 2)==0, "If a 3D array, `spectrum` must have an integral number of (amplitude,phase) channels!"
        return spectrum

    # assume we are 2D
    Naccum = spectrum.shape[0] / 5
    assert Naccum == np.round(Naccum),'If `spectrum is 2D, it should have shape (Naccumulations * 5, Nfreqs).'
    Nfreq = spectrum.shape[1]

    # Now make accumulations the first axis:
    newshape = (int(Naccum), 5, Nfreq) # Now it's the standard shape, but with one channel

    spectrum = np.reshape(spectrum, newshape)

    return spectrum

def zero_phases_in_interval(frequencies,phases,sabs,zero_phase_interval,
                            N2pis=100,weighted=False):

    # Set phase within some frequency interval to zero,
    #  optimizing for flatness up to an arbitrary factor of 2pi
    #  Which factor of 2pi is best?  Whichever minimizes the flatness residual
    if not (hasattr(zero_phase_interval, '__len__')
            and len(zero_phase_interval) >= 2): return phases

    # We want a list of phase spectra, and we will work over all of the individual spectra
    # We assume all share the same frequencies axis
    if phases.ndim==1:
        assert sabs.ndim==1
        phases=np.array([phases])
        sabs = np.array([sabs])
        single_spectrum = True
    else: single_spectrum=False

    fmin, fmax = np.min(zero_phase_interval), np.max(zero_phase_interval)
    f0 = np.mean((fmin, fmax))
    interval = (frequencies > fmin) * (frequencies < fmax)

    n2pis = np.arange(-N2pis, N2pis + 1, 1)
    n2pis = n2pis[np.newaxis, :]  # Try leveling into different `2pi` regimes

    if interval.any():  # Only do anything if interval turns out to have data
        for n, phase in enumerate(phases):  # iterate over points
            include = interval * (~np.isclose(sabs[n], 0))  # disregard empty data points
            if not include.any(): continue

            if weighted: pval = np.sum(sabs[n][include]**2*phase[include])/np.sum(sabs[n][include]**2)
            else: pval = np.mean(phase[include])  # average phase inside freq interval

            pcorr = frequencies[:, np.newaxis] / f0 * (
                        2 * np.pi * n2pis - pval) - 2 * np.pi * n2pis  # This will make phase in interval equal to some multiple of 2pi
            phase_options = phase[:, np.newaxis] + pcorr
            residuals = np.sum((phase_options[include]) ** 2, axis=0)  # Minimize the average slope
            phases[n] = phase_options[:, np.argmin(residuals)]

    if single_spectrum: phases = np.array(phases).squeeze() # Remove the redunant outer dimension

    return phases

def normalized_spectrum(sample_spectra, sample_BB_spectra,
                        ref_spectra, ref_BB_spectra,
                        apply_envelope=True, envelope_width=1.5,
                        level_phase=False, align_phase=True,
                        phase_offset=0,smoothing=None, valid_thresh=.01,
                        piecewise_flattening=0,
                        zero_phase_interval=None,
                        channel=1,channel_ref=1,
                        self_reference=False,channel_BB=1,
                        phase_alignment_exponent=0.25,
                        BB_normalize=False,
                        **kwargs):

    sample_spectra = standardize_spectrum_shape(sample_spectra)
    sample_BB_spectra = standardize_spectrum_shape(sample_BB_spectra)

    # In case we want to normalize one sample channel to another, we use `self_reference=True`
    if self_reference:
        ref_spectra=sample_spectra
        sample_BB_spectra = ref_BB_spectra = sample_spectra # Since `channel_ref` of sample spectrum will be in denominator, it should itself serve as BB
    else:
        ref_spectra = standardize_spectrum_shape(ref_spectra)
        ref_BB_spectra = standardize_spectrum_shape(ref_BB_spectra)

    try:
        SP = SpectralProcessor(sample_spectra, sample_BB_spectra,
                               ref_spectra, ref_BB_spectra,
                               phase_alignment_exponent=phase_alignment_exponent)

        f, snorm_abs,snorm = SP(channel=channel,channel_ref=channel_ref,channel_BB=channel_BB,
                                apply_envelope=apply_envelope, envelope_width=envelope_width,
                                smoothing=smoothing,
                                valid_thresh=valid_thresh,
                                window=False,
                                align_phase=align_phase,
                                view_phase_alignment=False,
                                BB_normalize=BB_normalize,
                                **kwargs)

        phase = SP.get_phase(f, snorm, level=False,
                             order=0, manual_offset=None)  # Only level using the offset we apply

        if level_phase:

            phase = zero_phases_in_interval(frequencies=f, phases=phase, sabs=snorm_abs,
                                             zero_phase_interval=zero_phase_interval)

            if piecewise_flattening:
                phase = SP.get_phase(f, snorm, level=False)
                to_fit = phase
                offset,params = numrec.PiecewiseLinearFit(f,to_fit,nbreakpoints=piecewise_flattening,error_exp=2)
                phase -= offset

            if phase_offset:
                #Now finally apply manual offset
                f0 = np.mean(f)
                phase += phase_offset * f / f0

        return np.array([f,
                         snorm_abs.real,
                         phase.real], dtype=float)

    except:

        #--- Dump the error
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'normalized_spectrum.err')
        with open(error_file,'w') as file: file.write(error_text)

        filepath = os.path.join(diagnostic_dir, 'normalized_spectrum_inputs.err.pickle')
        with open(filepath,'wb') as file:
            pickle.dump(dict(sample_spectra=sample_spectra,
                             sample_BB_spectra=sample_BB_spectra,
                             ref_spectra=ref_spectra,
                             ref_BB_spectra=ref_BB_spectra),
                        file)

        raise

def normalized_linescan(sample_linescan, sample_BB_spectra,
                        ref_spectra, ref_BB_spectra,
                        apply_envelope=True, envelope_width=1.5,
                        level_phase=False, align_phase=False,
                        phase_offset=0, smoothing=None, valid_thresh=.01,
                        piecewise_flattening=0,
                        zero_phase_interval=None,
                        channel=1,channel_ref=1,
                        self_reference=False, channel_BB=1,
                        heal_linescan=False,
                        phase_alignment_exponent=0.25,
                        subtract_baseline=False,
                        BB_normalize=False):
    # Don't mess with the order of the arguments list, it's sensitively tuned for LabView!

    global SP

    sample_linescan = np.array(sample_linescan)
    sample_BB_spectra = np.array(sample_BB_spectra)
    ref_spectra = np.array(ref_spectra)
    ref_BB_spectra = np.array(ref_BB_spectra)

    if heal_linescan:  sample_linescan = heal_linescan_recursive(sample_linescan)

    try:

        fmutual = None
        SP = None
        snorms_abs = []; phases=[]
        #Iterate through spatial pixels
        for i in range(len(sample_linescan)):

            print('Processing linescan pixel %i of %i...'%(i+1,len(sample_linescan)))
            sample_spectra = sample_linescan[i]

            if self_reference:
                ref_spectra = sample_spectra
                ref_BB_spectra = sample_BB_spectra = sample_spectra # Since `channel_ref` of sample spectrum will be in denominator, just use itself as BB for envelope!

            #--- Compute normalized spectrum at this pixel
            if SP is None: # Initialize spectral processor
                SP = SpectralProcessor(sample_spectra, sample_BB_spectra,
                                       ref_spectra, ref_BB_spectra,
                                       phase_alignment_exponent=phase_alignment_exponent)
                recompute_reference = True
            else:
                SP.sample_spectra = sample_spectra #re-set sample spectra, only
                # Here we are updating the reference spectrum locally, too
                if self_reference:
                    print('Using self referencing for ref spectra!')
                    SP.ref_spectra = sample_spectra
                    recompute_reference = True
                else:
                    recompute_reference = False #`recompute_reference=False` saves us half our effort, if we haven't changed it!

            f, snorm_abs, snorm = SP(channel=channel,channel_ref=channel_ref,channel_BB=channel_BB,
                                     apply_envelope=apply_envelope, envelope_width=envelope_width,
                                     smoothing=smoothing,
                                     valid_thresh=valid_thresh,
                                     window=False,
                                     align_phase=align_phase,
                                     BB_normalize=BB_normalize,
                                     view_phase_alignment=False,
                                     recompute_reference=recompute_reference)

            #--- Do all the phase leveling
            if level_phase: # We made a choice here not to allow baseline subtraction, keep the leveling "physical"
                snorm = SP.level_phase(f, snorm, order=1, manual_offset=None,
                                       weighted=False, subtract_baseline=subtract_baseline)

            # Now get phase
            phase = SP.get_phase(f, snorm, level=True, weighted=False,
                                 order=0, manual_offset=phase_offset)  # Order 0 means only the offset is used

            #--- Interpolate normalized spectrum at this pixel to common frequency axis
            if fmutual is None:  fmutual = f
            else:
                phase = interp1d(f,phase,**interp_kwargs)(fmutual)
                snorm_abs = interp1d(f,snorm_abs,**interp_kwargs)(fmutual)

            snorms_abs.append(snorm_abs)
            phases.append(phase)

        #Put everything together as three pages in a 3D array
        # 1) N copies of frequency axis (could in general have been distinct axes)
        # 2) N abs spectra
        # 3) N phase spectra
        fmutuals = np.array([fmutual]*len(sample_linescan))
        snorms_abs = np.array(snorms_abs)
        phases = np.array(phases)

        #--- Phase leveling
        if level_phase:

            # Set phase within some frequency interval to zero
            phases = zero_phases_in_interval(frequencies=fmutuals[0], phases=phases, sabs=snorms_abs,
                                             zero_phase_interval=zero_phase_interval)

            # Piecewise leveling only makes sense as the final step
            if piecewise_flattening:
                for n, phase in enumerate(phases): #iterate over points
                    to_fit = phase
                    offset, params = numrec.PiecewiseLinearFit(fmutual, to_fit, nbreakpoints=piecewise_flattening, error_exp=2)
                    phase -= offset

            # Finally restore the phase offset
            if phase_offset:
                f0=np.mean(fmutual)
                for n, phase in enumerate(phases): #iterate over points
                    phase += phase_offset * fmutual/f0

        return np.array([fmutuals.real,
                         snorms_abs.real,
                         phases.real],dtype=float)
    except:
        error_text = str(traceback.format_exc())
        error_text += '\n sample spectra shape: %s'%str(sample_spectra.shape)
        error_text += '\n sample BB spectra shape: %s'%str(sample_BB_spectra.shape)
        error_text += '\n reference BB spectra shape: %s'%str(ref_spectra.shape)
        error_text += '\n reference spectra shape: %s'%str(ref_BB_spectra.shape)
        error_file = os.path.join(diagnostic_dir,'normalized_linescan.err')
        with open(error_file,'w') as f:
            f.write(error_text)

        raise

def save_spectrum(sample_spectrum, BB_spectrum, path):

    sample_spectrum=np.array(sample_spectrum) #Assume array data type for now
    BB_spectrum = np.array(BB_spectrum)

    d = dict(sample_spectrum=sample_spectrum,
             BB_spectrum=BB_spectrum)

    with open(path,'wb') as f:
        pickle.dump(d,f)

    return True

def save_linescan(sample_linescan, BB_spectrum, path):

    sample_linescan=np.array(sample_linescan) #Assume array data type for now
    BB_spectrum = np.array(BB_spectrum)

    d = dict(sample_linescan=sample_linescan,
             BB_spectrum=BB_spectrum)

    with open(path,'wb') as f:
        pickle.dump(d,f)

    return True

def load_linescan(path):

    try:
        assert path.endswith('.pickle')
        # It could be that
        with open(path,'rb') as f:
            d = pickle.load(f)

        # First, "standard" format is both sample and BB in the same dictionary
        if isinstance(d,dict):
            keys = d.keys()
            assert 'sample_linescan' in keys
            assert 'BB_spectrum' in keys

            sample_linescan = d['sample_linescan'] # Assumed to be 4D: (Npints, Naccum, Nchannel+.., Nfreq)
            BB_spectrum = d['BB_spectrum'] # Assumed to be 3D: (Naccum, Nchannel+.., Nfreq)

        # If we don't have dictionary, assume we have some array,
        # and further require that there is a file ending "_BB.txt" in place of '.pickle' for BB spectra
        else:
            assert isinstance(d,np.array),'Data at "%s" must be an array!'%path
            BB_filepath = path.rstrip('.pickle')+'_BB.txt'
            assert os.path.exists(BB_filepath),'With this data, file "%s" must exist!'%BB_filepath

            sample_linescan = [standardize_spectrum_shape(point_spectrum) for point_spectrum in d]
            sample_linescan = np.array(sample_linescan)
            BB_spectrum = standardize_spectrum_shape(np.loadtxt(BB_filepath)) # Assume we are loading a 2D array

        # They should both have the same number of accumulations
        Naccum_samp = sample_linescan.shape[1]
        Naccum_BB = len(BB_spectrum)
        assert Naccum_samp==Naccum_BB,'Sample spectra and BB spectrum should have same number of accumulations, but %i!=%i!'%(Naccum_samp,Naccum_BB)

        # Output is tuple of a 4D + 3D array:
        # (Npoints, Naccum, Nchannel+.., Nfreq)
        # (Naccum, Nchannel+.., Nfreq)
        return (sample_linescan,BB_spectrum)

    except:

        #--- Dump the error
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'load_linescan.err')
        with open(error_file,'w') as f: f.write(error_text)

        raise


def load_spectrum(path):

    try:
        assert path.endswith('.pickle')
        # It could be that
        with open(path, 'rb') as f:
            d = pickle.load(f)

        # First, "standard" format is both sample and BB in the same dictionary
        if isinstance(d, dict):
            keys = d.keys()
            assert 'sample_spectrum' in keys
            assert 'BB_spectrum' in keys

            sample_spectrum = d['sample_spectrum']  # Assumed to be 3D: (Naccum, Nchannel+.., Nfreq)
            BB_spectrum = d['BB_spectrum']  # Assumed to be 3D: (Naccum, Nchannel+.., Nfreq)

        # If we don't have dictionary, assume we have some array,
        # and further require that there is a file ending "_BB.txt" in place of '.pickle' for BB spectra
        else:
            assert isinstance(d, np.array), 'Data at "%s" must be an array!' % path
            BB_filepath = path.rstrip('.pickle') + '_BB.txt'
            assert os.path.exists(BB_filepath), 'With this data, file "%s" must exist!' % BB_filepath

            sample_spectrum = standardize_spectrum_shape(d)
            BB_spectrum = standardize_spectrum_shape(np.loadtxt(BB_filepath))  # Assume we are loading a 2D array

        # They should both have the same number of accumulations
        Naccum_samp = len(sample_spectrum)
        Naccum_BB = len(BB_spectrum)
        assert Naccum_samp == Naccum_BB, 'Sample spectrum and BB spectrum should have same number of accumulations, but %i!=%i!' % (
        Naccum_samp, Naccum_BB)

        # Output is tuple of 3D arrays: (Naccum, Nchannel+.., Nfreq)
        return (sample_spectrum,BB_spectrum)

    except:

        # --- Dump the error
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir, 'load_linescan.err')
        with open(error_file, 'w') as f:
            f.write(error_text)

        raise