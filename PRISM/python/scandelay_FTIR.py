import os
import numpy as np
import traceback
import scipy
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

def test(intfgs_arr, delay_calibration_factor=1,
                         order=1,refresh_alignment=True,
                         flattening_order=20,Nwavelengths=4):

    try:
        file=os.path.join(diagnostic_dir,'intfg_err.txt')
        intfgs_arr = np.loadtxt(file)
        result =align_interferograms_base(intfgs_arr, delay_calibration_factor=-2.5,
                                     shift0=-10,optimize_shift=True,shift_range=10,
                                     flattening_order=10,noise=0)
        return str(1)

    except:
        #s=str(np.__version__)+','+str(scipy.__version__)
        s=str(np.__file__)
        s= str(traceback.format_exc())
        return s

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

def align_interferograms_test(intfgs_arr, delay_calibration_factor=1,
                         shift0=None,optimize_shift=True,shift_range=15,
                         flattening_order=5,noise=0,
                         fit_xs=True, fit_xs_order=6,smooth_xs = 100):

    global dX,best_delay,xaxis,all_xs0,all_xs_fitted
    global intfg_mutual_fwd,intfg_mutual_bwd
    if best_delay is None: best_delay=shift0

    intfgs_arr = np.array(intfgs_arr)
    Ncycles = len(intfgs_arr) // 2

    # --- Process data
    all_xs = []
    all_ys = []
    for i in range(Ncycles):
        # i=i+10
        ys, xs = intfgs_arr[2 * i:2 * (i + 1)]
        all_ys.append(ys)
        all_xs.append(xs)

    all_xs = np.mean(np.array(all_xs),axis=0)

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
        params, _ = numrec.ParameterFit(indices, all_xs, model_xs, params0)
        amps_phases = np.array(params[2:]).reshape((fit_xs_order,2))
        print('Fitted sinusoid parameters for X coordinate:')
        for i,(amp,phase) in enumerate(amps_phases):
            print('Harmonic %i: amp=%1.2G, phase=%1.2f'%(i+1,amp,phase%(2*np.pi)))
        all_xs0 = all_xs
        all_xs = model_xs(indices, params)
        all_xs_fitted = all_xs

    all_ys = np.mean(np.array(all_ys),axis=0)
    all_ys = all_ys - np.polyval(np.polyfit(x=all_xs, y=all_ys, deg=flattening_order), all_xs)

    # --- Get fwd/bwd interferograms
    idx_turn = np.argmax(all_xs) #We assume max is in midway point, BEFORE applying calibration factor
    all_xs *= delay_calibration_factor
    all_xs_fwd = all_xs[idx_turn:]
    all_ys_fwd = all_ys[idx_turn:]
    all_xs_bwd = all_xs[:idx_turn]
    all_ys_bwd = all_ys[:idx_turn]

    #--- Determine and update the global x-range, if needed
    x1 = np.min(all_xs)
    x2 = np.max(all_xs)
    Nbins = len(all_xs)//2
    if xaxis is None:
        print('Re-using pre-existing x-axis for interferograms...')
        xaxis = np.linspace(x1, x2, Nbins)

    #--- If we don't need to optimize, return what we need
    def shifted_intfg(shift):

        all_xs_rolled = np.roll(all_xs, shift, axis=0)
        result = binned_statistic(all_xs_rolled, all_ys, bins=Nbins)
        xnew = result.bin_edges[:-1]
        intfg_new = result.statistic

        keep = np.isfinite(intfg_new)
        intfg_new = intfg_new[keep]
        xnew = xnew[keep]

        return xnew, intfg_new

    if not optimize_shift:
        best_delay = shift0
        x, intfg = shifted_intfg(shift0)

        intfg_interp = interp1d(x=x, y=intfg,
                                **interp_kwargs)
        intfg_new = intfg_interp(xaxis)

        return np.array([intfg_new, xaxis])

    #--- Get fwd / bwd interferograms and their mutual overlap
    result = binned_statistic(all_xs_fwd.flatten(),
                              all_ys_fwd.flatten(),
                              bins=Nbins)
    x_fwd = result.bin_edges[:-1]
    intfg_fwd = result.statistic
    where_keep=np.isfinite(intfg_fwd)
    intfg_fwd=intfg_fwd[where_keep]
    x_fwd = x_fwd[where_keep]
    intfg_fwd = AWA(intfg_fwd, axes=[x_fwd])

    result = binned_statistic(all_xs_bwd.flatten(),
                              all_ys_bwd.flatten(),
                              bins=Nbins)
    x_bwd = result.bin_edges[:-1]
    intfg_bwd = result.statistic
    where_keep=np.isfinite(intfg_bwd)
    intfg_bwd=intfg_bwd[where_keep]
    x_bwd = x_bwd[where_keep]
    intfg_bwd = AWA(intfg_bwd, axes=[x_bwd])

    xmin = np.max((np.min(x_fwd), np.min(x_bwd)))
    xmax = np.min((np.max(x_fwd), np.max(x_bwd)))
    xmutual = np.linspace(xmin, xmax, Nbins)
    #raise ValueError
    intfg_mutual_fwd = intfg_fwd.interpolate_axis(xmutual, axis=0,**interp_kwargs)
    intfg_mutual_bwd = intfg_bwd.interpolate_axis(xmutual, axis=0,**interp_kwargs)

    #--- Define some helper functions for identifying x-coordinate of interferograms
    def get_dx(intfg_mutual_fwd, intfg_mutual_bwd, exp=4):

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

        return dx

    def get_x0(intfg_mutual_fwd, intfg_mutual_bwd, exp=2):

        x = intfg_mutual_fwd.axes[0]

        intfg_fwd = intfg_mutual_fwd
        intfg_bwd = intfg_mutual_bwd
        w = np.abs(intfg_fwd) ** exp + np.abs(intfg_bwd) ** exp

        x0 = np.sum(x * w) / np.sum(w)

        return x0

    #--- Find average characteristic index of x0_fwd vs x0_bwd = shift0
    if not shift0:
        x0 = get_x0(intfg_mutual_fwd, intfg_mutual_bwd, exp=4)
        print('x0:', x0)
        dx = get_dx(intfg_mutual_fwd, intfg_mutual_bwd, exp=4)
        print('Fwd/bwd dx separation:', dx)
        x0_fwd = x0 + dx / 2
        x0_bwd = x0 - dx / 2

        n2 = np.argmin(np.abs(all_xs_fwd - x0_fwd))
        n1 = np.argmin(np.abs(all_xs_fwd - x0_bwd))
        dn=np.round((n2 - n1) / 2.)

        shift0 = np.int(dn)
        print('Initially optimal shift:', shift0)

    # -- Optimize around shift0
    shifts = np.arange(-shift_range, shift_range, 1) + shift0
    sums = []
    for shift in shifts:
        x, intfg = shifted_intfg(shift)
        sums.append(np.sum(intfg ** 2))
    shift0 = shifts[np.argmax(sums)]

    print('Finally optimal shift:', shift0)
    best_delay = shift0
    x, intfg = shifted_intfg(shift0)

    intfg_interp = interp1d(x=x,y=intfg,
                            **interp_kwargs)
    intfg_new = intfg_interp(xaxis)

    return np.array([intfg_new,xaxis])

def align_interferograms_base(intfgs_arr, delay_calibration_factor=1,
                         shift0=None,optimize_shift=True,shift_range=15,
                         flattening_order=5,noise=0,
                         fit_xs=True, fit_xs_order=4,smooth_xs = 100):

    global dX,best_delay,Nbins
    global intfg_mutual_fwd,intfg_mutual_bwd
    if best_delay is None: best_delay=shift0

    intfgs_arr = np.array(intfgs_arr)
    Ncycles = len(intfgs_arr) // 2
    Nsamples = intfgs_arr.shape[1]

    indices = np.arange(Nsamples)

    # --- Process data
    all_xs_fwd = []
    all_ys_fwd = []
    all_xs_bwd = []
    all_ys_bwd = []
    all_xs = []
    all_ys = []
    for i in range(Ncycles):
        # i=i+10
        ys, xs = intfgs_arr[2 * i:2 * (i + 1)]

        if fit_xs:
            Nsamples = len(xs)
            indices = np.arange(Nsamples)
            offset = np.mean(xs)
            period = float(Nsamples)
            amps = [0] * fit_xs_order
            amps[0] = np.ptp(xs) / 2
            phases = [0] * fit_xs_order
            params0 = [offset, period]
            for amp, phase in zip(amps, phases): params0 = params0 + [amp, phase]
            params, _ = numrec.ParameterFit(indices, xs, model_xs, params0)
            amps_phases = np.array(params[2:]).reshape((fit_xs_order, 2))
            print('Fitted sinusoid parameters for X coordinate:')
            for j, (amp, phase) in enumerate(amps_phases):
                print('Harmonic %i: amp=%1.2G, phase=%1.2f' % (j + 1, amp, phase % (2 * np.pi)))
            xs = model_xs(indices, params)

        elif smooth_xs:
            xs = numrec.smooth(xs, window_len=smooth_xs, axis=0)

        if i == 0:
            idx_turn = np.argmax(xs)

            #We can set `Nbins` for the first time, based on the typical sampling capability for the interferograms
            if Nbins is None:
                Nbins = len(xs) #*np.max( (1,int(Ncycles/5)) )
                # We want odd number of bins, for an odd number of (symmetric) frequency channels in computed spectrum
                # This will allow positive frequencies to hold entirety of spectral content
                if not Nbins % 2: Nbins+=1

        xs = xs*delay_calibration_factor
        ys = ys + noise * np.random.randn(len(ys)) * (ys.max() - ys.min()) / 2
        ys = ys - np.polyval(np.polyfit(x=xs, y=ys, deg=flattening_order), xs)

        all_xs_fwd.append(xs[:idx_turn])
        all_ys_fwd.append(ys[:idx_turn])

        all_xs_bwd.append(xs[idx_turn:])
        all_ys_bwd.append(ys[idx_turn:])

        all_xs.append(xs)
        all_ys.append(ys)

    # --- Get fwd/bwd interferograms
    all_xs_fwd = np.array(all_xs_fwd)
    all_xs_bwd = np.array(all_xs_bwd)
    all_ys_fwd = np.array(all_ys_fwd)
    all_ys_bwd = np.array(all_ys_bwd)
    all_xs = np.array(all_xs).flatten()
    all_ys = np.array(all_ys).flatten()

    #--- Update the global x-range, if needed
    x1 = np.min(all_xs)
    x2 = np.max(all_xs)
    if dX is None: dX = x2 - x1
    xnew = np.linspace(x1, x1 + dX, Nbins)

    #--- If we don't need to optimize, return what we need
    def shifted_intfg(shift):

        all_xs_rolled = np.roll(all_xs, shift, axis=0)
        result = binned_statistic(all_xs_rolled, all_ys, bins=Nbins)
        xnew = result.bin_edges[:-1]
        intfg_new = result.statistic

        keep = np.isfinite(intfg_new)
        intfg_new = intfg_new[keep]
        xnew = xnew[keep]

        return xnew, intfg_new

    if not optimize_shift:
        best_delay = shift0
        x, intfg = shifted_intfg(shift0)

        intfg_interp = interp1d(x=x, y=intfg,
                                **interp_kwargs)
        intfg_new = intfg_interp(xnew)

        return np.array([intfg_new, xnew])

    #--- Get fwd / bwd interferograms and their mutual overlap
    result = binned_statistic(all_xs_fwd.flatten(),
                              all_ys_fwd.flatten(),
                              bins=Nbins)
    x_fwd = result.bin_edges[:-1]
    intfg_fwd = result.statistic
    where_keep=np.isfinite(intfg_fwd)
    intfg_fwd=intfg_fwd[where_keep]
    x_fwd = x_fwd[where_keep]
    intfg_fwd = AWA(intfg_fwd, axes=[x_fwd])

    result = binned_statistic(all_xs_bwd.flatten(),
                              all_ys_bwd.flatten(),
                              bins=Nbins)
    x_bwd = result.bin_edges[:-1]
    intfg_bwd = result.statistic
    where_keep=np.isfinite(intfg_bwd)
    intfg_bwd=intfg_bwd[where_keep]
    x_bwd = x_bwd[where_keep]
    intfg_bwd = AWA(intfg_bwd, axes=[x_bwd])

    xmin = np.max((np.min(x_fwd), np.min(x_bwd)))
    xmax = np.min((np.max(x_fwd), np.max(x_bwd)))
    xmutual = np.linspace(xmin, xmax, Nbins)
    intfg_mutual_fwd = intfg_fwd.interpolate_axis(xmutual, axis=0,**interp_kwargs)
    intfg_mutual_bwd = intfg_bwd.interpolate_axis(xmutual, axis=0,**interp_kwargs)

    #--- Define some helper functions for identifying x-coordinate of interferograms
    def get_dx(intfg_mutual_fwd, intfg_mutual_bwd, exp=4):

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

        return dx

    def get_x0(intfg_mutual_fwd, intfg_mutual_bwd, exp=2):

        global intfg_fwd_fil, intfg_bwd_fil
        x = intfg_mutual_fwd.axes[0]

        intfg_fwd = intfg_mutual_fwd
        intfg_bwd = intfg_mutual_bwd
        w = np.abs(intfg_fwd) ** exp + np.abs(intfg_bwd) ** exp

        x0 = np.sum(x * w) / np.sum(w)

        return x0

    #--- Find average characteristic index of x0_fwd vs x0_bwd = shift0
    if not shift0:
        x0 = get_x0(intfg_mutual_fwd, intfg_mutual_bwd, exp=4)
        print('x0:', x0)
        dx = get_dx(intfg_mutual_fwd, intfg_mutual_bwd, exp=4)
        print('Fwd/bwd dx separation:', dx)
        x0_fwd = x0 + dx / 2
        x0_bwd = x0 - dx / 2

        dns = []
        for xs_fwd in all_xs_fwd:
            n2 = np.argmin(np.abs(xs_fwd - x0_fwd))
            n1 = np.argmin(np.abs(xs_fwd - x0_bwd))
            dns.append((n2 - n1) / 2.)

        shift0 = np.int(np.round(np.mean(dns)))
        print('Initially optimal shift:', shift0)

    # -- Optimize around shift0
    shifts = np.arange(-shift_range, shift_range, 1) + shift0
    sums = []
    for shift in shifts:
        x, intfg = shifted_intfg(shift)
        sums.append(np.sum(intfg ** 2))
    shift0 = shifts[np.argmax(sums)]

    print('Finally optimal shift:', shift0)
    best_delay = shift0
    x, intfg = shifted_intfg(shift0)

    intfg_interp = interp1d(x=x,y=intfg,
                            **interp_kwargs)
    intfg_new = intfg_interp(xnew)

    return np.array([intfg_new,xnew])


#--- Wrapper
def align_interferograms(intfgs_arr, delay_calibration_factor=1,
                         shift0=0,optimize_shift=True,shift_range=15,
                         flattening_order=20,noise=0):

    try:
        result = align_interferograms_test(intfgs_arr, delay_calibration_factor=delay_calibration_factor,
                                           shift0=shift0,optimize_shift=optimize_shift,shift_range=shift_range,
                                           flattening_order=flattening_order,noise=noise)
        return result

    except:

        #--- Dump the error
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'align_interferograms.err')
        with open(error_file,'w') as f: f.write(error_text)

        #--- Dump the problematic interferograms
        error_file = os.path.join(diagnostic_dir,'intfg_err.txt')
        if not os.path.exists(error_file):
            np.savetxt(error_file,intfgs_arr)

        return False

def spectral_envelope(f,A,f0,df,expand_envelope=1):

    df = expand_envelope*df
    e = np.exp(-(f-f0)**2/(2*df**2))

    return A*e #maximum value will always be `A`

def fit_envelope(f,sabs,fmin=400,fmax=3500):

    global result
    sub=(f>=fmin)*(f<=fmax)
    fsub=f[sub]
    sabssub=sabs[sub]
    ind=np.argmax(sabssub)
    A=sabssub[ind]
    f0 = fsub[ind]
    df = f0/10
    x0=np.array([A,f0,df])

    def to_minimize(x):

        A,f0,df=x
        delta = sabssub-spectral_envelope(fsub,A,f0,df,
                                          expand_envelope=1)
        return delta

    envelope_params = leastsq(to_minimize,x0,maxfev=100)[0]

    envelope = spectral_envelope(f,*envelope_params)
    #The overall value will not be meaningful, so be sure to normalize to 1 later, before applying

    return envelope,envelope_params

def fourier_xform(a,tsubtract=0,envelope=True,gain=1,fmin=500,fmax=3000):

    v,t=np.array(a)
    c=3e8 #speed of light in m/s
    t=t-tsubtract #subtract a time correction in ps
    t*=1e-12 #ps in seconds
    d=t*c*2 #distance in m
    d*=1e2 #distance in cm

    v = v-np.mean(v)
    v *= gain
    w = np.blackman(len(v))
    #wmax_ind = np.argmax(w) #Let's roll window to position of maximum weight in interferogram
    #centerd = np.sum(d * v ** 2) / np.sum(v ** 2)
    #imax_ind = np.argmin((centerd - d) ** 2)
    #w = np.roll(w, imax_ind - wmax_ind, axis=0)
    #w = 1 #We are disabling the windowing for now

    pow0 = np.sum(v**2)
    v *= w
    pownew = np.sum(v**2)
    norm = np.sqrt(pow0/pownew)
    v = AWA(v, axes=[d])
    s = num.Spectrum(v, axis=0) * norm
    f = s.axes[0]

    # Just discard the information about absolute coordinates;
    # assume spectra will only be compared when they share the same x-axis
    # The starting coordinate in x is physically meaninful
    #pcorr = 2 * np.pi * (np.min(d)) * f
    #s *= np.exp(+1j * pcorr) #plane waves are in a basis `exp(-1j*x...)`, so we are shifting by `-x0`

    posf=(f>0)
    if fmax is not None: posf*=(f<fmax)
    if fmin is not None: posf*=(f>fmin)
    pvf=np.angle(s)[posf]
    pvf = (pvf+np.pi)%(2*np.pi)-np.pi #This modulo's the phase to closest interval near 0
    sabs=np.abs(s)[posf]
    f=f[posf]

    if envelope: env,envelope_params = fit_envelope(f,sabs)
    else: env=[1]*len(f); envelope_params=[0]
    Nf=len(f); Ne=len(envelope_params)
    if Ne<Nf: envelope_params=np.append(envelope_params,[0]*(Nf-Ne))

    assert len(f)==len(sabs)==len(pvf)==len(env)

    return np.array([f,sabs,pvf,env,envelope_params])

def fourier_xform_fwd_bwd(a,tsubtract=0,envelope=True):

    v1,t1,v2,t2=np.array(a)
    c=3e8 #speed of light in m/s

    pairs = [(v1,t1),
             (v2,t2)]
    ss=[]
    for i in range(2):
        v,t = pairs[i]
        t=t-tsubtract #subtract a time correction in ps
        t*=1e-12 #ps in seconds
        d=t*c*2 #distance in m
        d*=1e2 #distance in cm

        v = v-np.mean(v)
        w = np.blackman(len(v))
        """wmax_ind = np.argmax(w)
        centerd = np.sum(d * v ** 2) / np.sum(v ** 2)
        imax_ind = np.argmin((centerd - d) ** 2)
        w = np.roll(w, imax_ind - wmax_ind, axis=0)"""
        #w = 1 #We are disabling the windowing for now

        v *= w
        v = AWA(v, axes=[d])
        s = num.Spectrum(v, axis=0)

        # This applies knowledge that d does not start at zero, but min value is meaningful
        f = s.axes[0]
        pcorr = +2 * np.pi * (np.min(d)) * f
        s *= np.exp(1j * pcorr)

        ss.append(s)

    #-- Interpolate the fwd/bwd spectra to share frequency channels
    s1,s2=ss
    f1=s1.axes[0]
    s2 = s1.interpolate_axis(f1,axis=0,**interp_kwargs)
    s1abs,s2abs = np.abs(s1),np.abs(s2)
    p1,p2 = np.unwrap(np.angle(s1)),np.unwrap(np.angle(s2))
    pvf = (p1*s1abs+p2*s2abs)/(s1abs+s2abs) # weighted average phase
    sabs = (np.abs(s1)+np.abs(s2))/2.

    posf=(f>0)
    pvf=pvf[posf]
    pvf = (pvf+np.pi)%(2*np.pi)-np.pi #This modulo's the phase to closest interval near 0
    sabs=sabs[posf]
    f=f[posf]

    if envelope: env,envelope_params = fit_envelope(f,sabs)
    else: env=[1]*len(f); envelope_params=[0]
    Nf=len(f); Ne=len(envelope_params)
    if Ne<Nf: envelope_params=np.append(envelope_params,[0]*(Nf-Ne))

    assert len(f)==len(sabs)==len(pvf)==len(env)

    return np.array([f,sabs,pvf,env,envelope_params])

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
        fs_full = np.hstack( (-fs[::-1], [0], fs) ) #Include zero frequency
        scomplex_full = np.hstack( (scomplex.conj()[::-1],
                                    [0],
                                    scomplex) )
        interp=interp1d(x=fs_full,y=scomplex_full,
                        axis=0, fill_value=0,
                        kind='linear', bounds_error=False) #types beyond 'linear' type could give problematic extrapolations

        fs_fft = np.fft.fftfreq(len(scomplex_full), d=1 / (2 * fs.max()) ) #Assert `fs` goes up to Nyquist, so that `d` is sampling interval
        s_fft = interp(fs_fft)

        intfg = np.fft.ifft(s_fft).real
        Dx = 1 / np.diff(fs)[0]
        xs = np.linspace(0, Dx, len(intfg))

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
            pcorr = manual_offset * f/np.mean(np.abs(f))
            leveler *= np.exp(1j*pcorr)

        if return_leveler: return leveler
        else: return s*leveler

    @classmethod
    def get_phase(cls,f,s,
                  level_phase=True,
                  **kwargs):

        if level_phase:
            s=cls.level_phase(f,s,**kwargs)

        p = np.unwrap(np.angle(s))

        #Remove as many factors of 2*pi so that greatest intensity region is closest to zero
        if s.any():
            pavg = np.sum(p*np.abs(s)**2) / np.sum(np.abs(s)**2)
            p -= 2*np.pi * np.round(pavg/(2*np.pi))

        return p.real

    @classmethod
    def get_phases(cls,f,spectra,
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
            phase=cls.get_phase(f,s,manual_offset=0,
                                level_phase=False,**kwargs) #level phase has to be false, we want the overall phase!
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
        leveler = cls.level_phase(f1,s,order=order,manual_offset=0,return_leveler=True)

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

        use_previous = (valid_thresh is 'last') and hasattr(cls,'thresholded_range')
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

    @classmethod
    def accumulate_spectra(cls, spectra,
                           apply_envelope=True,
                           expand_envelope=1):

        spectra = np.array(spectra)

        Nrows = cls.Nrows

        assert len(spectra) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % Nrows
        Nspectra = np.max((len(spectra) // Nrows, 1))

        # Establish some mutual frequency axis
        all_fs = np.array([spectra[Nrows * i] for i in range(Nspectra)])
        f0 = None

        # Collect all the spectra and the envelopes
        ss = []
        for i in range(Nspectra):

            ## Analyte spectrum components
            f = spectra[Nrows * i]
            sabs = spectra[Nrows * i + 1]
            sphase = spectra[Nrows * i + 2]
            env_params = spectra[Nrows * i + 4][:3]

            s = sabs * np.exp(1j * sphase)

            # In case we have invalid entries (zeros), remove them
            where_valid = np.isfinite(f) * (f > 0)
            if not where_valid.any():
                print('Spectrum at entry %i is inferred empty!' % i)
                continue
            f = f[where_valid]
            s = s[where_valid]

            # Establish some mutual frequency axis
            if f0 is None: f0=f

            s = cls.interpolate_spectrum(s,f,f0)
            if apply_envelope:
                env = spectral_envelope(f0, *env_params,
                                        expand_envelope=expand_envelope)
                env /= env.max()
                s *= env

            ss.append(s)

        spectrum_abs = cls.summed_spectrum(ss, abs=True)
        spectrum_phase = cls.get_phase(f0,
                                       cls.summed_spectrum(ss,abs=False),
                                       level_phase=False) #We could add option for leveling phase

        return np.array([f0, spectrum_abs, spectrum_phase],dtype=np.float)

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

        return np.array(sout,dtype=np.complex)

    @classmethod
    def smoothed_spectrum(cls,f,scomplex,smoothing=4,order=2):

        leveler = cls.level_phase(f,scomplex,order=order,return_leveler=True,weighted=False)

        scomplexl = scomplex*leveler

        scomplexl = numrec.smooth(scomplexl,window_len=smoothing,axis=0)

        return scomplexl / leveler

    coincidence_exponent=6

    def phase_aligned_spectrum(self, f, spectra, verbose=False,
                               coincidence_exponent=None):

        if coincidence_exponent is None: coincidence_exponent=self.coincidence_exponent
        #self.s0s=[] #Debugging - check the operands for phase alignments

        def aligned_spectrum(s0, s, p0=0):

            #Determine and remove average phase of stot, since it could be wrapping like crazy
            leveler = self.level_phase(f, s0 , order=1, manual_offset=0, return_leveler=True)

            # Debugging - check the leveling
            #self.s0s.append(num.Spectrum(AWA(s0 * leveler, axes=[f], axis_names=['X Frequency']), axis=0))

            phase0 = self.get_phase(f, s0 * leveler, manual_offset=0,
                                    level_phase=False)
            phase = self.get_phase(f, s * leveler, manual_offset=0,
                                   level_phase=False)
            coincidence = (np.abs(s0) * np.abs(s)) ** coincidence_exponent

            if not coincidence.any(): return s,0

            norm = np.sum(coincidence)
            p = np.sum((phase0 - phase) * coincidence) / norm

            pdiff = p-p0
            p = p - np.round(pdiff/(2*np.pi))*2*np.pi #Add as many factors of 2*pi as minimizes `pdiff`
            # We are assuming that a physical shift the spectrum of order 2*pi is unlikely!

            fcenter = np.sum(f * coincidence) / norm

            spectrum_aligned = self.phase_displace(f, s, p/fcenter )

            return spectrum_aligned, p

        #Find the spectrum with greatest weight
        weights = [np.sum(np.abs(spectrum)**2)
                   for spectrum in spectra]
        ind0 = np.argmax(weights)
        if verbose: print('Primary index for phase alignment:',ind0)

        self.phase_alignments= [0] * len(spectra)
        spectra_aligned=[0]*len(spectra)
        spectra_aligned[ind0] = spectrum_cmp = spectra[ind0]
        total_count=1

        #Count backward, then forward
        index_list = np.append( np.arange(ind0-1,-1,-1),
                                np.arange(ind0+1,len(spectra),1) )
        phase_alignment_cmp = 0
        for i in index_list:
            spectrum_aligned, phase_alignment_cmp = aligned_spectrum(spectrum_cmp,
                                                                    spectra[i],
                                                                     phase_alignment_cmp)
            spectra_aligned[i] = spectrum_aligned
            self.phase_alignments[i] = phase_alignment_cmp
            if i==ind0+1: phase_alignment_cmp = 0 #Re-set if we're now comparing to `ind0`
            total_count+=1

            #When we hit the first, re-set our reference spectrum
            if i==0: spectrum_cmp=spectra[ind0]
            else: spectrum_cmp = spectrum_cmp+spectrum_aligned

        assert total_count==len(spectra)

        return spectra_aligned

    def align_and_envelope_spectra(self, spectra, spectra_ref,
                                   apply_envelope=True, envelope_width=1,
                                   BB_phase=False,
                                   smoothing=None,window=np.blackman,
                                   **kwargs):

        spectra = np.array(spectra)
        spectra_ref = np.array(spectra_ref)

        Nrows = 5
        assert len(spectra) == len(spectra_ref), \
            "Applying envelope requires same number of accumulations for both analyte and reference"

        assert len(spectra) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % Nrows
        assert len(spectra_ref) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % Nrows
        Nspectra = np.max((len(spectra) // Nrows, 1))

        # Establish some mutual frequency axis
        f0 = None

        # Collect all the spectra and the envelopes
        ss = []
        ss_ref = []
        env_sets = []
        for i in range(Nspectra):

            ##----- Reference spectrum components
            f_ref,sabs_ref,sphase_ref = spectra_ref[Nrows * i:Nrows * i+3]
            s_ref = sabs_ref * np.exp(1j * sphase_ref)

            # In case we have invalid entries (zeros), remove them
            where_valid = np.isfinite(f_ref) * (f_ref > 0)
            if not where_valid.any():
                print('Reference spectrum at entry %i is inferred empty!' % i)
                continue
            f_ref = f_ref[where_valid]
            s_ref = s_ref[where_valid]

            # Establish some mutual frequency axis
            if f0 is None: f0=f_ref #In this way, even if analyte spectra are empty, we have a definite frequency axis

            # Compute envelope
            env_set = spectra_ref[Nrows * i + 4][:3]
            if apply_envelope:
                env = spectral_envelope(f0, *env_set,
                                        expand_envelope = envelope_width)
                env /= env.max()

            # Touch up the spectrum in all the ways
            # Smooth before applying envelope
            if window:
                s_ref = self.windowed_spectrum(f_ref, s_ref, window=window)
            if smoothing:
                s_ref = self.smoothed_spectrum(f_ref, s_ref,smoothing=smoothing)
            s_ref = self.interpolate_spectrum(s_ref,f_ref,f0)
            if apply_envelope: s_ref *= env

            # Discard phase if we don't want it
            if not BB_phase:
                s_ref = np.abs(s_ref)

            ##-------- Analyte spectrum components
            f,sabs,sphase = spectra[Nrows * i:Nrows * i+3]
            s = sabs * np.exp(1j * sphase)

            # In case we have invalid entries (zeros), remove them
            where_valid = np.isfinite(f) * (f > 0)
            if not where_valid.any():
                print('Analyte spectrum at entry %i is inferred empty!' % i)
                continue
            f = f[where_valid]
            s = s[where_valid]

            # Touch up the spectrum in all the ways
            # Smooth before applying envelope
            if window:
                s = self.windowed_spectrum(f, s, window=window)
            if smoothing:
                s = self.smoothed_spectrum(f, s,smoothing=smoothing)
            s = self.interpolate_spectrum(s,f,f0)
            if apply_envelope: s *= env

            # We're done
            env_sets.append(env_set)
            ss.append(s)
            ss_ref.append(s_ref)

        # If we didn't process any reference spectra (at least) then we have an error
        if f0 is None: raise ValueError('All spectral data were supplied empty!')

        # Sort by center frequency
        fcenters = np.array([env_set[1] for env_set in env_sets])
        ordering = np.argsort(fcenters)
        ss = [ss[idx] for idx in ordering]
        ss_ref = [ss_ref[idx] for idx in ordering]

        #Determine global leveler that removes average x-coordinate of all interferograms
        if len(ss):
            S = np.sum(ss,axis=0)
            self.global_leveler = self.level_phase(f0,S,order=1,return_leveler=True)
            ss = [s * self.global_leveler for s in ss]
        else: self.global_leveler = np.ones(f0.shape) #In case there is no data to level

        #Spectra could all be empty, though f0 will not be
        return f0, ss, ss_ref

    ###########
    #- User API
    ###########

    Nrows = 5

    def __init__(self,
                 sample_spectra,
                 sample_BB_spectra,
                 ref_spectra,
                 ref_BB_spectra):

        assert len(sample_spectra) == len(sample_BB_spectra), \
            "We require the same number of spectrum accumulations for both sample and sample bright-beam!"
        assert len(sample_spectra) % self.Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % self.Nrows
        assert len(sample_BB_spectra) % self.Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % self.Nrows

        assert len(ref_spectra) == len(ref_BB_spectra), \
            "We require the same number of spectrum accumulations for both reference and reference bright-beam!"
        assert len(ref_spectra) % self.Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % self.Nrows
        assert len(ref_BB_spectra) % self.Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % self.Nrows

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
                         apply_envelope=True,
                         envelope_width=1,
                         valid_thresh=.01,
                         smoothing=None,
                         window=None,
                         align_phase=True,
                         BB_normalize=True,
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
            s_empty = np.zeros(f0.shape,dtype=np.float)
            return f0,s_empty,s_empty

        # Align phase only if commanded, store uncorrected version as `spectra0` for comparison
        if align_phase:
            print('Aligning phase...')
            spectra0 = spectra
            spectra = self.phase_aligned_spectrum(f0, spectra0, verbose=False)

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
                plt.xlim(500, 2500)

                # Now plot directly the corrected phases
                plt.subplot(212)
                phases0 = self.get_phases(f0,spectra0,
                                                     threshold=.1,
                                                     ps=None,
                                                     level=True,order=view_phase_alignment_leveling)
                phases = self.get_phases(f0,spectra,
                                                     threshold=.1,
                                                     ps=None,
                                                    level=True,order=view_phase_alignment_leveling)

                label0='Uncorrected'; label='Corrected'
                for i in range(len(phases0)):
                    l,=plt.plot(f0,phases0[i],ls='--',label=label0)
                    plt.plot(f0,phases[i],color=l.get_color(),label=label)
                    label0=label=None
                plt.xlabel('Frequency')
                plt.ylabel('Phase ')
                plt.legend(loc='best')

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

    def __call__(self,
                 optimize_BB=False,
                 align_phase=True,
                 apply_envelope=True,
                 envelope_width=1,
                 valid_thresh=.01,
                 smoothing=None,window=None,
                 view_phase_alignment=False,
                 BB_normalize=True,BB_phase=False,
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
                                                                    apply_envelope=apply_envelope,
                                                                    envelope_width=envelope_width,
                                                                    valid_thresh=valid_thresh,
                                                                    smoothing=None,window=window,
                                                                    align_phase=align_phase,
                                                                    view_phase_alignment=view_phase_alignment,
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
                                                                   apply_envelope=apply_envelope,
                                                                   envelope_width=envelope_width,
                                                                   valid_thresh=valid_thresh,
                                                                   smoothing=None,window=window,
                                                                   align_phase=align_phase,
                                                                   view_phase_alignment=view_phase_alignment,
                                                                   BB_normalize=BB_normalize,BB_phase=BB_phase,
                                                                   **kwargs)
        #If sample spectrum is non-empty, do normalization
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

        """#If sample spectrum is empty (allowed), just return 0
        else:
            self.f_sample = self.f_norm = self.f_ref
            self.sample_spectrum_abs = self.norm_spectrum_abs = np.zeros(self.f_ref.shape)
            self.sample_spectrum = self.norm_spectrum = np.zeros(self.f_ref.shape)"""

        return self.f_norm, self.norm_spectrum_abs, self.norm_spectrum


def accumulate_spectra(spectra, apply_envelope=True, expand_envelope=1):

    try:
        return SpectralProcessor.accumulate_spectra(spectra,
                                                    apply_envelope=apply_envelope,
                                                    expand_envelope=expand_envelope)

    except:
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'accumulate_spectra.err')
        with open(error_file,'w') as f:
            f.write(error_text)

        raise

def heal_linescan(linescan):

    print('Healing linescan!')

    linescan = np.array(linescan,dtype=float)
    Nrows = SpectralProcessor.Nrows

    isempty = lambda arr: np.isclose(arr,0).all()

    import copy
    healed_linescan = copy.copy(linescan)

    for i in range(len(healed_linescan)):

        # We need nonzero data from 'nextnext' and preferably also 'next' pixels
        if i <= len(healed_linescan) - 3:
            Nspectra = len(healed_linescan[i]) // Nrows

            for j in range(Nspectra):
                j1, j2 = j * Nrows, (j + 1) * Nrows
                shere = healed_linescan[i][j1:j2]

                if isempty(shere):
                    print('Spectrum at pixel %i, accumulation %i is empty!'%(i,j))
                    snext = healed_linescan[i + 1][j1:j2]
                    snextnext = healed_linescan[i + 2][j1:j2]

                    # Fill hole with data from next pixel, and fill next pixel with `next-nextnext` average
                    if not isempty(snext):
                        print("We're fixing it!")
                        healed_linescan[i][j1:j2] = snext  # Assume data for `here` landed at next pixel
                        if not isempty(snextnext):
                            healed_linescan[i + 1][j1:j2] = (snext + snextnext) / 2  # Fill (false) next pixel spectrum with an average
                    elif not isempty(snextnext):
                        print("We're fixing it!")
                        healed_linescan[i][j1:j2] = snextnext  # Assume data for `here` landed two pixels away

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
                           valid_thresh=.01):

    try:
        SP=SpectralProcessor(spectra,spectra_BB,
                             spectra_BB,spectra_BB)

        # `BB_normalize=False` is key, to compute only the envelope from BB,
        # and then to normalize spectra directly to BB as though it were reference
        f, spectrum_abs, spectrum = SP(apply_envelope=apply_envelope, envelope_width=envelope_width,
                                     smoothing=smoothing,
                                     valid_thresh=valid_thresh,
                                     window=False,
                                     BB_normalize=False,
                                     align_phase=align_phase,
                                     view_phase_alignment=False)

        phase = SP.get_phase(f, spectrum, level_phase=True)

        return np.array([f,
                         spectrum_abs.real,
                         phase.real],dtype=np.float)
    except:
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'BB_referenced_spectrum.err')
        with open(error_file,'w') as f:
            f.write(error_text)

        raise

def BB_referenced_linescan(linescan,spectra_BB,
                          apply_envelope=True, envelope_width=1,
                           smoothing=None,align_phase=False,
                           valid_thresh=.01):

    try:

        fmutual = None
        SP = None
        spectra_abs = []; phases=[]
        for spectra in linescan:

            if SP is None:
                SP=SpectralProcessor(spectra,spectra_BB,
                                     spectra_BB,spectra_BB)
            else:
                SP.sample_spectra = spectra #re-set sample spectra, only

            # `BB_normalize=False` is key, to use only the envelope from BB, but overall normalize spectra directly to BB
            f, spectrum_abs, spectrum = SP(apply_envelope=apply_envelope, envelope_width=envelope_width,
                                         smoothing=smoothing,
                                         valid_thresh=valid_thresh,
                                         BB_normalize=False,
                                         align_phase=align_phase,
                                         view_phase_alignment=False,
                                         recompute_reference=False) #`recompute_reference=False` saves us half our effort
            phase = SP.get_phase(f, spectrum, level_phase=True)

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
                         phases.real],dtype=np.float)
    except:
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'BB_referenced_linescan.err')
        with open(error_file,'w') as f:
            f.write(error_text)

        raise

def normalized_spectrum(sample_spectra, sample_BB_spectra,
                        ref_spectra, ref_BB_spectra,
                        apply_envelope=True, envelope_width=1,
                        level_phase=False, align_phase=True,
                        phase_offset=0,smoothing=None, valid_thresh=.01,
                        piecewise_flattening=0,
                        zero_phase_interval=None):

    try:
        SP = SpectralProcessor(sample_spectra, sample_BB_spectra,
                               ref_spectra, ref_BB_spectra)

        f, snorm_abs,snorm = SP(apply_envelope=apply_envelope, envelope_width=envelope_width,
                                smoothing=smoothing,
                                valid_thresh=valid_thresh,
                                window=False,
                                align_phase=align_phase,
                                view_phase_alignment=False)

        if level_phase: snorm = SP.level_phase(f,snorm,order=1,manual_offset=None,weighted=True)

        elif piecewise_flattening:
            phase = SP.get_phase(f, snorm, level_phase=False)
            to_fit = phase
            offset,params = numrec.PiecewiseLinearFit(f,to_fit,nbreakpoints=piecewise_flattening,error_exp=2)
            phase -= offset
            snorm = snorm_abs*np.exp(1j*phase)

        #Now finally apply manual offset
        phase=SP.get_phase(f, snorm, level_phase=True,
                           order=0, manual_offset=phase_offset) #Only level using the offset we apply

        # Try leveling phase across a particular range of energies
        if level_phase and hasattr(zero_phase_interval, '__len__') \
                and len(zero_phase_interval) >= 2:

            fmin, fmax = np.min(zero_phase_interval), np.max(zero_phase_interval)
            f0 = np.mean((fmin, fmax))
            interval = (f > fmin) * (f < fmax)
            if interval.any():  # Only do anything if interval turns out to have data
                pval = np.mean(phase[interval])  # average phase inside freq interval
                pcorr = f / f0 * (-pval) #This will zero by the average value inside interval
                phase += pcorr

        return np.array([f,
                         snorm_abs.real,
                         phase.real], dtype=np.float)

    except:

        #--- Dump the error
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'normalized_spectrum.err')
        with open(error_file,'w') as f: f.write(error_text)

        return False

def normalized_linescan(sample_linescan, sample_BB_spectra,
                        ref_spectra, ref_BB_spectra,
                        apply_envelope=True, envelope_width=1,
                        level_phase=False, align_phase=True,
                        phase_offset=0, smoothing=None, valid_thresh=.01,
                        piecewise_flattening=0,
                        zero_phase_interval=None,
                        heal_linescan=False):

    global SP

    sample_linescan = np.array(sample_linescan)
    sample_BB_spectra = np.array(sample_BB_spectra)

    if heal_linescan:  sample_linescan = heal_linescan_recursive(sample_linescan)

    try:

        fmutual = None
        SP = None
        snorms_abs = []; phases=[]
        #Iterate through spatial pixels
        for i in range(len(sample_linescan)):

            print('Processing linescan pixel %i of %i...'%(i+1,len(sample_linescan)))
            sample_spectra = sample_linescan[i]

            #--- Compute normalized spectrum at this pixel
            if SP is None: # Initialize spectral processor
                SP = SpectralProcessor(sample_spectra, sample_BB_spectra,
                                       ref_spectra, ref_BB_spectra)
            else:
                SP.sample_spectra = sample_spectra #re-set sample spectra, only

            f, snorm_abs, snorm = SP(apply_envelope=apply_envelope, envelope_width=envelope_width,
                                     smoothing=smoothing,
                                     valid_thresh=valid_thresh,
                                     window=False,
                                     align_phase=align_phase,
                                     view_phase_alignment=False,
                                     recompute_reference=False) #`recompute_reference=False` saves us half our effort

            #--- Do all the phase leveling
            if level_phase:
                snorm = SP.level_phase(f, snorm, order=1, manual_offset=None, weighted=False)

            # Now get phase
            phase = SP.get_phase(f, snorm, level_phase=True,
                                 order=0, manual_offset=phase_offset)  # Order 0 means only offset is used

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

            # Set phase within some frequency interval to zero,
            #  optimizing for flatness up to an arbitrary factor of 2pi
            #  Which factor of 2pi is best?  Whichever minimizes the flatness residual
            if hasattr(zero_phase_interval, '__len__') \
                 and len(zero_phase_interval) >= 2:

                fmin,fmax = np.min(zero_phase_interval), np.max(zero_phase_interval)
                f0 = np.mean((fmin,fmax))
                interval = (fmutual>fmin)*(fmutual<fmax)

                N2pis = 100
                n2pis = np.arange(-N2pis, N2pis+1, 1)
                n2pis = n2pis[np.newaxis, :]  # Try leveling into different `2pi` regimes

                if interval.any(): #Only do anything if interval turns out to have data
                    for n,phase in enumerate(phases): #iterate over points
                        include = interval * (~np.isclose(snorms_abs[n],0)) #disregard empty data points
                        if not include.any(): continue
                        pval = np.mean(phase[include])  # average phase inside freq interval
                        pcorr = fmutual[:,np.newaxis] / f0 * (2*np.pi*n2pis - pval) - 2*np.pi*n2pis #This will make phase in interval equal to some multiple of 2pi
                        phase_options = phase[:,np.newaxis] + pcorr
                        residuals = np.sum( (phase_options[include])**2, axis=0) #Minimize the average slope
                        phases[n] = phase_options[:,np.argmin(residuals)]

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
                    phase += offset * fmutual/f0

        return np.array([fmutuals.real,
                         snorms_abs.real,
                         phases.real],dtype=np.float)
    except:
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'normalized_linescan.err')
        with open(error_file,'w') as f:
            f.write(error_text)

        raise

def save_data(path,data):

    data=np.array(data) #Assume array data type for now

    import pickle
    with open(path,'wb') as f:
        pickle.dump(data,f)

    return 'Done'


def load_data(path):

    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)

    data=np.array(data) #Assume array data type for now

    return data