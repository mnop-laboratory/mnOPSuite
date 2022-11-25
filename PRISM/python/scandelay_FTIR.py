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

    global dX,best_delay,xaxis
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
        if i - Ncycles == -1: print(params[2:])
        all_xs = model_xs(indices, params)

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
            offset = np.mean(xs); period = float(Nsamples)
            amps = [0]*fit_xs_order; amps[0]=np.ptp(xs)/2
            phases=[0]*fit_xs_order
            params0 = [offset,period]
            for amp,phase in zip(amps,phases): params0 = params0+[amp,phase]
            params,_ = numrec.ParameterFit(indices,xs,model_xs,params0)
            if i-Ncycles==-1: print(params[2:])
            xs = model_xs(indices,params)

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

    envelope_params = leastsq(to_minimize,x0)[0]

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
    # w = np.blackman(len(v))
    #wmax_ind = np.argmax(w) #Let's roll window to position of maximum weight in interferogram
    #centerd = np.sum(d * v ** 2) / np.sum(v ** 2)
    #imax_ind = np.argmin((centerd - d) ** 2)
    #w = np.roll(w, imax_ind - wmax_ind, axis=0)
    w = 1 #We are disabling the windowing for now

    v *= w
    v = AWA(v, axes=[d])
    s = num.Spectrum(v, axis=0)
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
                                               valid_thresh=valid_thresh, abs=True)
        d1 = np.gradient(s_abs_norm) / s_abs_norm

        return np.mean(np.abs(d1) ** (1 / 2))  # minimize first derivative

    ##################################
    # - Phase alignment methods
    ##################################
    @staticmethod
    def phase_displace(f, s, p):

        return np.exp(1j * f * p) * s

    @staticmethod
    def get_intfg(f,scomplex):

        faxis = np.append(-f[::-1], f) #We assume we have only positive frequencies, but need negative ones too
        s = np.append(np.conj(scomplex[::-1]), scomplex)
        s = num.Spectrum(AWA(s, axes=[faxis], axis_names=['X Frequency']))

        ## Compute and window interferogram
        intfg = s.get_inverse().real
        x=intfg.axes[0]
        intfg = np.array(intfg)

        return x,intfg

    @classmethod
    def level_phase(cls,f, s, order=1, manual_offset=0, return_leveler=False):

        assert np.all(np.isfinite(s))

        leveler = 1
        if order:
            #Begin by de-shifting interferogram
            x,intfg = cls.get_intfg(f,s)
            x0 = np.sum(x*intfg**2)/np.sum(intfg**2)
            leveler *= np.exp(2*np.pi*1j*f*x0) #plane waves are in a basis `exp(-1j*x...)`, so we are shifting by `-x0`

            #Loop until we can flatten the phase no further
            while True:
                p = np.unwrap(np.angle( s*leveler ))
                m = list(np.polyfit(x=f, y=p, deg=order, w=np.abs(s)**2))
                if order == 1:  m[-1]=0 #no overall phase offsets, if we are restricting to physical offsets
                pcorr = -np.polyval(m, f)
                leveler *= np.exp(1j*pcorr) #Update leveler
                if np.isclose(np.sum(m[:-1]),0): break #If we can update non-constant offsets no further, break

        if manual_offset:
            pcorr = manual_offset * f/np.mean(np.abs(f))
            leveler *= np.exp(1j*pcorr)

        if return_leveler: return leveler
        else: return s*leveler

    @classmethod
    def get_phase(cls,f,s,
                  level_phase=True,order=1,
                  manual_offset=0):

        if level_phase:
            s=cls.level_phase(f,s,order=order,
                              manual_offset=manual_offset)

        p = np.unwrap(np.angle(s))

        #Remove as many factors of 2*pi so that greatest intensity region is closest to zero
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

        phase_avg = np.nanmean(phases,axis=0)
        keep = np.isfinite(phase_avg)
        poly = np.polyfit(f[keep],phase_avg[keep],deg=order)
        background = np.polyval(poly,f)
        for i in range(len(phases)): phases[i]-=background

        return phases

    @classmethod
    def interpolate_spectrum(cls,s,f1,f2,order=1):
        #Regular interpolation of a spectrum, but first removing overall phase slope, which can be smeared by interpolation

        # Find overall phase slope
        leveler = cls.level_phase(f1,s,order=order,manual_offset=0,return_leveler=True)

        result = interp1d(x=f1,y=s*leveler, **interp_kwargs)(f2)

        leveler = interp1d(x=f1,y=leveler,**interp_kwargs)(f2)

        result /= leveler #re-impose overall phase slope

        result[~np.isfinite(result)]=0

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
    def normalize_spectrum(cls, f_s, spectra,
                           f_r, spectra_ref,
                           abs=False, abs_ref=True,
                           valid_thresh=.01):

        spectrum = cls.summed_spectrum(spectra, abs=abs)
        spectrum_ref = cls.summed_spectrum(spectra_ref, abs=abs_ref)

        f_mutual = np.unique(np.append(f_s,f_r)) #Join and sort the frequencies
        from scipy.signal import decimate
        f_mutual = decimate(f_mutual,q=2,n=1) #down-sample by a factor of 2
        #f_mutual = f_s
        #spectrum = interp1d(x=f_s,y=spectrum,bounds_error=False,fill_value=0,kind='linear')(f_mutual)
        #spectrum_ref = interp1d(x=f_r,y=spectrum_ref,bounds_error=False,fill_value=0,kind='linear')(f_mutual)
        spectrum = cls.interpolate_spectrum(spectrum,f_s,f_mutual)
        spectrum_ref = cls.interpolate_spectrum(spectrum_ref,f_r,f_mutual)

        thresh = valid_thresh * np.abs(spectrum_ref[np.isfinite(spectrum_ref)]).max()
        where_valid = np.abs(spectrum_ref) > thresh
        snorm = (spectrum / spectrum_ref)[where_valid]
        f_mutual = f_mutual[where_valid]

        return f_mutual, snorm

    @classmethod
    def accumulate_spectra(cls, spectra,
                           apply_envelope=True,
                           expand_envelope=1):

        spectra = np.array(spectra)

        Nrows = 5

        assert len(spectra) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % Nrows
        Nspectra = np.max((len(spectra) // Nrows, 1))

        # Establish some mutual frequency axis
        all_fs = np.array([spectra[Nrows * i] for i in range(Nspectra)])
        f0 = spectra[0] #all_fs[0] #np.unique(all_fs.flatten())

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
            f = f[where_valid]
            s = s[where_valid]

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
                                       level_phase=True)

        return np.array([f0, spectrum_abs, spectrum_phase],dtype=np.float)

    window_order=1

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

        leveler = cls.level_phase(f,scomplex,order=order,return_leveler=True)

        scomplexl = scomplex*leveler

        scomplexl = numrec.smooth(scomplexl,window_len=smoothing,axis=0)

        return scomplexl / leveler

    coincidence_exponent=2

    def phase_aligned_spectrum(self, f, spectra, verbose=False,
                               coincidence_exponent=None):

        if coincidence_exponent is None: coincidence_exponent=self.coincidence_exponent
        #self.s0s=[] #Debugging - check the operands for phase alignments

        def aligned_spectrum(s0, s, p_cmp=0):

            #Determine and remove average phase of stot, since it could be wrapping like crazy
            leveler = self.level_phase(f, s0, order=1, manual_offset=0, return_leveler=True)

            # Debugging - check the leveling
            #self.s0s.append(num.Spectrum(AWA(s0 * leveler, axes=[f], axis_names=['X Frequency']), axis=0))

            phase0 = self.get_phase(f, s0 * leveler, manual_offset=0,
                                    level_phase=False)
            phase = self.get_phase(f, s * leveler, manual_offset=0,
                                   level_phase=False)
            coincidence = (np.abs(s0) * np.abs(s)) ** coincidence_exponent
            norm = np.sum(coincidence)

            pdiff = np.sum((phase0 - phase) * coincidence) / norm
            f0 = np.sum(f * coincidence) / norm

            # Decide how much of phase difference allocates to `2*pi` modulus and how much to physical shift
            npis = np.round( (pdiff-p_cmp)/(2*np.pi))
            pdiff = pdiff - npis*2*np.pi

            spectrum_aligned = self.phase_displace(f, s, pdiff/f0)

            return spectrum_aligned, pdiff

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
            spectrum_aligned, phase_alignment = aligned_spectrum(spectrum_cmp,
                                                                 spectra[i],
                                                                 phase_alignment_cmp)
            spectra_aligned[i] = spectrum_aligned
            self.phase_alignments[i] = phase_alignment
            total_count+=1
            if i==ind0+1: phase_alignment_cmp=0 #re-set, we compare to the `ind0` spectrum
            else: phase_alignment_cmp=phase_alignment

            #When we hit the first, re-set our reference spectrum
            if i==0: spectrum_cmp=spectra[ind0]
            else: spectrum_cmp = spectrum_aligned

        assert total_count==len(spectra)

        return spectra_aligned

    #@TODO: remove the BB-optimization
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
        all_fs = np.array([spectra[Nrows * i] for i in range(Nspectra)]+
                           [spectra_ref[Nrows * i] for i in range(Nspectra)])
        f0 = spectra[0] #all_fs[0]#np.unique(all_fs.flatten())

        # Collect all the spectra and the envelopes
        ss = []
        ss_ref = []
        env_sets = []
        for i in range(Nspectra):

            ##-------- Analyte spectrum components
            f,sabs,sphase = spectra[Nrows * i:Nrows * i+3]
            s = sabs * np.exp(1j * sphase)

            # In case we have invalid entries (zeros), remove them
            where_valid = np.isfinite(f) * (f > 0)
            f = f[where_valid]
            s = s[where_valid]
            s = self.interpolate_spectrum(s,f,f0)

            # Touch up the spectrum in all the ways
            if window:
                s = self.windowed_spectrum(f0, s, window=window)
            if smoothing:
                s=self.smoothed_spectrum(f0,s,smoothing=smoothing)

            # We're done
            ss.append(s)

            ##----- Reference spectrum components
            f_ref,sabs_ref,sphase_ref = spectra_ref[Nrows * i:Nrows * i+3]
            s_ref = sabs_ref * np.exp(1j * sphase_ref)

            # In case we have invalid entries (zeros), remove them
            where_valid = np.isfinite(f_ref) * (f_ref > 0)
            f_ref = f_ref[where_valid]
            s_ref = s_ref[where_valid]
            s_ref = self.interpolate_spectrum(s_ref,f_ref,f0)

            # Touch up the spectrum in all the ways
            if window:
                s_ref = self.windowed_spectrum(f0, s_ref, window=window)
            if smoothing:
                s_ref=self.smoothed_spectrum(f0,s_ref,smoothing=smoothing)

            # Discard phase if we don't want it
            if not BB_phase:
                s_ref = np.abs(s_ref)
            # We're done
            ss_ref.append(s_ref)

            # Envelope components and apply them
            env_set = spectra_ref[Nrows * i + 4][:3]
            env_sets.append(env_set)
            if apply_envelope:
                env = spectral_envelope(f0, *env_set,
                                        expand_envelope = envelope_width)
                env /= env.max()
                ss_ref[-1] *= env
                ss[-1] *= env

        # Sort by center frequency
        fcenters = [env_set[1] for env_set in env_sets]
        ordering = np.argsort(fcenters)
        ss = [ss[idx] for idx in ordering]
        ss_ref = [ss_ref[idx] for idx in ordering]

        return f0, ss, ss_ref

    ###########
    #- User API
    ###########

    def __init__(self,
                 sample_spectra,
                 sample_BB_spectra,
                 ref_spectra,
                 ref_BB_spectra):

        self.Nrows = Nrows = 5
        assert len(sample_spectra) == len(sample_BB_spectra), \
            "We require the same number of spectrum accumulations for both sample and sample bright-beam!"
        assert len(sample_spectra) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % Nrows
        assert len(sample_BB_spectra) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % Nrows

        assert len(ref_spectra) == len(ref_BB_spectra), \
            "We require the same number of spectrum accumulations for both reference and reference bright-beam!"
        assert len(ref_spectra) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % Nrows
        assert len(ref_BB_spectra) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % Nrows

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
                                                                  valid_thresh=valid_thresh)

        # Apply correction factor (as from detector calibration) if provided
        if hasattr(self,'%s_factors'%target):
            print('ya')
            factors=getattr(self,'%s_factors'%target)
            for s,factor in zip(spectra,factors): s *= factor

        self.f0 = f0
        self.processed_spectra = spectra
        self.processed_spectra_BB = spectra_BB

        # Align phase only if commanded, store uncorrected version as `spectra0` for comparison
        if align_phase:
            print('Aligning phase...')
            phase_alignment_method = self.phase_aligned_spectrum
            spectra0 = spectra
            spectra = phase_alignment_method(f0, spectra0, verbose=False)

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
            f, spectrum_abs = self.normalize_spectrum(f0, spectra,
                                                  f0, spectra_BB,
                                                  valid_thresh=valid_thresh,
                                                  abs=True,
                                                  abs_ref=True) #discard phase of BB

            f, spectrum = self.normalize_spectrum(f0, spectra,
                                                  f0, spectra_BB,
                                                  valid_thresh=valid_thresh,
                                                  abs=False,
                                                  abs_ref=True) #discard phase of BB
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
                 smoothing=None,window=np.blackman,
                 view_phase_alignment=False,
                 BB_normalize=True,
                 **kwargs):
        """This normalizes sample and reference spectra to their
        respective bright-beam spectra, and then normalizes these
        two BB-normalized spectra to each other."""

        print('Processing sample spectra...')
        self.f_sample,self.sample_spectrum_abs,self.sample_spectrum\
                                            =self.process_spectrum(target='sample',
                                                                   apply_envelope=apply_envelope,
                                                                   envelope_width=envelope_width,
                                                                   valid_thresh=valid_thresh,
                                                                   smoothing=smoothing,window=window,
                                                                   align_phase=align_phase,
                                                                   view_phase_alignment=view_phase_alignment,
                                                                   BB_normalize=BB_normalize,
                                                                   **kwargs)

        print('Processing reference spectra...')
        self.f_ref,self.ref_spectrum_abs,self.ref_spectrum\
                                        = self.process_spectrum(target='reference',
                                                                apply_envelope=apply_envelope,
                                                                envelope_width=envelope_width,
                                                                valid_thresh=valid_thresh,
                                                                smoothing=smoothing,window=window,
                                                                align_phase=align_phase,
                                                                view_phase_alignment=view_phase_alignment,
                                                                   BB_normalize=BB_normalize,
                                                                **kwargs)

        self.f_norm_abs, self.norm_spectrum_abs = self.normalize_spectrum(self.f_sample, [self.sample_spectrum_abs],
                                                                            self.f_ref, [self.ref_spectrum_abs],
                                                                            abs=True, abs_ref=True,
                                                                             valid_thresh=valid_thresh)
        self.f_norm, self.norm_spectrum = self.normalize_spectrum(self.f_sample, [self.sample_spectrum],
                                                                  self.f_ref, [self.ref_spectrum],
                                                                  abs=False, abs_ref=False,
                                                                  valid_thresh=valid_thresh)

        self.norm_spectrum_abs = self.interpolate_spectrum(self.norm_spectrum_abs,
                                                           self.f_norm_abs,self.f_norm).real

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

def BB_referenced_spectrum(spectra,spectra_BB,
                          apply_envelope=True, envelope_width=1,
                           smoothing=None,align_phase=False,
                           valid_thresh=.01):

    try:
        SP=SpectralProcessor(spectra,spectra_BB,
                             [],[]) #Leave reference spectra empty

        f, spectrum_abs, spectrum =\
                        SP.process_spectrum(target='sample',
                                            apply_envelope=apply_envelope,
                                            envelope_width=envelope_width,
                                            valid_thresh=valid_thresh,
                                            smoothing=smoothing,window=False,
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

def normalized_spectrum(sample_spectra, sample_BB_spectra,
                        ref_spectra, ref_BB_spectra,
                        apply_envelope=True, envelope_width=1,
                        level_phase=False, align_phase=True,
                        phase_offset=0,smoothing=None, valid_thresh=.01,
                        piecewise_flattening=0):

    try:
        SP = SpectralProcessor(sample_spectra, sample_BB_spectra,
                               ref_spectra, ref_BB_spectra)

        f, snorm_abs,snorm = SP(apply_envelope=apply_envelope, envelope_width=envelope_width,
                                smoothing=smoothing,
                                valid_thresh=valid_thresh,
                                window=False,
                                optimize_BB=False,
                                align_phase=align_phase,
                                view_phase_alignment=False)

        if level_phase: snorm = SP.level_phase(f,snorm,order=1,manual_offset=None)

        elif piecewise_flattening:
            phase = SP.get_phase(f, snorm, level_phase=False)
            to_fit = phase
            """ if not optimize_phase: to_fit = phase
            else:
                # We want to bring "aligned" phase closer to its unadulterated version via piecewise slopes
                f, _, snorm0 = SP(apply_envelope=apply_envelope, envelope_width=envelope_width,
                                  smoothing=smoothing,
                                 valid_thresh=valid_thresh,
                                 optimize_BB=optimize_BB,
                                 optimize_phase=False,
                                 view_phase_alignment=False)
                phase0 = SP.get_phase(f, snorm0, level_phase=False)
                to_fit = phase - phase0"""

            offset,params = numrec.PiecewiseLinearFit(f,to_fit,nbreakpoints=piecewise_flattening,error_exp=2)
            phase -= offset
            snorm = snorm_abs*np.exp(1j*phase)

        #Now finally apply manual offset
        phase=SP.get_phase(f, snorm, level_phase=True,
                           order=0, manual_offset=phase_offset) #Only level using the offset we apply

        # `get_phase` will apply some phase leveling
        return np.array([f,
                         snorm_abs.real,
                         phase.real], dtype=np.float)

    except:

        #--- Dump the error
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'normalized_spectrum.err')
        with open(error_file,'w') as f: f.write(error_text)

        return False
