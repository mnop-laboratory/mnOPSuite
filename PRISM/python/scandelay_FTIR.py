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
        result =align_interferograms_new(intfgs_arr, delay_calibration_factor=-2.5,
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

def align_interferograms_old(intfgs_arr, delay_calibration_factor=1,
                             shift0=0, optimize_shift=True, shift_range=15,
                             flattening_order=20, noise=0):
    #At a scan rate of 250kHz, a 12-point delay corresponds to only about 50 microseconds

    global dX,best_delay
    from common import numerical_recipes as numrec

    intfgs_arr = np.array(intfgs_arr)
    Ncycles = len(intfgs_arr) // 2
    Nsamples = intfgs_arr.shape[1]
    x_smoothing = np.min((Nsamples/100,50))

    all_xs = []
    all_ys = []
    for i in range(Ncycles):
        # i=i+10
        ys, xs = np.array(intfgs_arr[2 * i:2 * (i + 1)]).copy()
        if i==0: Nbins = len(xs)

        xs *= delay_calibration_factor
        xs = numrec.smooth(xs, window_len=x_smoothing, axis=0)
        ys = flatten_interferogram(xs,ys,flattening_order=flattening_order)
        if noise: ys+=noise*np.random.randn(len(ys))*(ys.max()-ys.min())/2

        all_xs = np.append(all_xs, xs)
        all_ys = np.append(all_ys, ys)

    x1 = np.min(all_xs)
    x2 = np.max(all_xs)
    if dX is None: dX = x2 - x1

    def delayed_intfg(delay):

        #Assume positions are reported in delayed fashion;
        # a negative delay means the detector data precedes the x-position data
        # apply the same shift to x-data to align with detector data
        #  negative delay means roll x-data backwards to correspond to earlier time points
        all_xs_rolled = np.roll(all_xs, +delay, axis=0)
        result = binned_statistic(all_xs_rolled, all_ys, bins=Nbins)
        xnew = result.bin_edges[:-1]
        intfg_new = result.statistic

        keep=np.isfinite(intfg_new)
        xnew=xnew[keep]
        intfg_new=intfg_new[keep]

        return xnew, intfg_new

    #Scan over delays to find most ideal value
    if optimize_shift:
        delays = np.arange(int(shift0 - shift_range),
                           int(shift0 + shift_range), 1)
        sums = []
        for delay in delays:
            xnew, intfg_new = delayed_intfg(delay)
            sums.append(np.sum(intfg_new ** 2))

        shift0 = delays[np.argmax(sums)]

    best_delay=shift0
    print('Optimal shift:', shift0)

    xnew, intfg_new = delayed_intfg(shift0)

    intfg_interp = interp1d(x=xnew,y=intfg_new,
                            **interp_kwargs)
    xnew=np.linspace(x1,x1+dX,len(xnew))
    intfg_new = intfg_interp(xnew)

    return np.array([intfg_new,xnew])

def align_interferograms_new(intfgs_arr, delay_calibration_factor=1,
                         shift0=0,optimize_shift=True,shift_range=15,
                         flattening_order=20,noise=0):

    global dX,best_delay,Nbins
    global intfg_mutual_fwd,intfg_mutual_bwd
    if best_delay is None: best_delay=shift0

    intfgs_arr = np.array(intfgs_arr)
    Ncycles = len(intfgs_arr) // 2
    Nsamples = intfgs_arr.shape[1]
    x_smoothing = np.min((Nsamples / 100, 100))

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

        xs = numrec.smooth(xs, window_len=x_smoothing, axis=0)
        if i == 0:
            idx_turn = np.argmax(xs)

            #We can set `Nbins` for the first time, based on the typical sampling capability for the interferograms
            if Nbins is None:
                Nbins = len(xs) #*np.max( (1,int(Ncycles/5)) )

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
    intfg_mutual_fwd = intfg_fwd.interpolate_axis(xmutual, axis=0)
    intfg_mutual_bwd = intfg_bwd.interpolate_axis(xmutual, axis=0)

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
        print(shift)
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

def align_interferograms_fwd_bwd(intfgs_arr, delay_calibration_factor=1,
                                 flattening_order=20, noise=0):
    #At a scan rate of 250kHz, a 12-point delay corresponds to only about 50 microseconds

    global dX,best_delay,bins,idx,x_smoothing
    from common import numerical_recipes as numrec

    intfgs_arr = np.array(intfgs_arr)
    Ncycles = len(intfgs_arr) // 2
    Nsamples = intfgs_arr.shape[1]
    x_smoothing = np.min((Nsamples/100,50))

    all_xs_fwd = []
    all_ys_fwd = []
    all_xs_bwd = []
    all_ys_bwd = []
    all_xs = []
    for i in range(Ncycles):
        ys, xs = np.array(intfgs_arr[2 * i:2 * (i + 1)]).copy()

        xs = numrec.smooth(xs, window_len=x_smoothing, axis=0)
        if i==0:
            idx = np.argmax(xs) #This is where we switch from forward to backward scan
            Nbins = len(xs)
        ys -= np.polyval(np.polyfit(x=xs, y=ys, deg=flattening_order), xs)
        if noise: ys+=noise*np.random.randn(len(ys))*(ys.max()-ys.min())/2

        xs *= delay_calibration_factor

        all_xs_fwd = np.append(all_xs_fwd, xs[:idx])
        all_ys_fwd = np.append(all_ys_fwd, ys[:idx])

        all_xs_bwd = np.append(all_xs_bwd, xs[idx:])
        all_ys_bwd = np.append(all_ys_bwd, ys[idx:])

        all_xs = np.append(all_xs, xs)

    x1 = np.min(all_xs)
    x2 = np.max(all_xs)
    if dX is None: dX = x2 - x1
    bins=np.linspace(x1,x2,Nbins)

    result = binned_statistic(all_xs_fwd, all_ys_fwd, bins=bins)
    intfg_x_fwd = result.bin_edges[:-1]
    intfg_y_fwd = result.statistic
    intfg_y_fwd[~np.isfinite(intfg_y_fwd)]=0

    result = binned_statistic(all_xs_bwd, all_ys_bwd, bins=bins)
    intfg_x_bwd = result.bin_edges[:-1]
    intfg_y_bwd = result.statistic
    intfg_y_bwd[~np.isfinite(intfg_y_bwd)]=0

    return np.array([intfg_y_fwd,intfg_x_fwd,
                     intfg_y_bwd,intfg_x_bwd])

def align_interferograms_test(intfgs_arr, delay_calibration_factor=1,
                                 flattening_order=20, noise=0,
                                shift0=0,optimize_shift=True,shift_range=7):
    #At a scan rate of 250kHz, a 12-point delay corresponds to only about 50 microseconds

    global dX,best_delay,bins,idx,x_smoothing
    from common import numerical_recipes as numrec

    intfgs_arr = np.array(intfgs_arr)
    Ncycles = len(intfgs_arr) // 2
    Nsamples = intfgs_arr.shape[1]
    x_smoothing = np.min((Nsamples/100,50))

    shifts=np.arange(-int(shift_range),int(shift_range),1)

    intfg=0
    for i in range(Ncycles):
        ys, xs = np.array(intfgs_arr[2 * i:2 * (i + 1)]).copy()

        xs *= delay_calibration_factor
        xs = numrec.smooth(xs, window_len=x_smoothing, axis=0)

        ys -= np.polyval(np.polyfit(x=xs, y=ys, deg=flattening_order), xs)
        if noise: ys+=noise*np.random.randn(len(ys))*(ys.max()-ys.min())/2

        if i==0:
            idx = np.argmax(xs) #This is where we switch from forward to backward scan
            Nbins = len(xs)

        # Select forward scan
        xs=xs[:idx]
        ys=ys[:idx]

        xs,ys = zip( *sorted( zip(xs,ys),
                              key = lambda tup: tup[0]))
        xs=np.array(xs)
        ys=np.array(ys)

        if i==0:
            if dX is None: dX=np.max(xs)-np.min(xs)
            xs_mutual = np.linspace(np.min(xs),np.max(xs),5*Nbins)

        interp=interp1d(xs,ys,**interp_kwargs)
        new_intfg = interp(xs_mutual)

        if not optimize_shift or i==0:
            intfg=intfg+new_intfg
            continue

        diffs=[]
        for shift in shifts:
            test_intfg = np.roll(new_intfg,shift)
            diffs.append(np.sum( (test_intfg - intfg/i)**2 ) ) #Running intfg has been accumulated `i` times
        shift_best = shifts[np.argmin(diffs)]
        intfg+=np.roll(new_intfg,shift_best)

    intfg/=Ncycles #We have accumulated `Ncycles` times

    xs_new = np.linspace(np.min(xs_mutual),
                         np.max(xs_mutual),
                         Nbins)
    interp = interp1d(xs_mutual,intfg,**interp_kwargs)
    intfg_new = interp(xs_new)

    return np.array([intfg_new,xs_new])

#--- Wrapper
def align_interferograms(intfgs_arr, delay_calibration_factor=1,
                         shift0=0,optimize_shift=True,shift_range=15,
                         flattening_order=20,noise=0):

    try:
        result = align_interferograms_new(intfgs_arr, delay_calibration_factor=delay_calibration_factor,
                                         shift0=shift0,optimize_shift=optimize_shift,shift_range=shift_range,
                                         flattening_order=flattening_order,noise=noise)
        #result = align_interferograms_fwd_bwd(intfgs_arr, delay_calibration_factor=delay_calibration_factor,
        #                                         flattening_order=flattening_order,noise=noise)[:2]
        #result = align_interferograms_test(intfgs_arr, delay_calibration_factor=delay_calibration_factor,
        #                                 shift0=shift0,optimize_shift=optimize_shift,shift_range=shift_range,
        #                                 flattening_order=flattening_order,noise=noise)
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

def fourier_xform(a,tsubtract=0,envelope=True,gain=1,fmin=500,fmax=2000):

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

    v *= w
    v = AWA(v, axes=[d])
    s = num.Spectrum(v, axis=0)

    # This applies knowledge that d does not start at zero, but min value is meaningful
    f = s.axes[0]
    pcorr = +2 * np.pi * (np.min(d)) * f
    s *= np.exp(1j * pcorr)

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

    ##########################
    # - General static methods
    ##########################
    @staticmethod
    def summed_spectrum(spectra, abs=False):

        if abs:
            spectrum = np.sum([np.abs(s) for s in spectra], axis=0)
        else:
            spectrum = np.sum([s for s in spectra], axis=0)

        spectrum[~np.isfinite(spectrum)] = 0

        return spectrum

    @classmethod
    def normalize_spectrum(cls, f, spectra,
                           f_ref, spectra_ref,
                           abs=False, abs_ref=True,
                           valid_thresh=.01):

        spectrum = cls.summed_spectrum(spectra, abs=abs)
        spectrum_ref = cls.summed_spectrum(spectra_ref, abs=abs_ref)
        spectrum_ref = interp1d(x=f_ref, y=spectrum_ref,
                                **interp_kwargs)(f)

        thresh = valid_thresh * np.abs(spectrum_ref).max()
        where_valid = np.abs(spectrum_ref) > thresh
        snorm = (spectrum / spectrum_ref)[where_valid]
        f = f[where_valid]

        return f, snorm

    @classmethod
    def accumulate_spectra(cls, spectra,
                           apply_envelope=True,
                           expand_envelope=1,
                           abs=True):

        spectra = np.array(spectra)

        Nrows = 5

        assert len(spectra) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % Nrows
        Nspectra = np.max((len(spectra) // Nrows, 1))

        # Establish some mutual frequency axis

        all_fs = np.append([], [spectra[Nrows * i] for i in range(Nspectra)])
        Nf = len(all_fs) // (2 * Nspectra)
        f0 = np.linspace(np.min(all_fs),
                         np.max(all_fs),
                         Nf)

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

            s = interp1d(f, s, **interp_kwargs)(f0)
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

        return np.array([f0, spectrum_abs, spectrum_phase])

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

    @classmethod
    def align_and_envelope_spectra(cls,spectra, spectra_ref,
                                   apply_envelope=True,expand_envelope=1,
                                   BB_phase=False, valid_thresh=.01,
                                   factor=1, optimize_BB=False,
                                   verbose=True):

        spectra = np.array(spectra)
        spectra_ref = np.array(spectra_ref)

        Nrows = 5
        assert len(spectra) == len(spectra_ref), \
            "Applying envelope requires same number of accumulations for both analyte and reference"

        assert len(spectra) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % Nrows
        assert len(spectra_ref) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors' % Nrows
        Nspectra = np.max((len(spectra) // Nrows, 1))

        # Establish some mutual frequency axis

        all_fs = np.append([], [spectra[Nrows * i] for i in range(Nspectra)] \
                           + [spectra_ref[Nrows * i] for i in range(Nspectra)])
        Nf = len(all_fs) // (2 * Nspectra)
        f0 = np.linspace(np.min(all_fs),
                         np.max(all_fs),
                         Nf)

        # Collect all the spectra and the envelopes
        ss = []
        ss_interp_ref = []
        env_params = []
        for i in range(Nspectra):

            ## Analyte spectrum components
            f = spectra[Nrows * i]
            sabs = spectra[Nrows * i + 1]
            sphase = spectra[Nrows * i + 2]
            s = sabs * np.exp(1j * sphase)

            # In case we have invalid entries (zeros), remove them
            where_valid = np.isfinite(f) * (f > 0)
            f = f[where_valid]
            s = s[where_valid]
            ss.append(interp1d(f, s, **interp_kwargs)(f0))

            ## Reference spectrum components
            f_ref = spectra_ref[Nrows * i]
            sabs_ref = spectra_ref[Nrows * i + 1]
            sphase_ref = spectra_ref[Nrows * i + 2]
            # Discard the BB phase if we don't want it
            if BB_phase:
                s_ref = sabs_ref * np.exp(1j * sphase_ref)
            else:
                s_ref = sabs_ref

            # In case we have invalid entries (zeros), remove them
            where_valid = np.isfinite(f_ref) * (f_ref > 0)
            f_ref = f_ref[where_valid]
            s_ref = s_ref[where_valid]
            ss_interp_ref.append(interp1d(f_ref, s_ref, **interp_kwargs))

            env_params.append(spectra_ref[Nrows * i + 4][:3])

        # Optimize calibration between BB and sample, if called for
        args = (f0, ss, ss_interp_ref, env_params, apply_envelope, expand_envelope,valid_thresh)
        if optimize_BB:
            result = minimize(cls.calibration_to_minimize, (factor,), method='Nelder-Mead', args=args)
            factor = result.x[0]
            if verbose: print('Identified calibration factor:',factor)

        spectra_cal, spectra_BB_cal = cls.apply_frequency_calibration(factor, *args[:-1])

        return f0, spectra_cal, spectra_BB_cal

    ##################################
    # - Phase alignment static methods
    ##################################
    @staticmethod
    def phase_displace(f, s, p):

        return np.exp(1j * f * p) * s

    @staticmethod
    def level_phase(f, s, order=1, manual_offset=0):

        p = np.unwrap(np.angle(s))
        m = np.polyfit(x=f, y=p, deg=order, w=np.abs(s)**2)
        m[-1]=0 #no overall phase offsets
        p -= np.polyval(m, f)

        if manual_offset:
            p += manual_offset * f/np.mean(np.abs(f))

        return np.abs(s) * np.exp(1j * p)

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

        return p

    @classmethod
    def get_phases(cls,f,spectra,
                      threshold=1e-2,
                      ps=None,
                      level=True,order=2,
                      **kwargs):
        """Return the phases corresponding to `spectra`, shifting each by an
        appropriate factor of `2*pi` so that they are quasi=contiguous.
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

    new_phase_alignment = True

    def phase_aligned_spectrum(cls, f, spectra, verbose=False,
                               coincidence_exponent=2):

        def aligned_spectrum(stot,s):

            phase0 = cls.get_phase(f, stot, manual_offset=0,
                                   level_phase=False)
            phase = cls.get_phase(f, s, manual_offset=0,
                                  level_phase=False)
            coincidence = (np.abs(stot) * np.abs(s)) ** coincidence_exponent
            norm = np.sum(coincidence)

            pdiff = np.sum((phase0 - phase) * coincidence) / norm
            f0 = np.sum(f * coincidence) / norm

            # Decide how much of phase difference allocates to `2*pi` modulus and how much to physical shift
            pdiff_pis = np.round(pdiff / (2 * np.pi)) * 2 * np.pi
            phase_alignment = (pdiff - pdiff_pis) / f0
            offset = pdiff_pis + phase_alignment * f
            phase += offset

            spectrum_aligned = cls.phase_displace(f, s, phase_alignment)

            return spectrum_aligned, phase_alignment

        #Find the spectrum with greatest weight
        weights = [np.sum(np.abs(spectrum)**2)
                   for spectrum in spectra]
        ind0 = np.argmax(weights)
        stot = spectra[ind0]
        if verbose: print('Primary index for phase alignment:',ind0)

        cls.phase_alignments=[0]*len(spectra)
        spectra_aligned=[0]*len(spectra)
        spectra_aligned[ind0]=spectra[ind0]
        total_count=1

        #Count backward, then forward
        index_list = np.append( np.arange(ind0-1,-1,-1),
                                np.arange(ind0+1,len(spectra),1) )
        for i in index_list:
            spectrum_aligned, phase_alignment = aligned_spectrum(stot,spectra[i])
            stot = stot+spectrum_aligned
            spectra_aligned[i] = spectrum_aligned
            cls.phase_alignments[i] = phase_alignment
            total_count+=1

        assert total_count==len(spectra)

        return spectra_aligned

    ###########
    #- User API
    ###########

    def __init__(self,
                 sample_spectra,
                 sample_BB_spectra,
                 ref_spectra,
                 ref_BB_spectra):

        Nrows = 5
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

    def process_spectrum(self,target='sample',
                        apply_envelope=True,
                        envelope_width=1,
                        valid_thresh=.01,
                        optimize_BB=False,
                        optimize_phase=True,
                        view_phase_alignment=False,
                         view_phase_alignment_leveling=6):

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
                                                                   expand_envelope=envelope_width,
                                                                   factor=1,
                                                                   optimize_BB=optimize_BB,
                                                                   valid_thresh=valid_thresh)
        self.f0 = f0
        self.processed_spectra = spectra
        self.processed_spectra_BB = spectra_BB

        # Align phase only if commanded, store uncorrected version as `spectra0` for comparison
        if optimize_phase:
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
                phases0 = self.get_phases_of_spectra(f0,spectra0,
                                                     threshold=.1,
                                                     ps=None,
                                                     level=True,order=view_phase_alignment_leveling)
                phases = self.get_phases_of_spectra(f0,spectra,
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

        return f,spectrum_abs,spectrum

    def __call__(self,
                 optimize_BB=False,
                 optimize_phase=True,
                 apply_envelope=True,
                 envelope_width=1,
                 valid_thresh=.01,
                 view_phase_alignment=False,
                 abs_only=False):

        print('Processing sample spectra...')
        self.f_sample,self.sample_spectrum_abs,self.sample_spectrum\
                                            =self.process_spectrum(target='sample',
                                                                optimize_BB=optimize_BB,
                                                                 apply_envelope=apply_envelope,
                                                                envelope_width=envelope_width,
                                                                valid_thresh=valid_thresh,
                                                                optimize_phase=optimize_phase,
                                                                view_phase_alignment=view_phase_alignment)

        print('Processing reference spectra...')
        self.f_ref,self.ref_spectrum_abs,self.ref_spectrum\
                                        = self.process_spectrum(target='reference',
                                                                optimize_BB=optimize_BB,
                                                                 apply_envelope=apply_envelope,
                                                                envelope_width=envelope_width,
                                                                valid_thresh=valid_thresh,
                                                                optimize_phase=optimize_phase,
                                                                view_phase_alignment=view_phase_alignment)

        self.f_norm_abs, self.norm_spectrum_abs = self.normalize_spectrum(self.f_sample, [self.sample_spectrum_abs],
                                                                  self.f_ref, [self.ref_spectrum_abs],
                                                                  abs=True, abs_ref=True, valid_thresh=valid_thresh)
        self.f_norm, self.norm_spectrum = self.normalize_spectrum(self.f_sample, [self.sample_spectrum],
                                                                  self.f_ref, [self.ref_spectrum],
                                                                  abs=False, abs_ref=False, valid_thresh=valid_thresh)

        self.norm_spectrum_abs = interp1d(x=self.f_norm_abs,
                                   y=self.norm_spectrum_abs,
                                   **interp_kwargs)(self.f_norm)

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

        return False

def BB_referenced_spectrum(spectra,spectra_BB,
                          apply_envelope=True, envelope_width=1,
                           optimize_BB=False,
                          optimize_phase=False,valid_thresh=.01,
                           abs_only=True):

    try:
        SP=SpectralProcessor(spectra,spectra_BB,
                             [],[]) #Leave reference spectra empty

        f, spectrum_abs, spectrum =\
                        SP.process_spectrum(target='sample',
                                           apply_envelope=apply_envelope,
                                            envelope_width=envelope_width,
                                            valid_thresh=valid_thresh,
                                            optimize_BB=optimize_BB,
                                            optimize_phase=optimize_phase,
                                            view_phase_alignment=False)
        return np.array([f,
                         spectrum_abs,
                         SP.get_phase(f, spectrum, level_phase=True)])
    except:
        error_text = str(traceback.format_exc())
        error_file = os.path.join(diagnostic_dir,'BB_referenced_spectrum.err')
        with open(error_file,'w') as f:
            f.write(error_text)

        raise

def normalized_spectrum(sample_spectra, sample_BB_spectra,
                       ref_spectra, ref_BB_spectra,
                       apply_envelope=True, envelope_width=1,
                       optimize_BB=True, optimize_phase=True,
                       phase_offset=0,valid_thresh=.01, piecewise_flattening=0):

    SP = SpectralProcessor(sample_spectra, sample_BB_spectra,
                           ref_spectra, ref_BB_spectra)

    f, snorm_abs,snorm = SP(apply_envelope=apply_envelope, envelope_width=envelope_width,
                          valid_thresh=valid_thresh,
                          optimize_BB=optimize_BB,
                          optimize_phase=optimize_phase,
                          view_phase_alignment=False)

    if piecewise_flattening:
        phase = SP.get_phase(f, snorm, level_phase=True, manual_offset=0)
        offset,params = numrec.PiecewiseLinearFit(f,phase,nbreakpoints=piecewise_flattening,error_exp=0.25)
        phase -= offset
        snorm = snorm_abs*np.exp(1j*phase)

    #Now finally apply manual offset
    phase=SP.get_phase(f, snorm, level_phase=True, manual_offset=phase_offset)

    # `get_phase` will apply some phase leveling
    return np.array([f,
                     snorm_abs,
                     phase])
