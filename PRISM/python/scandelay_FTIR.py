import os
import numpy as np
from scipy.interpolate import interp1d
from common import numerics as num
from common.baseclasses import AWA
from scipy.signal import fftconvolve as convolve
from scipy.stats import binned_statistic
from scipy.optimize import leastsq,minimize

basedir = os.path.dirname(__file__)

dX=None #This will imply the same df throughout, until (say), `normalize_spectra`

interp_kwargs=dict(bounds_error=False,
                   fill_value=0,
                   kind='cubic')

def test(intfgms_arr, delay_calibration_factor=1,
                         order=1,refresh_alignment=True,
                         flattening_order=20,Nwavelengths=4):

    import traceback
    try:
        result = align_interferograms(intfgms_arr)
        return str(1)

    except:

        return str(traceback.format_exc())

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

def align_interferograms(intfgs_arr, delay_calibration_factor=1,
                         delay0=12,optimize_delay=True,delay_range=15,
                         flattening_order=20):
    #At a scan rate of 250kHz, a 12-point delay corresponds to only about 50 microseconds

    global dX
    from common import numerical_recipes as numrec
    Nx=50

    Ncycles = len(intfgs_arr) // 2

    all_xs = []
    all_ys = []
    for i in range(Ncycles):
        # i=i+10
        ys, xs = np.array(intfgs_arr[2 * i:2 * (i + 1)])
        Nbins = len(xs)

        xs *= delay_calibration_factor
        xs = numrec.smooth(xs, window_len=Nx, axis=0)
        ys -= np.polyval(np.polyfit(x=xs, y=ys, deg=flattening_order), xs)

        all_xs = np.append(all_xs, xs)
        all_ys = np.append(all_ys, ys)

    x1 = np.min(all_xs)
    x2 = np.max(all_xs)
    if dX is None: dX = x2 - x1

    def delayed_intfg(delay):

        #Assume positions are reported in delayed fashion;
        # correct delay by rolling them backwards to correspond to earlier time points
        all_xs_rolled = np.roll(all_xs, -delay, axis=0)
        result = binned_statistic(all_xs_rolled, all_ys, bins=Nbins)
        xnew = result.bin_edges[:-1]
        intfg_new = result.statistic

        keep=np.isfinite(intfg_new)
        xnew=xnew[keep]
        intfg_new=intfg_new[keep]

        return xnew, intfg_new

    #Scan over delays to find most ideal value
    if optimize_delay:
        delays = np.arange(int(delay0-delay_range),
                           int(delay0+delay_range), 1)
        sums = []
        for delay in delays:
            xnew, intfg_new = delayed_intfg(delay)
            sums.append(np.sum(intfg_new ** 2))

        delay0 = delays[np.argmax(sums)]

    xnew, intfg_new = delayed_intfg(delay0)

    intfg_interp = interp1d(x=xnew,y=intfg_new,
                            **interp_kwargs)
    xnew=np.linspace(x1,x1+dX,len(xnew))
    intfg_new = intfg_interp(xnew)

    return np.array([intfg_new,xnew])

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
    #w = 1 #We are disabling the windowing for now

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

    if envelope: env,envelope_params = fit_envelope(f,sabs)
    else: env=[1]*len(f); envelope_params=[0]
    Nf=len(f); Ne=len(envelope_params)
    if Ne<Nf: envelope_params=np.append(envelope_params,[0]*(Nf-Ne))

    # this will apply the knowledge that distance is not zero at first index of spectrum
    pcorr = -2 * np.pi * (np.min(d)) * f
    pvf += pcorr
    pvf = (pvf+np.pi)%(2*np.pi)-np.pi

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
                                   factor=1, optimize_BB=True,
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
    def level_phase(f, s, order=1):

        p = np.unwrap(np.angle(s))
        m = np.polyfit(x=f, y=p, deg=order)
        # m[-1]=0 #no overall phase offsets
        p -= np.polyval(m, f)

        return np.abs(s) * np.exp(1j * p)

    @classmethod
    def get_phase(cls,f,s,
                  level_phase=True,order=1):

        if level_phase:
            s=cls.level_phase(f,s,order=order)

        return np.unwrap(np.angle(s))

    @classmethod
    def pair_difference(cls,ps, f, spectrum, spectrum0, exp=1):  # shift spectrum to match spectrum0
        p = ps[0]

        target = np.abs(spectrum0) + np.abs(spectrum)
        actual = np.abs(spectrum0 + cls.phase_displace(f, spectrum, p))
        delta = np.abs(target - actual) ** exp

        return delta  # *penalty**1

    @classmethod
    def overall_difference(cls,ps, f, spectra, spectrum0, exp=1):

        spectra = [cls.phase_displace(f, s, p) for s, p in zip(spectra, ps)]
        spectrum = cls.summed_spectrum(spectra, abs=False)
        delta = np.abs(np.abs(spectrum) - np.abs(spectrum0)) ** exp
        #pvariation = np.unwrap(np.angle(cls.level_phase(f, spectrum, 1)))
        # pvariation=np.unwrap(np.angle(spectrum))
        # pvariation=np.diff(pvariation)
        #penalty = np.sqrt(np.sum(np.diff(pvariation) ** 2))

        return delta# * penalty ** (1 / 2)

    @classmethod
    def phase_aligned_spectrum(cls, f, spectra, verbose=False):

        # The result seems robust but potentially A LOT of evaluations are needed
        lskwargs = dict(factor=.1, maxfev=int(1e5))
        error_exp = 1  # Somehow a medium exponent converges fastest

        Nspectra = len(spectra)
        ps = [0] * Nspectra
        center_ind = np.argmax([np.abs(s).max() for s in spectra])
        spectrum0 = spectra[center_ind]
        spectra_new = [0] * Nspectra
        spectra_new[center_ind] = spectrum0

        #Phase-align spectra to the "left" of the starting spectrum
        for i in range(center_ind):
            workind = center_ind - (i + 1)
            if verbose: print('working on index %i' % workind)
            spectrum = spectra[workind]
            p = leastsq(cls.pair_difference, [0],
                        args=(f, spectrum, spectrum0, error_exp),
                        **lskwargs)[0][0]
            ps[workind] = p
            spectra_new[workind] = cls.phase_displace(f, spectrum, p)
            spectrum0 = np.sum(spectra_new, axis=0)

        #Phase-align spectra to the "right" of the starting spectrum
        for j in range(Nspectra - (center_ind + 1)):
            workind = center_ind + (j + 1)
            if verbose: print('working on index %i' % workind)
            spectrum = spectra[workind]
            p = leastsq(cls.pair_difference, [0],
                        args=(f, spectrum, spectrum0, error_exp),
                        **lskwargs)[0][0]
            ps[workind] = p
            spectra_new[workind] = cls.phase_displace(f, spectrum, p)
            spectrum0 = np.sum(spectra_new, axis=0)

        # report the result
        cls.phase_alignments = ps
        if verbose:
            print('Phase compensations (2pi*cm):\n\t',
                  '\n\t'.join((str(p) for p in ps)))

        stot = spectrum0
        #If we want a single-pass at all-at-once alignment (not very good)
        """ stot0 = cls.summed_spectrum(spectra,abs=True)
        ps = leastsq(cls.overall_difference,ps,
                     args=(f,spectra,stot0,1),
                     **lskwargs)[0]
        stot = cls.summed_spectrum([cls.phase_displace(f,s,p) for s,p
                                    in zip(spectra,ps)],
                                   abs = False)    
        """
        return stot

    @classmethod
    def phase_aligned_spectrum_multipass(cls, f, spectra, verbose=False):

        cls.phase_aligned_spectrum(f, spectra, verbose=False)
        ps = cls.phase_alignments

        # The result seems robust but potentially A LOT of evaluations are needed
        lskwargs = dict(factor=.1, maxfev=int(1e5))
        error_exp = 1  # Somehow a medium exponent converges fastest

        Nspectra = len(spectra)
        spectrum0 = spectra[0]
        for center_ind in range(Nspectra):

            spectrum0 = np.sum([cls.phase_displace(f, s, p) \
                                for s, p in zip(spectra[:center_ind + 1],
                                                ps[:center_ind + 1])],
                               axis=0)

            # Phase-align spectra to the "right" of the starting spectrum
            for j in range(Nspectra - (center_ind + 1)):
                workind = center_ind + (j + 1)
                if verbose: print('working on index %i' % workind)
                spectrum = spectra[workind]
                p0=ps[workind]
                p = leastsq(cls.pair_difference, [p0],
                            args=(f, spectrum, spectrum0, error_exp),
                            **lskwargs)[0][0]
                ps[workind] = p

                spectrum0 = np.sum([cls.phase_displace(f, s, p) \
                                    for s, p in zip(spectra[:workind + 1],
                                                    ps[:workind + 1])],
                                   axis=0)

        # report the result
        cls.phase_alignments = ps
        if verbose:
            print('Phase compensations (2pi*cm):\n\t',
                  '\n\t'.join((str(p) for p in ps)))

        stot = spectrum0
        return stot

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
                        optimize_BB=True,
                        optimize_phase=True,
                        view_phase_alignment=False):

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

        # Align phase only if commanded and if complex value is to be retained
        if optimize_phase:
            print('Aligning phase...')
            spectrum = self.phase_aligned_spectrum(f0, spectra, verbose=False)

            if view_phase_alignment:
                from matplotlib import pyplot as plt
                plt.figure()
                a = self.summed_spectrum(spectra, abs=True)
                b = np.abs(self.summed_spectrum(spectra, abs=False))
                c = np.abs(spectrum)
                plt.plot(f0, a, label='Phase discarded')
                plt.plot(f0, b, label='Phase unaligned')
                plt.plot(f0, c, label='Phase aligned')
                plt.ylabel('Power spectral density')
                plt.legend(loc='best')
                plt.twinx()
                plt.plot(f0, a - c, color='r')
                plt.ylabel('Spectral density lost to phase',
                           rotation=270,labelpad=20,color='r')
                plt.xlim(500, 2500)

            spectra = [spectrum]

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
                 optimize_BB=True,
                 optimize_phase=True,
                 apply_envelope=True,
                 envelope_width=1,
                 valid_thresh=.01,
                 view_phase_alignment=True,
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

    return SpectralProcessor.accumulate_spectra(spectra,
                                                apply_envelope=apply_envelope,
                                                expand_envelope=expand_envelope)

def BB_referenced_spectrum(spectra,spectra_BB,
                          apply_envelope=True, envelope_width=1,
                           optimize_BB=False,
                          optimize_phase=False,abs_only=True,
                           valid_thresh=.01):

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

def normalized_spectrum(sample_spectra, sample_BB_spectra,
                       ref_spectra, ref_BB_spectra,
                       apply_envelope=True, envelope_width=1,
                       optimize_BB=True, optimize_phase=True,
                       valid_thresh=.01):

    SP = SpectralProcessor(sample_spectra, sample_BB_spectra,
                           ref_spectra, ref_BB_spectra)

    f, snorm_abs,snorm = SP(apply_envelope=apply_envelope, envelope_width=envelope_width,
                          valid_thresh=valid_thresh,
                          optimize_BB=optimize_BB,
                          optimize_phase=optimize_phase,
                          view_phase_alignment=False)

    # `get_phase` will apply some phase leveling
    return np.array([f,
                     snorm_abs,
                     SP.get_phase(f, snorm, level_phase=True)])


############
#- Old stuff
############

def subtract_phase_slope(f,s,phase_slope):

    s=s*np.exp(-1j*f*phase_slope) #positive phase slope means removal of positive time delay

    return s

def accumulate_spectra_old(spectra, apply_envelope=True,expand_envelope=1):
    # `Spectra` is expected to be a set of stacked row vectors in groups of five,
    # each group is indexed by row as:
    # 0: frequencies
    # 1: spectrum amplitude
    # 2: spectrum phase
    # 3: spectrum envelope
    # 4: spectrum envelope parameters

    phase_slope = 0

    spectra = np.array(spectra)
    Nrows = 5
    assert len(spectra) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors'%Nrows

    s_accumulated = 0
    s_accumulated_abs = 0
    Nspectra = len(spectra) // Nrows
    for i in range(Nspectra):

        f = spectra[Nrows * i]
        sabs = spectra[Nrows * i + 1]
        sphase = spectra[Nrows * i + 2]
        #env = spectra[4 * i + 3]
        s = sabs * np.exp(1j * sphase)

        # In case we have invalid entries (zeros), remove them
        where_valid = np.isfinite(f) * (f > 0)
        f = f[where_valid]
        s = s[where_valid]

        if i == 0:
            f0 = f
        else:
            # interpolate to original frequencies
            s = interp1d(f, s, **interp_kwargs)(f0)
        if apply_envelope:
            env_params = spectra[Nrows * i + 4][:3]
            env = spectral_envelope(f0,*env_params,
                                    expand_envelope=expand_envelope)
            s = s * (env / env.max())  # normalize the envelope to unity

        s_accumulated += s
        s_accumulated_abs += np.abs(s)

    # Subtract a phase slope
    s_accumulated=subtract_phase_slope(f0,s_accumulated,phase_slope)

    s_accum_phase = np.angle(s_accumulated)

    return np.array([f0, s_accumulated_abs, s_accum_phase])

def normalize_spectra_old(spectra, spectra_ref,
                      apply_envelope=True, expand_envelope=1,
                      reference_phase=False):
    # `Spectra` is expected to be a set of stacked row vectors in groups of five,
    # each group is indexed by row as:
    # 0: frequencies
    # 1: spectrum amplitude
    # 2: spectrum phase
    # 3: spectrum envelope
    # 4: envelope parameters
    # Same for reference spectra

    # We need to apply envelopes from reference to analyte spectra!

    # Presumably we can now refresh any dX
    #global dX
    #dX = None #Keep dX through duration of experiment

    spectra = np.array(spectra)
    spectra_ref = np.array(spectra_ref)
    phase_slope = 0

    Nrows = 5
    if apply_envelope:
        assert len(spectra)==len(spectra_ref),\
            "Applying envelope requires same number of accumulations for both analyte and reference"

    assert len(spectra) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors'%Nrows
    assert len(spectra_ref) % Nrows == 0, 'Input spectra must come as stacked groups of %i row vectors'%Nrows

    s_accumulated = 0
    s_accumulated_abs = 0
    s_accumulated_ref = 0
    s_accumulated_abs_ref = 0
    Nspectra = np.max( (len(spectra) // Nrows, 1) )

    # Establish some mutual frequency axis
    all_fs = np.append([], [spectra[Nrows * i] for i in range(Nspectra)] \
                           + [spectra_ref[Nrows * i] for i in range(Nspectra)] )
    Nf = len(all_fs)//(2*Nspectra)
    f0 = np.linspace(np.min(all_fs),
                     np.max(all_fs),
                     Nf)

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
        s = interp1d(f, s, **interp_kwargs)(f0)

        ## Reference spectrum components
        f_ref = spectra_ref[Nrows * i]
        sabs_ref = spectra_ref[Nrows * i + 1]
        sphase_ref = spectra_ref[Nrows * i + 2]
        s_ref = sabs_ref * np.exp(1j * sphase_ref)

        # In case we have invalid entries (zeros), remove them
        where_valid = np.isfinite(f_ref) * (f_ref > 0)
        f_ref = f_ref[where_valid]
        s_ref = s_ref[where_valid]

        s_ref = interp1d(f_ref, s_ref,  **interp_kwargs)(f0)

        if apply_envelope:
            env_params = spectra_ref[Nrows * i + 4][:3]
            env = spectral_envelope(f0, *env_params,
                                    expand_envelope=expand_envelope)
            env_norm = env/env.max() # normalize the envelope to unity
            s_ref = s_ref * env_norm
            s = s * env_norm

        s_accumulated += s
        s_accumulated_abs += np.abs(s)
        s_accumulated_ref += s_ref
        s_accumulated_abs_ref += np.abs(s_ref)

    # Decide whether to propagate the reference phase or not to normalized spectrum
    if reference_phase: s_norm = s_accumulated / s_accumulated_ref
    else: s_norm = s_accumulated / s_accumulated_abs_ref
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

def double_normalize_spectra(spectra1, spectra1_ref,
                             spectra2, spectra2_ref,
                             apply_envelope=True, expand_envelope=1,
                             reference_phase=False):
    # Normalize `spectra1` with `spectra2` while accumulating with their own internal references.

    phase_slope = 0

    f1,s1_norm_abs,s1_norm_phase = normalize_spectra_old(spectra1, spectra1_ref,
                                                    apply_envelope=apply_envelope,expand_envelope=expand_envelope,
                                                     reference_phase=reference_phase)
    s1_norm = s1_norm_abs*np.exp(1j*s1_norm_phase)

    f2,s2_norm_abs,s2_norm_phase = normalize_spectra_old(spectra2, spectra2_ref,
                                                    apply_envelope=apply_envelope,expand_envelope=expand_envelope,
                                                     reference_phase=reference_phase)
    s2_norm = s2_norm_abs*np.exp(1j*s2_norm_phase)

    #Find mutual frequency axis
    Nf = int( np.mean( (len(f1), len(f2)) ) )
    fmin = np.max( (np.min(f1), np.min(f2)) )
    fmax = np.min( (np.max(f1), np.max(f2)) )
    f0 = np.linspace(fmin,fmax,Nf)

    # interpolate second spectrum to mutual frequencies
    s1_norm = interp1d(f1, s1_norm, **interp_kwargs)(f0)

    # interpolate second spectrum to mutual frequencies
    s2_norm = interp1d(f2, s2_norm, **interp_kwargs)(f0)
    s2_norm_abs = np.abs(s2_norm) #Recompute now that frequencies were interpolated

    s_norm = s1_norm / s2_norm
    s_norm_abs = np.abs(s_norm)

    # having divided by a reference possibly with zeros, remove those entries
    where_valid = (s2_norm_abs != 0 )*np.isfinite(s2_norm_abs)
    s_norm=s_norm[where_valid]
    s_norm_abs=s_norm_abs[where_valid]
    f0 = f0[where_valid]

    # Subtract a phase slope
    s_norm = subtract_phase_slope(f0, s_norm, phase_slope)

    s_norm_phase = np.angle(s_norm)

    return np.array([f0, s_norm_abs, s_norm_phase])