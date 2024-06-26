import numpy as np
import glob, h5py, time
from scipy.special import erf
from scipy.optimize import curve_fit
from datetime import datetime
import colorcet as cc


class Fit:
    
    def __init__(self, func, estim, p0=None, **kwargs):
        self.func = func
        self.estim = estim
        self.p0 = self.popt = p0
        self.pcov = None
        self.kwargs = kwargs
   
    def estimate(self, x, y):
        self.p0 = self.popt = self.estim(x,y)

    def fit(self, x, y, **kwargs):
        self.popt, self.pcov = curve_fit(self.func, x, y, p0=self.p0, **self.kwargs, **kwargs)
    
    def eval(self, x):
        return self.func(x, *self.popt)

def better_p0(p0, index, value):
    mod = list(p0)
    mod[index] = value
    return tuple(mod)

def timestamp(json_file):
    file_split = json_file.split('/')[:-1]
    path_to_bsdata = '/'.join([*file_split[:-1], 'data', '*BSDATA.h5'])
    
    timestamp_s = []
    
    for file in glob.glob(path_to_bsdata):
        with h5py.File(file) as f:
            timestamp_ns = f['SAR-CVME-TIFALL5:EvtSet']['timestamp'][:]
            timestamp_s.append(np.mean(timestamp_ns) * 1e-9)
    timestamp_s = np.mean(timestamp_s)
    timestamp_datetime = datetime.fromtimestamp(timestamp_s)
    return np.datetime64(timestamp_datetime)

def timestamp_hms(json_file):
    file_split = json_file.split('/')[:-1]
    path_to_bsdata = '/'.join([*file_split[:-1], 'data', '*BSDATA.h5'])
    
    timestamp_s = []
    
    for file in glob.glob(path_to_bsdata):
        with h5py.File(file) as f:
            timestamp_ns = f['SAR-CVME-TIFALL5:EvtSet']['timestamp'][:]
            timestamp_s.append(np.mean(timestamp_ns) * 1e-9)
    timestamp_s = np.mean(timestamp_s)
    return time.strftime('%H:%M:%S', time.localtime(timestamp_s) )


def _bin(a, binning):
    if a.size % binning != 0:
        rounded = a.size // binning * binning
        a = a[0:rounded]
    return a.reshape((-1, binning))

def bin_sum(a, binning):
    return _bin(a, binning).sum(axis=1)

def bin_mean(a, binning):
    return _bin(a, binning).mean(axis=1)

def rebin2D(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).sum(-1).sum(1)


def convert_to_photon_num_range(image, photon_range):
    """
    Convert energy to a number of photons counting values falling within a particular range.
    This will always return integer photon counts.
    """
    offset = photon_range[0]
    mean = np.mean(photon_range)
    return np.ceil(np.divide(image - offset, mean))

def convert_to_photon_num_mean(image, photon_range):
    """
    Convert energy to a number of photons using the central energy of a single photon.
    This can return fractional number of photons.
    """
    return image / np.mean(photon_range)

def threshold(data, lower=None, upper=None, inplace=True, fill=0):
    if not inplace:
        data = data.copy()
    if lower is not None:
        data[data <= lower] = fill
    if upper is not None:
        data[data > upper] = fill
    return data


def crop_roi(arr, roi):
    if roi is None:
        return arr
    r0, r1 = make_roi(roi)
    return arr[..., r0, r1]

def make_roi(roi):
    roi = np.array(roi).ravel()
    r0 = slice(*roi[2:])
    r1 = slice(*roi[:2])
    return r0, r1

def errfunc_fwhm(x, x0, amplitude, width, offset):
    return offset + amplitude*erf((x0-x)*2*np.sqrt(np.log(2))/(np.abs(width)))         #d is fwhm

def errfunc_1e2(x, x0, amplitude, width, offset):
    return offset + amplitude*erf((x0-x)*2*np.sqrt(2*np.log(2))/(np.abs(width)))       #d is 1/e2, 1/e2 = 1.699 * fwhm

def errfunc_sigma(x, x0, amplitude, width, offset):
    return offset + amplitude*erf((x0-x)/(np.sqrt(2)*np.abs(width)))                   #d is sigma, fwhm = 2.355 * sigma

def estimate_errfunc_parameters(x,y):
    x0 = x.mean()
    amplitude = y.max()
    width = np.diff(x).mean()
    offset = y.min()
    return x0, amplitude, width, offset


def conv_exp_gauss(x,a,b,c,d,e):
    A_fun = 1/(d*np.sqrt(2*np.pi))*np.exp(-((-x+c)**2)/(2*d**2))
    B_fun = np.heaviside((-x+c),0)*np.exp(-(-x+c)/e)   
    return  a + b*np.convolve(A_fun,B_fun,'same')
#return a + b*np.convolve(1/(d*np.sqrt(2*np.pi))*np.exp(-((x-c)**2)/(2*d**2)),np.heaviside((x-c),0)*np.exp(-(x-c)/e),'same')
  #d is sigma, fwhm = 2.355 * sigma

def gaussian(x, x0, amplitude, sigma, offset):
    return amplitude*np.exp(-(x - x0)**2/(2*sigma**2)) + offset

def estimate_gaussian_parameters(x,y):
    x0 = x[np.argmax(np.abs(y))]
    amplitude = np.max(np.abs(y))
    sigma =  np.diff(x).mean()
    offset = np.min(np.abs(y)) 
    return x0, amplitude, sigma, offset

def three_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, offset):
    return (gaussian(x, h1, c1, w1, offset=0) +
        gaussian(x, h2, c2, w2, offset=0) +
        gaussian(x, h3, c3, w3, offset=0) + offset)

def two_gaussians(x, h1, c1, w1, h2, c2, w2, offset):
    return three_gaussians(x, h1, c1, w1, h2, c2, w2, 0,0,1, offset)


def conv_exp_gauss_heaviside(x,x0,amplitude,width,offset,lifetime):
    sigma = width/2./np.sqrt(2*np.log(2))
    frac1 = (sigma**2-2*lifetime*(x0-x))/2./lifetime**2
    frac2 = (sigma**2 - lifetime*(x0-x))/np.sqrt(2)/sigma/lifetime
    return amplitude*0.5*np.exp(frac1)*(1-erf(frac2)) + offset


def estimate_conv_exp_gauss_heaviside_parameters(x,y):
    x0 = x.mean()
    amplitude = y.max()
    width = np.diff(x).mean()
    offset = y.min()
    lifetime = 0.25
    return x0, amplitude, width,offset,lifetime


def conv_exp_gauss_heaviside2(x,x0,amplitude,width,offset,lifetime,a,b):
    sigma = width/2./np.sqrt(2*np.log(2))
    frac1 = (sigma**2-2*lifetime*(x0-x))/2./lifetime**2
    frac2 = (sigma**2 - lifetime*(x0-x))/np.sqrt(2)/sigma/lifetime
    return amplitude*0.5*(np.exp(frac1)+a*x+b)*(1-erf(frac2)) + offset


def estimate_conv_exp_gauss_heaviside2_parameters(x,y):
    x0 = x.mean()
    amplitude = y.max()
    width = np.diff(x).mean()
    offset = y.min()
    lifetime = 0.25
    a = 0
    b = 0
    return x0, amplitude, width,offset,lifetime, a, b

def model_decay_1exp(x, x0, sigma, amp1, tau1, C):
    first_exp  = 0.5*(np.exp(-1/tau1*(x-x0-sigma**2/tau1))*(1 + erf((x-x0-sigma**2/tau1)/(np.sqrt(2)*sigma))))
    total = C + amp1*first_exp# + amp2*second_exp 
    return total

def model_decay_2exp(x, x0, sigma, amp1, tau1, C, amp2, tau2):
    first_exp  = 0.5*(np.exp(-1/tau1*(x-x0-sigma**2/tau1))*(1 + erf((x-x0-sigma**2/tau1)/(np.sqrt(2)*sigma))))
    second_exp = 0.5*(np.exp(-1/tau2*(x-x0-sigma**2/tau2))*(1 + erf((x-x0-sigma**2/tau2)/(np.sqrt(2)*sigma))))
    total = C + amp1*first_exp + amp2*second_exp 
    return total

def estimate_model_decay_2exp_parameters(x,y):
    x0 = 0
    amp1 = y.max()
    amp2 = y.max()/2
    amp3 = y.max()/2
    tau1 = x.mean()/2
    tau2 = x.mean()
    sigma = np.diff(x).mean() 
    offset = y.min()

def AsymPseudoVoigt(x, x0, amplitude, w0, a, m, slope, C):
    w1 = 2*w0/(1+np.exp(-a*(x-x0)))
    a1 = (1-m)*np.sqrt(4*np.log(2)/(np.pi*w1**2))*np.exp(-(x-x0)**2*(4*np.log(2)/w1**2))
    a2 = m*(1/2*np.pi)*(w1/((w1/2)**2+4*x**2))
    offset = slope*(x-x0)+C
    Total = amplitude*(a1+a2+offset)
    return Total
    
def mm2fs(x, t0_mm):
     return (x-t0_mm)*2/(299792458*1e3*1e-15)

    
def fs2mm(x,t0_fs):
     return (t0_fs + x)/2*(299792458*1e3*1e-15)


def cut(arr, minlen):
    return np.array([i[:minlen] for i in arr])

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin),
                     np.arange(npt),
                     np.sort(x))

def rebin2D(arr, axis, bin_):
    arr = np.array(arr)
    arr_new=[]
    if axis == 1:
        arr=arr.T
    for index in range(len(arr)):
        cut=arr[index]
        new=bin_sum(cut,bin_)
        arr_new.append(new)
    arr_new=np.array(arr_new)
    if axis == 1:
        arr_new=arr_new.T
    return arr_new

def plot_tool_2D(matrix_ON, matrix_OFF, axis, x_axis, bin_):
    
    matrix_on_rebin  = rebin2D(matrix_ON, axis, bin_)
    matrix_off_rebin = rebin2D(matrix_OFF, axis, bin_)
    x_axis_rebin = bin_mean(x_axis, bin_)
    
    return x_axis_rebin, matrix_on_rebin, matrix_off_rebin#, low_err, high_err

def plot_tool_static_2D(matrix, axis, x_axis, bin_):
    
    matrix_rebin  = rebin2D(matrix, axis, bin_)
    x_axis_rebin = bin_mean(x_axis, bin_)
    
    return x_axis_rebin, matrix_rebin

def unwrap_spectra(ROIs, counter, spectra_shots_on, spectra_shots_off):
    
    s_all_on  = {}
    s_all_off = {}
    for key in ROIs:
        spectra_all_on  = []
        spectra_all_off = []
        for index_step in range(counter):
            spectra_all_on.extend(spectra_shots_on[index_step][key])
            spectra_all_off.extend(spectra_shots_off[index_step][key])
        s_all_on[key]  = spectra_all_on
        s_all_off[key] = spectra_all_off
    
    return s_all_on, s_all_off


def color_enumerate(iterable, start=0, cmap=cc.cm.rainbow):
    """
    same functionality as enumerate, but additionally yields sequential colors from
    a given cmap
    """

    n = start
    try:
        length = len(iterable)
    except TypeError:
        length = len(list(iterable))
    for item in iterable:
        yield n, cmap(n/(length-0.99)), item
        n += 1


