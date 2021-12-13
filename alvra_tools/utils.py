import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit

class Fit:
    
    def __init__(self, func, estim, p0=None):
        self.func = func
        self.estim = estim
        self.p0 = self.popt = p0
        self.pcov = None
   
    def estimate(self, x, y):
        self.p0 = self.popt = self.estim(x,y)

    def fit(self, x, y):
        self.popt, self.pcov = curve_fit(self.func, x, y, p0=self.p0)
    
    def eval(self, x):
        return self.func(x, *self.popt)


def _bin(a, binning):
    if a.size % binning != 0:
        rounded = a.size // binning * binning
        a = a[0:rounded]
    return a.reshape((-1, binning))

def bin_sum(a, binning):
    return _bin(a, binning).sum(axis=1)

def bin_mean(a, binning):
    return _bin(a, binning).mean(axis=1)


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


def mm2fs(x, t0_mm):
     return (x-t0_mm)*2/(299792458*1e3*1e-15)

    
def fs2mm(x,t0_fs):
     return (t0_fs + x)/2*(299792458*1e3*1e-15)


def cut(arr, minlen):
    return np.array([i[:minlen] for i in arr])

def better_p0(p0, index, value):
    mod = list(p0)
    mod[index] = value
    return tuple(mod)
