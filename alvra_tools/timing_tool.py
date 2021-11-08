import os
import pathlib

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import tukey
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve1d

#spectrometer wavelength calibration / frequency conversion
lambdas = 467.55 + 0.07219*np.arange(0,2047) # calibration from 23-9-2020
nus = 299792458 / (lambdas * 10**-9) # frequency space, uneven
nus_new = np.linspace(nus[0], nus[-1], num=2047, endpoint=True) # frequency space, even
pixelNum = np.arange(0,2047)
filters = {
    "YAG": np.concatenate((np.ones(50),tukey(40)[20:40], np.zeros(1977), np.zeros(2047))), # fourier filter for YAGS
    "SiN": np.concatenate((tukey(40)[20:40], np.zeros(2027), np.zeros(2047))), # fourier filter for 5um SiN
    "babyYAG": np.concatenate((tukey(40)[20:40], np.zeros(2028), np.zeros(2048))), # baby timetool YAG filter
    "babyYAG2": np.concatenate((np.ones(50),tukey(40)[20:40], np.zeros(1978), np.zeros(2048))) # baby timetool YAG
}


Heaviside = np.concatenate((np.zeros(100), np.ones(100)))

#Baby PSEN spectrometer wavelength calibration from 06-10-2021
lambdas_baby = 528.34 + 0.0261*np.arange(0,2048) # calibration from 06-10-2021 on the small kymera
nus_baby = 299792458 / (lambdas_baby * 10**-9) # frequency space, uneven, baby
nus_new_baby = np.linspace(nus_baby[0], nus_baby[-1], num=2048, endpoint=True) # frequency space, even, baby
derivFilter = np.concatenate((tukey(2000)[0:500], np.ones(2048-1000), tukey(2000)[1500:2000])) # to get rid of the edges



def arrivalTimes_selfRef(filter_name, px2fs, background_avg, signals):
    """
    returns:
    - arrival times in fs determined from argmax of peak traces and the calibration px2fs
    - amplitudes of the peak traces
    - signal traces (edgewfm), should show a change in transmission near px 1024 if set up correctly
    - peak traces (peakwfm), which are the derivative of signal traces
    """
    
    p0 = 1024
    
    edgepos, edgewfm, peakwfm = edge_selfRef(filter_name, background_avg, signals)

    arrivalTimes = (p0 - edgepos)*px2fs
    arrivalAmplitudes = np.max(peakwfm, axis = -1) * 11500

    return arrivalTimes, arrivalAmplitudes, edgewfm, peakwfm


def arrivalTimes(filter_name, px2fs, backgrounds, signals, background_from_fit, peakback):
    """
    returns:
    - arrival times in fs determined from argmax of peak traces and the calibration px2fs
    - amplitudes of the peak traces
    - signal traces (edgewfm), should show a change in transmission near px 1024 if set up correctly
    - peak traces (peakwfm), which are the derivative of signal traces
    """

    p0 = 1024

    edgepos, edgewfm, peakwfm = edge(filter_name, backgrounds, signals, background_from_fit, peakback)   

    arrivalTimes = (p0 - edgepos)*px2fs
    arrivalAmplitudes = np.max(peakwfm, axis = -1) * 11500    
    
    return arrivalTimes, arrivalAmplitudes, edgewfm, peakwfm

def _get_base_folder(fname):
    fname = fname.split(os.sep)
    return os.sep.join(fname[:5])

def find_backgrounds(fname, path):
    fpath = pathlib.Path(fname)
    fmtime = fpath.stat().st_mtime

    background_path = _get_base_folder(fname) + path
    background_path = pathlib.Path(background_path)

    background = None
    peak_background = None
    min_time_diff1 = float('inf')
    min_time_diff2 = float('inf')
    for entry in background_path.iterdir():
        if entry.is_file() and 'psen-background' in entry.name:
            pmtime = entry.stat().st_mtime
            time_diff1 = abs(pmtime - fmtime)
            if time_diff1 < min_time_diff1:
                min_time_diff1 = time_diff1
                background = entry
        elif entry.is_file() and 'psen-peak-background' in entry.name:    
            pmtime = entry.stat().st_mtime
            time_diff2 = abs(pmtime - fmtime)
            if time_diff2 < min_time_diff2:
                min_time_diff2 = time_diff2
                peak_background = entry

    return background, peak_background, fmtime


def edge_selfRef(filter_name, back_avg, signals):
    """
    returns:
    edge positions determined from argmax of peak traces
    signal traces, should show a change in transmission near px 1024 if set up correctly
    peak traces, which are the derivative of signal traces
    """
    
    ffilter = filters[filter_name]
    # background subtraction
    sig2 = np.nan_to_num(signals / back_avg)
    # interpolate to get evenly sampled in frequency space
    sig3inter = interp1d(nus_baby, sig2, kind='cubic')
    sig3 = sig3inter(nus_new_baby)
    sig4 = np.hstack((sig3, np.zeros_like(sig3)))
    # Fourier transform, filter, inverse fourier transform, take the real part, take the derivative (sig5gaussO1)
    sig4fft = np.fft.fft(sig4)
    sig4filtered = sig4fft * ffilter
    sig4inverse = np.fft.ifft(sig4filtered)
    sig4invreal = 2 * np.real(sig4inverse)
    sig4inter = interp1d(nus_new_baby, sig4invreal[..., 0:2048], kind='cubic')
    sig5 = sig4inter(nus_baby)

    # transmissive edges, not used, just for plotting if you like.
    sig5gaussO0 = gaussian_filter1d(sig5, 30)
    sig6 = convolve1d(sig5gaussO0, Heaviside)
    # peaks
    sig5gaussO1 = gaussian_filter1d(sig5, 50, order = 1)
    peak2 = np.argmax(sig5gaussO1*derivFilter, axis = -1)
    #peak2 = np.argmax(sig5gaussO1[500:1500], axis = -1)+500
    sig5gaussO1 = sig5gaussO1*derivFilter

    return peak2, sig6, sig5gaussO1


def edge(filter_name, backgrounds, signals, background_from_fit, peakback):
    """
    returns:
    edge positions determined from argmax of peak traces
    signal traces, should show a change in transmission near px 1024 if set up correctly
    peak traces, which are the derivative of signal traces
    """

    ffilter = filters[filter_name]
    # background subtraction
    sig2 = np.nan_to_num(signals / backgrounds) / background_from_fit
    # interpolate to get evenly sampled in frequency space
    sig3inter = interp1d(nus, sig2, kind='cubic')
    sig3 = sig3inter(nus_new)
    sig4 = np.hstack((sig3, np.zeros_like(sig3)))
    # Fourier transform, filter, inverse fourier transform, take the real part, take the derivative (sig5gaussO1)
    sig4fft = np.fft.fft(sig4)
    sig4filtered = sig4fft * ffilter
    sig4inverse = np.fft.ifft(sig4filtered)
    sig4invreal = 2 * np.real(sig4inverse)
    sig4inter = interp1d(nus_new, sig4invreal[..., 0:2047], kind='cubic')
    sig5 = sig4inter(nus)

    # transmissive edges, not used, just for plotting if you like.
    sig5gaussO0 = gaussian_filter1d(sig5, 30)
    sig6 = convolve1d(sig5gaussO0, Heaviside)
    # peaks
    sig5gaussO1 = gaussian_filter1d(sig5, 50, order = 1) - peakback
    peak2 = np.argmax(sig5gaussO1, axis = -1)

    return peak2, sig6, sig5gaussO1
