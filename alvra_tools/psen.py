import numpy as np
from bsread import source, Source
from scipy.interpolate import interp1d
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve1d
from epics import caput, PV
from datetime import datetime

ROI_background = 'SARES11-SPEC125-M2.roi_background_x_profile'
ROI_signal = 'SARES11-SPEC125-M2.roi_signal_x_profile'
Events = 'SAR-CVME-TIFALL4:EvtSet'
IZero = 'SAROP11-PBPS117:INTENSITY'
Channels = [ROI_background, ROI_signal, Events, IZero]

#spectrometer wavelength calibration / frequency conversion
lambdas = 467.55 + 0.07219*np.arange(0,2047) # calibration from 23-9-2020
nus = 299792458 / (lambdas * 10**-9) # frequency space, uneven
nus_new = np.linspace(nus[0], nus[-1], num=2047, endpoint=True) # frequency space, even

pixelNum = np.arange(0,2047)

def measure(nshots, chans):

    pulse_ids = np.empty(nshots)
    events = np.empty((nshots, 256))
    backgrounds = np.empty((nshots, 2047))
    signals = np.empty((nshots, 2047))
    iZeros = np.empty(nshots)
    
    stream = Source(channels=chans)
    stream.connect()

    ntotal = 0
    i = 0
    while i < nshots:
        ntotal += 1
        try:
            message = stream.receive()
        except Exception as e:
            print(type(e).__name__, e)
            while True:
                try:
                    stream = Source(channels=chans)
                    stream.connect()
                except Exception as e2:
                    print(type(e2).__name__, e2)
                else:
                    break

        data = message.data.data

        sig =  data[ROI_signal].value
        back = data[ROI_background].value
        i0 = data[IZero].value
        evs = data[Events].value
        if (sig is None) or (back is None) or (evs is None) or (i0 is None):
            continue
        pulse_ids[i] = message.data.pulse_id
        events[i] = evs
        backgrounds[i] = back
        signals[i] = sig
        iZeros[i] = i0
        i += 1
            
    stream.disconnect()

    timeofdata = datetime.now()
    print('Good shots: {} out of a total {} requested'.format(nshots, ntotal))
#    print(f'Good shots: {ngood} out of a total {ntotal} requested')

    return pulse_ids, events, backgrounds, signals, iZeros

filters = {
    "YAG": np.concatenate((np.ones(50),signal.tukey(40)[20:40], np.zeros(1977), np.zeros(2047))), # fourier filter for YAGS
    "SiN": np.concatenate((signal.tukey(40)[20:40], np.zeros(2027), np.zeros(2047))) # fourier filter for 5um SiN
}

Heaviside = np.concatenate((np.zeros(100), np.ones(100)))

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
#    sig4 = np.pad(sig3inter(nus_new), ((0,len(sig3inter(nus_new)))), 'constant', constant_values=0)
    sig4 = np.hstack((sig3, np.zeros_like(sig3)))
    # Fourier transform, filter, inverse fourier transform, take the real part, take the derivative (sig5gaussO1)
    sig4fft = np.fft.fft(sig4)
    sig4filtered = sig4fft * ffilter
    sig4inverse = np.fft.ifft(sig4filtered)
    sig4invreal = 2 * np.real(sig4inverse)
    sig4inter = interp1d(nus_new, sig4invreal[..., 0:2047], kind='cubic')
    sig5 = sig4inter(nus)
    
    # transmissive edges, not used, just for plotting if you like.
    sig5gaussO0 = gaussian_filter1d(sig5, 50)
    sig6 = convolve1d(sig5gaussO0, Heaviside)
    # peaks
    sig5gaussO1 = gaussian_filter1d(sig5, 50, order = 1) - peakback
    peak2 = np.argmax(sig5gaussO1, axis = -1)
    
    return peak2, sig6, sig5gaussO1

def goodshots(events, *arrays):
    fel = events[:, 13]
    laser = events[:, 18]
    darkShot = events[:, 21]
    good_shots = np.logical_and.reduce((fel, laser, np.logical_not(darkShot)))
    return [a[good_shots] for a in arrays]
