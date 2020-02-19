import h5py
import numpy as np

import jungfrau_utils as ju

from .channels import *
from .utils import crop_roi, make_roi


def _get_data(f):
    if "data" in f:
        return f["data"]
    else:
        return f

def _get_modulo(pulse_ids, modulo):
    nshots = len(pulse_ids)
    print ("Found {} shots in the file".format(nshots))
    nshots = nshots - (nshots % modulo)
    print ("Load {} shots".format(nshots))
    return nshots

def _cut_to_shortest_length(*args):
    shortest_length = min(len(i) for i in args)
    return [a[:shortest_length] for a in args]

def _average(data, modulo):
    length = len(data)
    length //= modulo
    length *= modulo
    data = data[:length]
    return data.reshape(-1, modulo).mean(axis=1)

def _make_reprates_on_off(pulse_ids, reprate_FEL, reprate_laser):
    #reprate_off = ((pulse_ids%10 == 0) & (pulse_ids%20 != 0))            #This is for 10 Hz
    #reprate_on  = pulse_ids%20 == 0                                      #This is for 5 Hz
    #reprate_off = ((pulse_ids%4 == 0) & (pulse_ids%8 != 0))              #This is for 25 Hz
    #reprate_on  = pulse_ids%8 == 0                                       #This is for 12.5 Hz
    reprate_on  = _make_reprates_on(pulse_ids, reprate_laser)
    reprate_off = _make_reprates_FEL_on_laser_off(pulse_ids, reprate_FEL, reprate_laser)
    return reprate_on, reprate_off

make_reprates_on_off = _make_reprates_on_off

def _make_reprates_on(pulse_ids, reprate):
    return pulse_ids % (100 / reprate) == 0

def _make_reprates_off(pulse_ids, reprate):
    return np.logical_not(_make_reprates_on(pulse_ids, reprate))

def _make_reprates_FEL_on_laser_off(pulse_ids, reprate_FEL, reprate_laser):
    return np.logical_and(_make_reprates_on(pulse_ids, reprate_FEL), _make_reprates_off(pulse_ids, reprate_laser))


def _get_detector_name(f):
    return f["general/detector_name"][()].decode()


def _make_empty_image(image, module_map):
    return np.zeros((512 * len(module_map), 1024), dtype=image.dtype)


def load_corr_JF_data(fname, nshots=None):
    with h5py.File(fname, "r") as f:

        data = _get_data(f)
        pulse_ids = data[channel_corr_JF_pulse_ids][:nshots]#.T[0] # pulse_ids comes in a weird shape
        images    = data[channel_corr_JF_images][:nshots]

    return images, pulse_ids


def load_JF_cropped_data(fname, roi, nshots=None):
    roi = str(roi)
    if not roi.startswith("images_roi"):
        roi = "roi" + roi if not roi.startswith("roi") else roi
        roi = "images_" + roi if not roi.startswith("images_") else roi

    coords = "coords_roi{}".format(roi[len("images_roi"):])

    with h5py.File(fname, "r") as f:
        keys = list(f.keys())
       # print(f"{fname} contains {keys}")

        coords = f[coords][:]
        print(f"{roi}: {coords}")

        pulse_ids = f["pulse_ids"][:nshots]
        images    = f[roi][:nshots]
    return images, pulse_ids


def load_crop_JF_data_on_off(fname, roi1, roi2, reprate_FEL, reprate_laser,
                             gain_file=None, pedestal_file=None, nshots=None):

    with ju.File(fname, gain_file=gain_file, pedestal_file=pedestal_file) as juf:
        images = juf[:nshots]
        pulse_ids = juf["pulse_id"][:nshots].T[0]

    images_roi1 = crop_roi(images, roi1)
    images_roi2 = crop_roi(images, roi2)

    reprate_on, reprate_off = _make_reprates_on_off(pulse_ids, reprate_FEL, reprate_laser)

    images_on_roi1  = images_roi1[reprate_on]
    images_on_roi2  = images_roi2[reprate_on]
    images_off_roi1 = images_roi1[reprate_off]
    images_off_roi2 = images_roi2[reprate_off]
    pulse_ids_on    = pulse_ids[reprate_on]
    pulse_ids_off   = pulse_ids[reprate_off]

    return images_on_roi1, images_on_roi2, pulse_ids_on, images_off_roi1, images_off_roi2, pulse_ids_off









###    Next: 2 functions to load pump-probe YAG data (events/pulseIDs)


def load_YAG_events(filename, modulo = 2, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:nshots]
        nshots = _get_modulo(pulse_ids,modulo)

        FEL = data[channel_Events][:nshots,48]
        Laser = data[channel_Events][:nshots,18]
        Darkshot = data[channel_Events][:nshots,21]
        
        index_dark_before = np.append([True], np.logical_not(Darkshot))[:-1]
        index_pump = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot), index_dark_before))
        index_unpump = np.logical_and.reduce((np.logical_not(FEL), Laser, np.logical_not(Darkshot), index_dark_before))
        
        LaserDiode_pump = data[channel_LaserDiode][:nshots][index_pump].ravel()
        LaserDiode_pump = _average(LaserDiode_pump, modulo -1)
        LaserDiode_unpump = data[channel_LaserDiode][:nshots][index_unpump].ravel()
        LaserDiode_unpump = _average(LaserDiode_unpump, modulo -1)

        LaserRefDiode_pump = data[channel_Laser_refDiode][:nshots][index_pump].ravel()
        LaserRefDiode_pump = _average(LaserRefDiode_pump, modulo -1)
        LaserRefDiode_unpump = data[channel_Laser_refDiode][:nshots][index_unpump].ravel()
        LaserRefDiode_unpump = _average(LaserRefDiode_unpump, modulo -1)

        IzeroFEL = data[channel_Izero][:nshots][index_pump].ravel()
        IzeroFEL = _average(IzeroFEL, modulo -1)

        #PIPS = data[channel_PIPS_trans][:nshots][index_pump]
        PIPS = data[channel_LaserDiode][:nshots][index_pump].ravel()

        Delay = data[channel_delay][:nshots][index_unpump]
        #Delay = BS_file[channel_laser_pitch][:][index_unpump]

        #BAM = BS_file[channel_BAM][:][index_pump]
        
        print ("Pump/umpump arrays have {} shots each".format(len(LaserDiode_pump), len(LaserDiode_unpump)))
        return _cut_to_shortest_length(LaserDiode_pump, LaserDiode_unpump, LaserRefDiode_pump, LaserRefDiode_unpump, IzeroFEL, PIPS, Delay, pulse_ids)


def load_YAG_pulseID(filename, reprateFEL, repratelaser):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        reprate_FEL, reprate_laser = _make_reprates_on_off(pulse_ids, reprateFEL, repratelaser)

        LaserDiode_pump = BS_file[channel_LaserDiode][:][reprate_FEL]
        LaserDiode_unpump = BS_file[channel_LaserDiode][:][reprate_laser]
        LaserRefDiode_pump = BS_file[channel_Laser_refDiode][:][reprate_FEL]
        LaserRefDiode_unpump = BS_file[channel_Laser_refDiode][:][reprate_laser]
        IzeroFEL = BS_file[channel_Izero][:][reprate_FEL]
        PIPS = BS_file[channel_PIPS_trans][:][reprate_FEL]

        Delay = BS_file[channel_delay][:][reprate_laser]
        #Delay = BS_file[channel_laser_pitch][:][index_unpump]

        #BAM = BS_file[channel_BAM][:][reprate_FEL]

    return LaserDiode_pump, LaserDiode_unpump, LaserRefDiode_pump, LaserRefDiode_unpump, IzeroFEL, PIPS, Delay, pulse_ids


###    Next: 2 functions to load pump-probe XAS data (energy-delay) (events/pulseIDs)

def load_PumpProbe_events(filename, channel_variable, modulo=2, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)
        
        pulse_ids = BS_file[channel_BS_pulse_ids][:nshots]
        nshots = _get_modulo(pulse_ids,modulo)
        
        FEL = BS_file[channel_Events][:nshots,48]
        Laser = BS_file[channel_Events][:nshots,18]
        Darkshot = BS_file[channel_Events][:nshots,21]
        
        index_pump = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))
        index_unpump = np.logical_and.reduce((FEL, Laser, Darkshot))
 #       print (index_pump, index_unpump)
                
        DataFluo_pump = BS_file[channel_PIPS_fluo][:nshots][index_pump].ravel()
        DataFluo_pump = _average(DataFluo_pump, modulo - 1)
        DataFluo_unpump = BS_file[channel_PIPS_fluo][:nshots][index_unpump].ravel()
        
        DataTrans_pump = BS_file[channel_PIPS_trans][:nshots][index_pump].ravel()
        DataTrans_pump = _average(DataTrans_pump, modulo - 1)
        DataTrans_unpump = BS_file[channel_PIPS_trans][:nshots][index_unpump].ravel()
        
        IzeroFEL_pump = BS_file[channel_Izero][:nshots][index_pump].ravel()
        IzeroFEL_pump = _average(IzeroFEL_pump, modulo - 1)
        IzeroFEL_unpump = BS_file[channel_Izero][:nshots][index_unpump].ravel()
        
        Variable = BS_file[channel_variable][:nshots][index_unpump]
        
        print ("Pump/umpump arrays have {} shots each".format(len(DataFluo_pump), len(DataFluo_unpump)))
             
    return _cut_to_shortest_length(DataFluo_pump, DataFluo_unpump, IzeroFEL_pump, IzeroFEL_unpump, Variable, DataTrans_pump, DataTrans_unpump)


def load_PumpProbe_pulseID(filename, channel_variable, reprateFEL, repratelaser):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        reprate_FEL, reprate_laser = _make_reprates_on_off(pulse_ids, reprateFEL, repratelaser)

        DataFluo_pump = BS_file[channel_PIPS_fluo][:][reprate_laser]
        DataFluo_unpump = BS_file[channel_PIPS_fluo][:][reprate_FEL]

        DataTrans_pump = BS_file[channel_PIPS_trans][:][reprate_laser]
        DataTrans_unpump = BS_file[channel_PIPS_trans][:][reprate_FEL]

        IzeroFEL_pump = BS_file[channel_Izero][:][reprate_laser]
        IzeroFEL_unpump = BS_file[channel_Izero][:][reprate_FEL]

        Variable = BS_file[channel_variable][:][reprate_FEL]

    return DataFluo_pump, DataFluo_unpump, IzeroFEL_pump, IzeroFEL_unpump, Variable, DataTrans_pump, DataTrans_unpump


def load_laserIntensity(filename):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        FEL = BS_file[channel_Events][:,48]
        Laser = BS_file[channel_Events][:,18]
        Darkshot = BS_file[channel_Events][:,21]
        Jungfrau = BS_file[channel_Events][:,40]

        index_light = np.logical_and(Jungfrau,Laser,np.logical_not(Darkshot))

        DataLaser = BS_file[channel_LaserDiode_DIAG][:][index_light]

        PulseIDs = pulse_ids[:][index_light]

    return DataLaser, PulseIDs


def load_FEL_scans(filename, channel_variable, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:nshots]

        FEL = data[channel_Events][:nshots,48]
        index_light = FEL == 1

        DataFEL_t  = data[channel_PIPS_trans][:nshots][index_light]
        DataFEL_f  = data[channel_PIPS_fluo][:nshots][index_light]
        Izero      = data[channel_Izero][:nshots][index_light]
        Laser      = data[channel_LaserDiode][:nshots][index_light]
        Variable   = data[channel_variable][:nshots][index_light]

        PulseIDs = pulse_ids[:nshots][index_light]

    return DataFEL_t, DataFEL_f, Izero, Laser, Variable, PulseIDs


def load_FEL_scans_pulseID(filename, channel_variable, reprateFEL, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:nshots]

        reprate_FEL = _make_reprates_on(pulse_ids, reprateFEL)

        DataFEL_t  = data[channel_PIPS_trans][:nshots][reprate_FEL]
        DataFEL_f  = data[channel_PIPS_fluo][:nshots][reprate_FEL]
        Izero      = data[channel_Izero][:nshots][reprate_FEL]
        Laser      = data[channel_LaserDiode][:nshots][reprate_FEL]
        Variable   = data[channel_variable][:nshots][reprate_FEL]

        PulseIDs = pulse_ids[:nshots][reprate_FEL]

    return DataFEL_t, DataFEL_f, Izero, Laser, Variable, PulseIDs


def load_FEL_pp_pulseID(filename, channel_variable, reprateFEL, repratelaser, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:nshots]

        reprate_FEL, reprate_laser = _make_reprates_on_off(pulse_ids, reprateFEL, repratelaser)

        IzeroFEL_pump = data[channel_Izero][:nshots][reprate_laser]
        IzeroFEL_unpump = data[channel_Izero][:nshots][reprate_FEL]
        Variable = data[channel_variable][:nshots][reprate_FEL]

        PulseIDs = pulse_ids[:nshots][reprate_FEL]

    return IzeroFEL_pump, IzeroFEL_unpump, Variable, PulseIDs



def load_laser_scans(filename):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        Laser = BS_file[channel_Events][:,18]
        index_light = Laser == 1

        DataLaser = BS_file[channel_LaserDiode][:][index_light]
        Position = BS_file[channel_position][:][index_light]

        PulseIDs = pulse_ids[:][index_light]

    return DataLaser, Position, PulseIDs


def load_single_channel(filename, channel, eventCode):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        condition_array = data[channel_Events][:,eventCode]
        condition = condition_array == 1

        DataBS = data[channel][:][condition]
        PulseIDs = data[channel_BS_pulse_ids][:][condition]

    return DataBS, PulseIDs

def load_single_channel_pulseID(filename, channel, reprate):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:]

        condition = _make_reprates_on(pulse_ids, reprate)

        DataBS = data[channel][:][condition]
        PulseIDs = data[channel_BS_pulse_ids][:][condition]

    return DataBS, PulseIDs

def load_single_channel_pp_pulseID(filename, channel, reprateFEL, repratelaser):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:]

        reprate_FEL, reprate_laser = _make_reprates_on_off(pulse_ids, reprateFEL, repratelaser)

        DataBS_ON = data[channel][:][reprate_laser]
        DataBS_OFF = data[channel][:][reprate_FEL]
        PulseIDs_ON = data[channel_BS_pulse_ids][:][reprate_laser]
        PulseIDs_OFF = data[channel_BS_pulse_ids][:][reprate_FEL]

    return DataBS_ON, DataBS_OFF, PulseIDs_ON, PulseIDs_OFF


def load_PSSS_data_from_scans_pulseID(filename, channel_variable, reprateFEL, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:nshots]

        reprate_FEL = _make_reprates_on(pulse_ids, reprateFEL)

        PSSS_center  = data[channel_PSSS_center][:nshots][reprate_FEL]
        PSSS_fwhm    = data[channel_PSSS_fwhm][:nshots][reprate_FEL]
        PSSS_x       = data[channel_PSSS_x][:nshots][reprate_FEL]
        PSSS_y       = data[channel_PSSS_y][:nshots][reprate_FEL]

        PulseIDs = pulse_ids[:nshots][reprate_FEL]

    return PSSS_center, PSSS_fwhm, PSSS_x, PSSS_y, PulseIDs
