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


def _make_reprates_on_off(pulse_ids, reprate_FEL, reprate_laser):
    #reprate_off = ((pulse_ids%10 == 0) & (pulse_ids%20 != 0))            #This is for 10 Hz
    #reprate_on  = pulse_ids%20 == 0                                      #This is for 5 Hz
    #reprate_off = ((pulse_ids%4 == 0) & (pulse_ids%8 != 0))              #This is for 25 Hz
    #reprate_on  = pulse_ids%8 == 0                                       #This is for 12.5 Hz
    reprate_on  = _make_reprates_off(pulse_ids, reprate_laser)
    reprate_off = _make_reprates_FEL_on_laser_off(pulse_ids, reprate_FEL, reprate_laser)
    return reprate_on, reprate_off

def _make_reprates_on(pulse_ids, reprate):
    return pulse_ids % (100 / reprate) == 0

def _make_reprates_off(pulse_ids, reprate):
    return np.logical_not(_make_reprates_on(pulse_ids, reprate))

def _make_reprates_FEL_on_laser_off(pulse_ids, reprate_FEL, reprate_laser):
    return np.logical_and(_make_reprates_on(pulse_ids, reprate_FEL), _make_reprates_off(pulse_ids, reprate_laser))



def load_JF_data(fname, max_num_frames=None):
    with h5py.File(fname, "r") as f:
        data = _get_data(f)
        pulse_ids = data[channel_JF_pulse_ids][:max_num_frames].T[0] # pulse_ids comes in a weird shape
        images    = data[channel_JF_images][:max_num_frames]
    return images, pulse_ids


def load_JF_data_on_off(fname, reprate_FEL, reprate_laser, max_num_frames=None):
    images, pulse_ids = load_JF_data(fname, max_num_frames=max_num_frames)

    reprate_on, reprate_off = _make_reprates_on_off(pulse_ids, reprate_FEL, reprate_laser)

    images_on  = images[reprate_on]
    images_off = images[reprate_off]

    return images_on, images_off, pulse_ids


def load_crop_JF_data(fname, roi1, roi2, max_num_frames=None):
    with h5py.File(fname, "r") as f:
        detector_name = f["general/detector_name"].value.decode()

    if detector_name == "JF02T09V01":
        #print(f"got {detector_name} assuming v01")
        return load_crop_JF_data_v01(fname, roi1, roi2, max_num_frames=max_num_frames)
    else:
        #print(f"got {detector_name} assuming v02")
        return load_crop_JF_data_v02(fname, roi1, roi2, max_num_frames=max_num_frames, detector_name=detector_name)


def load_crop_JF_data_v01(fname, roi1, roi2, max_num_frames=None):
    # v01 does not need geometry correction, lazy load from file is possible
    r10, r11 = make_roi(roi1)
    r20, r21 = make_roi(roi2)

    with h5py.File(fname, "r") as f:
        data = _get_data(f)
        pulse_ids = data[channel_JF_pulse_ids][:max_num_frames].T[0] # pulse_ids comes in a weird shape

        img_data = data[channel_JF_images]
        images_roi1 = img_data[:max_num_frames, r10, r11]
        images_roi2 = img_data[:max_num_frames, r20, r21]

    return images_roi1, images_roi2, pulse_ids


def load_crop_JF_data_v02(fname, roi1, roi2, max_num_frames=None, detector_name="JF02T09V02"):
    # v02 needs geometry correction, cannot load lazily
    images, pulse_ids = load_JF_data(fname, max_num_frames=max_num_frames)

    images = np.stack(ju.apply_geometry(img, detector_name) for img in images)
    images_roi1 = crop_roi(images, roi1)
    images_roi2 = crop_roi(images, roi2)

    return images_roi1, images_roi2, pulse_ids


def load_crop_JF_data_on_off(fname, roi1, roi2, reprate_FEL, reprate_laser, G=None, P=None, pixel_mask=None, max_num_frames=None):
    images_roi1, images_roi2, pulse_ids = load_crop_JF_data(fname, roi1, roi2, max_num_frames=max_num_frames)

    reprate_on, reprate_off = _make_reprates_on_off(pulse_ids, reprate_FEL, reprate_laser)

    if any((G, P, pixel_mask)):
        G_roi1 = crop_roi(G, roi1)
        G_roi2 = crop_roi(G, roi2)
        P_roi1 = crop_roi(P, roi1)
        P_roi2 = crop_roi(P, roi2)
        pixel_mask_roi1 = crop_roi(pixel_mask, roi1)
        pixel_mask_roi2 = crop_roi(pixel_mask, roi2)
        images_roi1 = ju.apply_gain_pede(images_roi1, G=G_roi1, P=P_roi1, pixel_mask=pixel_mask_roi1, highgain=False)
        images_roi2 = ju.apply_gain_pede(images_roi2, G=G_roi2, P=P_roi2, pixel_mask=pixel_mask_roi2, highgain=False)

    images_on_roi1  = images_roi1[reprate_on]
    images_on_roi2  = images_roi2[reprate_on]
    images_off_roi1 = images_roi1[reprate_off]
    images_off_roi2 = images_roi2[reprate_off]

    return images_on_roi1, images_on_roi2, images_off_roi1, images_off_roi2, pulse_ids


###    Next: 2 functions to load pump-probe YAG data (events/pulseIDs)


def load_YAG(filename):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        FEL = BS_file[channel_Events][:,48]
        Laser = BS_file[channel_Events][:,18]
        Darkshot = BS_file[channel_Events][:,21]

        index_pump = np.logical_and(FEL, Laser, np.logical_not(Darkshot))
        index_unpump = np.logical_and(np.logical_not(FEL), Laser, np.logical_not(Darkshot))

        LaserDiode_pump = BS_file[channel_LaserDiode][:][index_pump]
        LaserDiode_unpump = BS_file[channel_LaserDiode][:][index_unpump]
        LaserRefDiode_pump = BS_file[channel_Laser_refDiode][:][index_pump]
        LaserRefDiode_unpump = BS_file[channel_Laser_refDiode][:][index_unpump]
        IzeroFEL = BS_file[channel_Izero][:][index_pump]
        PIPS = BS_file[channel_PIPS_trans][:][index_pump]

        Delay = BS_file[channel_delay][:][index_unpump]
        #Delay = BS_file[channel_laser_pitch][:][index_unpump]

        #BAM = BS_file[channel_BAM][:][index_pump]

    return LaserDiode_pump, LaserDiode_unpump, LaserRefDiode_pump, LaserRefDiode_unpump, IzeroFEL, PIPS, Delay, pulse_ids


def load_YAG_pulseID(filename, reprateFEL, repratelaser):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        reprate_laser, reprate_FEL = _make_reprates_ON_OFF(pulse_ids, reprate_FEL, reprate_laser)
        
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

def load_XAS(filename, channel_variable):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)
        
        pulse_ids = BS_file[channel_BS_pulse_ids][:]
        
        FEL = BS_file[channel_Events][:,48]
        Laser = BS_file[channel_Events][:,18]
        Darkshot = BS_file[channel_Events][:,21]
        
        index_pump = np.logical_and(FEL, Laser, np.logical_not(Darkshot))
        index_unpump = np.logical_and(FEL, np.logical_not(Laser), np.logical_not(Darkshot))
                
        DataFluo_pump = BS_file[channel_PIPS_fluo][:][index_pump]
        DataFluo_unpump = BS_file[channel_PIPS_fluo][:][index_unpump]
        
        DataTrans_pump = BS_file[channel_PIPS_trans][:][index_pump]
        DataTrans_unpump = BS_file[channel_PIPS_trans][:][index_unpump]
        
        IzeroFEL_pump = BS_file[channel_Izero][:][index_pump]
        IzeroFEL_unpump = BS_file[channel_Izero][:][index_unpump]
        
        Variable = BS_file[channel_variable][:][index_unpump]
             
    return DataFluo_pump, DataFluo_unpump, IzeroFEL_pump, IzeroFEL_unpump, Variable, DataTrans_pump, DataTrans_unpump


def load_XAS_pulseID(filename, channel_variable, reprateFEL, repratelaser):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)
        
        pulse_ids = BS_file[channel_BS_pulse_ids][:]
        
        reprate_laser, reprate_FEL = _make_reprates_ON_OFF(pulse_ids, reprate_FEL, reprate_laser)
        
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


def load_FEL_scans(filename, channel_variable):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        FEL = BS_file[channel_Events][:,48]
        index_light = FEL == 1

        DataFEL = BS_file[channel_PIPS_trans][:][index_light]
        Izero = BS_file[channel_Izero][:][index_light]
        Variable = BS_file[channel_variable][:][index_light]

        PulseIDs = pulse_ids[:][index_light]

    return DataFEL, Izero, Variable, PulseIDs


def load_FEL_scans_pulseID(filename, channel_variable, reprateFEL, max_num_shots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:max_num_shots]

        reprate_FEL = _make_reprate_on(reprateFEL)

        DataFEL = data[channel_PIPS_trans][:max_num_shots][reprate_FEL]
        Izero = data[channel_Izero][:max_num_shots][reprate_FEL]
        Variable = data[channel_variable][:max_num_shots][reprate_FEL]

        PulseIDs = pulse_ids[:max_num_shots][reprate_FEL]

    return DataFEL, Izero, Variable, PulseIDs


def load_FEL_pp_pulseID(filename, channel_variable, reprateFEL, reparatelaser, max_num_shots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:max_num_shots]

        reprate_laser, reprate_FEL = _make_reprates_on_off(pulse_ids, reprate_FEL, reprate_laser)

        IzeroFEL_pump = data[channel_Izero][:max_num_shots][reprate_laser]
        IzeroFEL_unpump = data[channel_Izero][:max_num_shots][reprate_FEL]
        Variable = data[channel_variable][:max_num_shots][reprate_FEL]

        PulseIDs = pulse_ids[:max_num_shots][reprate_FEL]

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
        BS_file = _get_data(BS_file)

        condition_array = BS_file[channel_Events][:,eventCode]
        condition = condition_array == 1

        DataBS = BS_file[channel][:][condition]
        PulseIDs = BS_file[channel_BS_pulse_ids][:][condition]

    return DataBS, PulseIDs





