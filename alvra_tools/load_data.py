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



def load_JF_data(fname, max_num_frames=None):
    with h5py.File(fname, "r") as f:
        data = _get_data(f)
        pulse_ids = data[channel_JF_pulse_ids][:max_num_frames].T[0] # pulse_ids comes in a weird shape
        images    = data[channel_JF_images][:max_num_frames]
    return images, pulse_ids


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


load_crop_JF_data = load_crop_JF_data_v01 #TODO should actually switch between versions based on f["general/detector_name"], default to v01 for now


def load_crop_JF_data_v02(fname, roi1, roi2, max_num_frames=None):
    # v02 needs geometry correction, cannot load lazily
    images, pulse_ids = load_JF_data(fname, max_num_frames=max_num_frames)

    images = np.stack(ju.apply_geometry(img, "JF02T09V02") for img in images)
    images_roi1 = crop_roi(images, roi1)
    images_roi2 = crop_roi(images, roi2)

    return images_roi1, images_roi2, pulse_ids


#TODO: delete? replaced by load_crop_JF_data_v01
#def load_JF_data_crop1(filename, roi1, roi2):
#    with h5py.File(filename, 'r') as JF_file:
#        pulse_ids = JF_file[channel_pulse_idsJF][:]
#        pulse_ids = np.reshape(pulse_ids, (pulse_ids.size,)) # .ravel()
#        # ^what is this doing?!

#        image_JF = JF_file[channel_JFimages]

#        image_roi1 = image_JF[:, roi1[0][0]:roi1[0][1], roi1[1][0]:roi1[1][1]]
#        image_roi2 = image_JF[:, roi2[0][0]:roi2[0][1], roi2[1][0]:roi2[1][1]]

#    return image_roi1, image_roi2, pulse_ids


def load_JF_data_crop_lolo(filename, roi1, roi2):
    with h5py.File(filename, 'r') as JF_file:
        JF_file = _get_data(JF_file)

        pulse_ids = JF_file[channel_pulse_idsJF][:]
        pulse_ids = np.reshape(pulse_ids, (pulse_ids.size,))

        #reprate_FEL = ((pulse_ids%10 == 0) & (pulse_ids%20 != 0))              #This is for 10 Hz
        #reprate_laser = pulse_ids%20 == 0                                      #This is for 5 Hz
        reprate_FEL = ((pulse_ids%4 == 0) & (pulse_ids%8 != 0))                #This is for 25 Hz
        reprate_laser = pulse_ids%8 == 0                                       #This is for 12.5 Hz

        image_JF_ON = JF_file[channel_JFimages][reprate_laser,:,:][...]
        image_JF_OFF = JF_file[channel_JFimages][:,:,:][reprate_FEL]

        pulse_ids_ON = pulse_ids[reprate_laser]
        pulse_ids_OFF = pulse_ids[reprate_FEL]

        image_roi1_ON = image_JF_ON[:, roi1[0][0]:roi1[0][1], roi1[1][0]:roi1[1][1]]
        image_roi2_ON = image_JF_ON[:, roi2[0][0]:roi2[0][1], roi2[1][0]:roi2[1][1]]

        image_roi1_OFF = image_JF_OFF[:, roi1[0][0]:roi1[0][1], roi1[1][0]:roi1[1][1]]
        image_roi2_OFF = image_JF_OFF[:, roi2[0][0]:roi2[0][1], roi2[1][0]:roi2[1][1]]

    return image_roi1_ON, image_roi1_OFF, image_roi2_ON, image_roi2_OFF, pulse_ids_ON, pulse_ids_OFF


def load_YAG_events(filename):
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

        reprate_FEL = pulse_ids%(100 / reprateFEL) == 0
        reprate_laser = np.logical_and((pulse_ids%(100 / repratelaser) == 0), (pulse_ids%(100 / reprateFEL) != 0))

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


def load_FEL_scans_pulseID(filename, channel_variable, reprateFEL):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        reprate_FEL = pulse_ids%(100 / reprateFEL) == 0

        DataFEL = BS_file[channel_PIPS_trans][:][reprate_FEL]
        Izero = BS_file[channel_Izero][:][reprate_FEL]
        Variable = BS_file[channel_variable][:][reprate_FEL]

        PulseIDs = pulse_ids[:][reprate_FEL]

    return DataFEL, Izero, Variable, PulseIDs


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





