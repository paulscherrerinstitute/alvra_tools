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
    nshots = nshots - (nshots % modulo)
    print ("Load {} shots".format(nshots))
    return nshots

def _average(data, modulo):
    data = [np.sum(data[i:i+modulo])/modulo for i in np.arange(0,len(data)-1,modulo)]
    data = np.asarray(data)
    return data

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

def _get_module_map(f):
    detector_name = _get_detector_name(f)
    try:
        module_map = f[f"data/{detector_name}/module_map"][:]
    except:
        return None
    if -1 not in module_map:
        return None
    return module_map


def apply_module_map(image, module_map, original_mask, chip_size=512):
    if module_map is None or -1 not in module_map:
        #print("module map skipped")
        return image, original_mask

    mask = original_mask.copy()
    image_ext = _make_empty_image(image, module_map)

    for i, m in enumerate(module_map):
        if m == -1:
            mask[chip_size * i : chip_size * (i + 1), :] = 1
        else:
            image_ext[chip_size * i : chip_size * (i + 1), :] = image[chip_size * m : chip_size * (m + 1), :]
            mask[chip_size * i : chip_size * (i + 1), :] = original_mask[chip_size * i : chip_size * (i + 1), :]

    return image_ext, mask


def _make_empty_image(image, module_map):
    return np.zeros((512 * len(module_map), 1024), dtype=image.dtype)


def _apply_to_all_images(func, images, *args, **kwargs):
    nshots = len(images)
    one_image = func(images[0],  *args, **kwargs)
    target_dtype = one_image.dtype
    target_shape = one_image.shape
    target_shape = [nshots] + list(target_shape)
    images_corr = np.empty(shape=target_shape, dtype=target_dtype)
    for n, img in enumerate(images):
        images_corr[n] = func(img,  *args, **kwargs)
    return images_corr



def load_JF_data(fname, nshots=None):
    with h5py.File(fname, "r") as f:
        detector_name = _get_detector_name(f)

        data = _get_data(f)
        pulse_ids = data[detector_name + channel_JF_pulse_ids][:nshots].T[0] # pulse_ids comes in a weird shape
        images    = data[detector_name + channel_JF_images][:nshots]
    
    return images, pulse_ids


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


def load_JF_data_on_off(fname, reprate_FEL, reprate_laser, nshots=None):
    images, pulse_ids = load_JF_data(fname, nshots=nshots)

    reprate_on, reprate_off = _make_reprates_on_off(pulse_ids, reprate_FEL, reprate_laser)

    images_on  = images[reprate_on]
    images_off = images[reprate_off]

    return images_on, images_off, pulse_ids


def load_crop_JF_data(fname, roi1, roi2, nshots=None):
    with h5py.File(fname, "r") as f:
        detector_name = _get_detector_name(f)

    if detector_name == "JF02T09V01":
        #print(f"got {detector_name} assuming v01")
        return load_crop_JF_data_v01(fname, roi1, roi2, nshots=nshots)
    else:
        #print(f"got {detector_name} assuming v02")
        return load_crop_JF_data_v02(fname, roi1, roi2, nshots=nshots, detector_name=detector_name)


def load_crop_JF_data_v01(fname, roi1, roi2, nshots=None):
    # v01 does not need geometry correction, lazy load from file is possible
    r10, r11 = make_roi(roi1)
    r20, r21 = make_roi(roi2)

    with h5py.File(fname, "r") as f:
        detector_name = _get_detector_name(f)

        data = _get_data(f)
        pulse_ids = data[detector_name + channel_JF_pulse_ids][:nshots].T[0] # pulse_ids comes in a weird shape

        img_data = data[detector_name + channel_JF_images]
        images_roi1 = img_data[:nshots, r10, r11]
        images_roi2 = img_data[:nshots, r20, r21]

    return images_roi1, images_roi2, pulse_ids


def load_crop_JF_data_v02(fname, roi1, roi2, nshots=None, detector_name="JF02T09V02"):
    # v02 needs geometry correction, cannot load lazily
    images, pulse_ids = load_JF_data(fname, nshots=nshots)

    images = np.stack(ju.apply_geometry(img, detector_name) for img in images)
    images_roi1 = crop_roi(images, roi1)
    images_roi2 = crop_roi(images, roi2)

    return images_roi1, images_roi2, pulse_ids


def load_crop_JF_data_on_off(fname, roi1, roi2, reprate_FEL, reprate_laser,
                             G=None, P=None, pixel_mask=None, highgain=False, nshots=None):
    images, pulse_ids = load_JF_data(fname, nshots=nshots)

    with h5py.File(fname, "r") as f:
        detector_name = _get_detector_name(f)
        module_maps = _get_module_map(f)

    if module_maps is not None:
        print ("Will apply module map:", module_maps[0])
        images_full = []
        for image, module_map in zip(images, module_maps):
            image, pixel_mask = apply_module_map(image, module_map, pixel_mask)
            image = ju.apply_gain_pede(image, G=G, P=P, pixel_mask=pixel_mask, highgain=highgain)
            images_full.append(image)
        images = images_full #np.stack(images_full)
    else:
        print ("All modules are active")
        if any(i is not None for i in (G, P, pixel_mask)):
            images = ju.apply_gain_pede(images, G=G, P=P, pixel_mask=pixel_mask, highgain=highgain)

    #images = np.stack(ju.apply_geometry(img, detector_name) for img in images)
    images = _apply_to_all_images(ju.apply_geometry, images, detector_name)

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

        index_pump = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))
        index_unpump = np.logical_and.reduce((np.logical_not(FEL), Laser, np.logical_not(Darkshot)))

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

    return LaserDiode_pump, LaserDiode_unpump, LaserRefDiode_pump, LaserRefDiode_unpump, IzeroFEL, PIPS, Delay, pulse_ids


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

        Delay = BS_file[channel_delay_SH][:][reprate_laser]
        #Delay = BS_file[channel_laser_pitch][:][index_unpump]

        #BAM = BS_file[channel_BAM][:][reprate_FEL]

    return LaserDiode_pump, LaserDiode_unpump, LaserRefDiode_pump, LaserRefDiode_unpump, IzeroFEL, PIPS, Delay, pulse_ids


###    Next: 2 functions to load pump-probe XAS data (energy-delay) (events/pulseIDs)

def load_PumpProbe_events(filename, channel_variable, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)
        
        pulse_ids = BS_file[channel_BS_pulse_ids][:nshots]
        
        FEL = BS_file[channel_Events][:nshots,48]
        Laser = BS_file[channel_Events][:nshots,18]
        Darkshot = BS_file[channel_Events][:nshots,21]
        
        index_pump = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))
        index_unpump = np.logical_and.reduce((FEL, Laser, Darkshot))
 #       print (index_pump, index_unpump)
                
        DataFluo_pump = BS_file[channel_PIPS_fluo][:nshots][index_pump]
        DataFluo_unpump = BS_file[channel_PIPS_fluo][:nshots][index_unpump]
        
        DataTrans_pump = BS_file[channel_PIPS_trans][:nshots][index_pump]
        DataTrans_unpump = BS_file[channel_PIPS_trans][:nshots][index_unpump]
        
        IzeroFEL_pump = BS_file[channel_Izero4][:nshots][index_pump]
        IzeroFEL_unpump = BS_file[channel_Izero4][:nshots][index_unpump]
        
        Variable = BS_file[channel_variable][:nshots][index_unpump]
             
    return DataFluo_pump, DataFluo_unpump, IzeroFEL_pump, IzeroFEL_unpump, Variable, DataTrans_pump, DataTrans_unpump


def load_PumpProbe_events_test(filename, channel_variable, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)
        
        pulse_ids = BS_file[channel_BS_pulse_ids][:nshots]
        
        FEL = BS_file[channel_Events][:nshots,48]
        Laser = BS_file[channel_Events][:nshots,18]
        Darkshot = BS_file[channel_Events][:nshots,21]
        
        index_pump = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))       
        index_unpump = np.logical_and.reduce((FEL, Laser, Darkshot))
 #       print (index_pump, index_unpump)
                
       
        DataFluo_pump_1 = BS_file[channel_PIPS_fluo][:nshots][index_pump][:333]
        DataFluo_pump_2 = BS_file[channel_PIPS_fluo][:nshots][index_pump][333:666]
        DataFluo_pump = (DataFluo_pump_1 + DataFluo_pump_2)/2
        DataFluo_unpump = BS_file[channel_PIPS_fluo][:nshots][index_unpump][:333]
        
        DataTrans_pump_1 = BS_file[channel_PIPS_trans][:nshots][index_pump][:333]
        DataTrans_pump_2 = BS_file[channel_PIPS_trans][:nshots][index_pump][333:666]
        DataTrans_pump = (DataTrans_pump_1 + DataTrans_pump_2)/2 
        DataTrans_unpump = BS_file[channel_PIPS_trans][:nshots][index_unpump][:333]
        
        IzeroFEL_pump_1 = BS_file[channel_Izero4][:nshots][index_pump][:333]
        IzeroFEL_pump_2 = BS_file[channel_Izero4][:nshots][index_pump][333:666]
        IzeroFEL_pump = (IzeroFEL_pump_1 + IzeroFEL_pump_2)/2
        
        
        IzeroFEL_unpump = BS_file[channel_Izero4][:nshots][index_unpump][:333]
        
        Variable = BS_file[channel_variable][:nshots][index_unpump][:333]
             
    return DataFluo_pump, DataFluo_unpump, IzeroFEL_pump, IzeroFEL_unpump, Variable, DataTrans_pump, DataTrans_unpump


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
        Izero      = data[channel_Izero2][:nshots][index_light]
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

