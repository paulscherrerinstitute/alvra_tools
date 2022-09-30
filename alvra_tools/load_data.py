import h5py
import numpy as np

import jungfrau_utils as ju
from sfdata import SFDataFile, SFDataFiles
from collections import namedtuple
from glob import glob

from .channels import *
from .utils import crop_roi, make_roi
from sfdata.utils import cprint, print_line


def _correct_path(f, pgroup):
    newf = f.replace("/gpfs/photonics/swissfel/raw/alvra-staff/","/sf/alvra/data/").replace("/gpfs/photonics/swissfel/raw/alvra/","/sf/alvra/data/")
    if (f'/{pgroup}/raw/') not in newf:
        newf =  newf.replace(f'/{pgroup}/',f'/{pgroup}/raw/')
    return newf

def _get_data(f):
    if "data" in f:
        return f["data"]
    else:
        return f

def _get_modulo(pulse_ids, modulo):
    nshots = len(pulse_ids)
#    print ("Found {} shots in the file".format(nshots))
    nshots = nshots - (nshots % modulo)
#    print ("Load {} shots".format(nshots))
    return nshots

def _get_modulo_length(length, modulo):
    nshots = length
    nshots = nshots - (nshots % modulo)
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

def _get_reprates_from_file(JF_file, nshots):
    all_files = JF_file.replace('.{}.h5'.format(_get_detector_name(JF_file)),'.*.h5')
    data = SFDataFiles(all_files)
    channel_list = [_get_detector_name(JF_file), channel_Events]
     
    subset = data[channel_list]
    subset.print_stats(show_complete=True)
    subset.drop_missing()
        
    Event_code = subset[channel_Events].data
    FEL_raw  = Event_code[:,13] #Event 13: changed from 12 on June 22
    Ppicker  = Event_code[:,200] 
    Laser    = Event_code[:,18]
    Darkshot = Event_code[:,21]
    
    #FEL = np.logical_and(FEL_raw, np.logical_not(Ppicker))
    FEL = FEL_raw
    
    if Darkshot.mean()==0:
        laser_reprate = (1 / Laser.mean() - 1).round().astype(int)
        index_light = np.logical_and.reduce((FEL, Laser))[:nshots]
        index_dark  = np.logical_and.reduce((FEL, np.logical_not(Laser)))[:nshots]
    else:
        laser_reprate = (Laser.mean() / Darkshot.mean() - 1).round().astype(int)
        index_light = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))[:nshots]
        index_dark = np.logical_and.reduce((FEL, Laser, Darkshot))[:nshots]
    pids_light = subset[channel_Events].pids[:nshots][index_light]
    pids_dark  = subset[channel_Events].pids[:nshots][index_dark]
    return index_light, index_dark, pids_light, pids_dark


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
    with h5py.File(f, "r") as ff:
        group = ff['general']
        detector = group['detector_name'][()].decode()
    return detector


def _make_empty_image(image, module_map):
    return np.zeros((512 * len(module_map), 1024), dtype=image.dtype)

def remove_JF_from_scan(scan):
    for i, files in enumerate(scan.files):
        new_files = []
        for sc_file in files:
            if "JF" in sc_file:
                continue
            new_files.append(sc_file)
        scan.files[i] = new_files
    return scan

def get_timezero_NBS(json_file):
    from sfdata import SFScanInfo
    scan = SFScanInfo(json_file)
    for file in scan.files[0]:
        if 'BSDATA' in file:
            bsfile= str(file)
    with SFDataFile(bsfile) as sfd:
        ch = sfd['SARES11-CVME-EVR0:DUMMY_PV2_NBS']
        t0mm = ch.data[0]
    return t0mm

#def get_timezero_NBS(json_file):
#    from sfdata import SFScanInfo
#    scan = SFScanInfo(json_file)
#    fn = scan.files[0][0].replace('.BSDATA.h5','*').replace('.PVDATA.h5','*').replace('.PVCHANNELS.h5','*').replace('.CAMERAS.h5','*').replace('.*JF*.h5','*')
#    with SFDataFiles(fn) as sfd:
#        ch = sfd['SARES11-CVME-EVR0:DUMMY_PV2_NBS']
#        t0mm = ch.data[0]
#    return t0mm

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

########################################################################################

def load_JF_static(fname, pgroup, gain_file=None, pedestal_file=None, nshots=None):
    fname = _correct_path(fname, pgroup)
    with ju.File(fname, gain_file=gain_file, pedestal_file=pedestal_file) as juf:
        images = juf[:nshots]
        pulse_ids = juf["pulse_id"][:nshots].T[0]
    return (images, pulse_ids)

########################################################################################

def load_JF_static_batches(fname, pgroup, gain_file=None, pedestal_file=None, nshots=None, batch_size = 1000):
    fname = _correct_path(fname, pgroup)
    with ju.File(fname, gain_file=gain_file, pedestal_file=pedestal_file) as juf:
        pulse_ids = juf["pulse_id"][:nshots].T[0]

        if (nshots is None):
           nshots = pulse_ids.shape[0]
        
        if (nshots < batch_size):
           images = juf[:nshots]

        else:
           n_images = pulse_ids.shape[0]
           images= []
        
           print ('Total images = {}, load them in batches of {}'.format(n_images, batch_size))
        
           for ind in range(0, n_images, batch_size):
               batch_slice = slice(ind, min(ind + batch_size, n_images))
            
               print ('Load batch = {}'.format(batch_slice))
            
               batch_images = juf[batch_slice, :, :]
            
               images.extend(batch_images)
               del batch_images
    
    images = np.asarray(images)

    return (images, pulse_ids)

########################################################################################

def load_JF_pp_batches(fname, pgroup, reprate_FEL=100, reprate_laser=50, gain_file=None, pedestal_file=None, nshots=None, batch_size = 1000):
    fname = _correct_path(fname, pgroup)
    with ju.File(fname, gain_file=gain_file, pedestal_file=pedestal_file) as juf:
        pulse_ids = juf["pulse_id"][:nshots].T[0]

        if (nshots is None):
           nshots = pulse_ids.shape[0]
        
        if (nshots < batch_size):
           images = juf[:nshots]

        else:
           n_images = pulse_ids.shape[0]
           images= []
        
           print ('Total images = {}, load them in batches of {}'.format(n_images, batch_size))
        
           for ind in range(0, n_images, batch_size):
               batch_slice = slice(ind, min(ind + batch_size, n_images))
            
               print ('Load batch = {}'.format(batch_slice))
            
               batch_images = juf[batch_slice, :, :]
            
               images.extend(batch_images)
               del batch_images
    
    images = np.asarray(images)
    
    reprate_on, reprate_off,_,_ = _get_reprates_from_file(fname, nshots)
    #reprate_on, reprate_off = _make_reprates_on_off(pulse_ids, reprate_FEL, reprate_laser)
    
    images_on = images[reprate_on]
    images_off = images[reprate_off]
    pulse_ids_on    = pulse_ids[reprate_on]
    pulse_ids_off   = pulse_ids[reprate_off]

    return (images_on,images_off,pulse_ids_on,pulse_ids_off)

########################################################################################

def load_and_crop_JF_static_batches(fname, pgroup, roi1, roi2, roi3, roi4,
                             gain_file=None, pedestal_file=None, nshots=None, batch_size = 1000):
    fname = _correct_path(fname, pgroup)
    with ju.File(fname, gain_file=gain_file, pedestal_file=pedestal_file) as juf:
        pulse_ids = juf["pulse_id"][:nshots].T[0]
        n_images = pulse_ids.shape[0]
        images_roi1 = []
        images_roi2 = []
        images_roi3 = []
        images_roi4 = []   
        
        print ('Total images = {}, load them in batches of {}'.format(n_images, batch_size))
        
        for ind in range(0, n_images, batch_size):
            batch_slice = slice(ind, min(ind + batch_size, n_images))
            
            print ('Load batch = {}'.format(batch_slice))
            
            batch_images = juf[batch_slice, :, :]
            
            images_roi1.extend(crop_roi(batch_images, roi1))
            images_roi2.extend(crop_roi(batch_images, roi2))
            images_roi3.extend(crop_roi(batch_images, roi3))
            images_roi4.extend(crop_roi(batch_images, roi4))
            
            del batch_images

    images_roi1 = np.asarray(images_roi1)
    images_roi2 = np.asarray(images_roi2)
    images_roi3 = np.asarray(images_roi3)
    images_roi4 = np.asarray(images_roi4)
   
    return images_roi1, images_roi2, images_roi3, images_roi4, pulse_ids

########################################################################################

def load_and_crop_JF_pp_batches_4rois(fname, pgroup, roi1, roi2, roi3, roi4, reprate_FEL=100, reprate_laser=50, 
                         gain_file=None, pedestal_file=None, nshots=None, batch_size = 1000):
    fname = _correct_path(fname, pgroup)
    with ju.File(fname, gain_file=gain_file, pedestal_file=pedestal_file) as juf:
        
        pulse_ids = juf["pulse_id"][:nshots].T[0]
        n_images = pulse_ids.shape[0]
        images_roi1 = []
        images_roi2 = []
        images_roi3 = []
        images_roi4 = []   
        
        print ('Total images = {}, load them in batches of {}'.format(n_images, batch_size))
        
        for ind in range(0, n_images, batch_size):
            batch_slice = slice(ind, min(ind + batch_size, n_images))
            
            print ('Load batch = {}'.format(batch_slice))
            
            batch_images = juf[batch_slice, :, :]
            
            images_roi1.extend(crop_roi(batch_images, roi1))
            images_roi2.extend(crop_roi(batch_images, roi2))
            images_roi3.extend(crop_roi(batch_images, roi3))
            images_roi4.extend(crop_roi(batch_images, roi4))
            
            del batch_images
            
    images_roi1 = np.asarray(images_roi1)
    images_roi2 = np.asarray(images_roi2)
    images_roi3 = np.asarray(images_roi3)
    images_roi4 = np.asarray(images_roi4)

    reprate_on, reprate_off,_,_ = _get_reprates_from_file(fname, nshots)
    #reprate_on, reprate_off = _make_reprates_on_off(pulse_ids, reprate_FEL, reprate_laser)

    images_on_roi1  = images_roi1[reprate_on]
    images_on_roi2  = images_roi2[reprate_on]
    images_off_roi1 = images_roi1[reprate_off]
    images_off_roi2 = images_roi2[reprate_off]
    images_on_roi3  = images_roi3[reprate_on]
    images_on_roi4  = images_roi4[reprate_on]
    images_off_roi3 = images_roi3[reprate_off]
    images_off_roi4 = images_roi4[reprate_off]
    pulse_ids_on    = pulse_ids[reprate_on]
    pulse_ids_off   = pulse_ids[reprate_off]

    return images_on_roi1, images_on_roi2, images_on_roi3, images_on_roi4, pulse_ids_on, images_off_roi1, images_off_roi2, images_off_roi3, images_off_roi4, pulse_ids_off

########################################################################################


def load_JF_data_4rois_on_off(fname, pgroup, index_light, index_dark, roi1, roi2, roi3, roi4,
                             gain_file=None, pedestal_file=None, nshots=None):
    fname = _correct_path(fname, pgroup)
    with ju.File(fname, gain_file=gain_file, pedestal_file=pedestal_file) as juf:
        images = juf[:nshots]
        pulse_ids = juf["pulse_id"][:nshots].T[0]

    images_roi1     = crop_roi(images, roi1)
    images_on_roi1  = images_roi1[index_light]
    images_off_roi1 = images_roi1[index_dark]
    
    images_roi2     = crop_roi(images, roi2)
    images_on_roi2  = images_roi2[index_light]
    images_off_roi2 = images_roi2[index_dark]
    
    images_roi3     = crop_roi(images, roi3)
    images_on_roi3  = images_roi3[index_light]
    images_off_roi3 = images_roi3[index_dark]
    
    images_roi4     = crop_roi(images, roi4)
    images_on_roi4  = images_roi4[index_light]
    images_off_roi4 = images_roi4[index_dark]
    
    pulse_ids_on    = pulse_ids[index_light]
    pulse_ids_off   = pulse_ids[index_dark]
    
    return images_on_roi1, images_off_roi1, images_on_roi2, images_off_roi2, images_on_roi3, images_off_roi3, images_on_roi4, images_off_roi4, pulse_ids_on, pulse_ids_off

########################################################################################

def load_crop_JF_batches_on_off_2rois(fname, roi1, roi2, reprate_FEL, reprate_laser, 
                         gain_file=None, pedestal_file=None, nshots=None, batch_size = 1000):

    with ju.File(fname, gain_file=gain_file, pedestal_file=pedestal_file) as juf:
        
        pulse_ids = juf["pulse_id"][:nshots].T[0]
        n_images = pulse_ids.shape[0]
        images_roi1 = []
        images_roi2 = []   
        
        print ('Total images = {}, load them in batches of {}'.format(n_images, batch_size))
        
        for ind in range(0, n_images, batch_size):
            batch_slice = slice(ind, min(ind + batch_size, n_images))
            
            print ('Load batch = {}'.format(batch_slice))
            
            batch_images = juf[batch_slice, :, :]
            
            images_roi1.extend(crop_roi(batch_images, roi1))
            images_roi2.extend(crop_roi(batch_images, roi2))
            
            del batch_images
            
    images_roi1 = np.asarray(images_roi1)
    images_roi2 = np.asarray(images_roi2)

    reprate_on, reprate_off = _make_reprates_on_off(pulse_ids, reprate_FEL, reprate_laser)

    images_on_roi1  = images_roi1[reprate_on]
    images_on_roi2  = images_roi2[reprate_on]
    images_off_roi1 = images_roi1[reprate_off]
    images_off_roi2 = images_roi2[reprate_off]
    pulse_ids_on    = pulse_ids[reprate_on]
    pulse_ids_off   = pulse_ids[reprate_off]

    return images_on_roi1, images_on_roi2, pulse_ids_on, images_off_roi1, images_off_roi2, pulse_ids_off

########################################################################################


def load_crop_JF_batches_on_off2(fname, roi1, roi2, reprate_FEL, reprate_laser, 
                                gain_file=None, pedestal_file=None, nshots=None, batch_size = 1000):
    
    with ju.File(fname, gain_file=gain_file, pedestal_file=pedestal_file) as juf:
        
        pulse_ids = juf["pulse_id"][:nshots].T[0]
        n_images = pulse_ids.shape[0]

        first_images = juf[0, :, :]
        first_image_roi1 = crop_roi(first_images, roi1)
        first_image_roi2 = crop_roi(first_images, roi2)

        images_roi1 = np.empty((n_images, *first_image_roi1.shape))
        images_roi2 = np.empty((n_images, *first_image_roi2.shape))

        print ('Total images = {}, load them in batches of {}'.format(n_images, batch_size))
        
        for ind in range(0, n_images, batch_size):
            batch_slice = slice(ind, min(ind + batch_size, n_images))
            
            print ('Load batch = {}'.format(batch_slice))
            
            batch_images = juf[batch_slice, :, :]

            images_roi1[batch_slice] = crop_roi(batch_images, roi1)
            images_roi2[batch_slice] = crop_roi(batch_images, roi2)

    reprate_on, reprate_off = _make_reprates_on_off(pulse_ids, reprate_FEL, reprate_laser)
    images_on_roi1  = images_roi1[reprate_on]
    images_on_roi2  = images_roi2[reprate_on]
    images_off_roi1 = images_roi1[reprate_off]
    images_off_roi2 = images_roi2[reprate_off]
    pulse_ids_on    = pulse_ids[reprate_on]
    pulse_ids_off   = pulse_ids[reprate_off]
    
    return images_on_roi1, images_on_roi2, pulse_ids_on, images_off_roi1, images_off_roi2, pulse_ids_off

########################################################################################

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

########################################################################################

def load_crop_JF_data(fname, roi1, roi2,roi3,roi4,
                             gain_file=None, pedestal_file=None, nshots=None):

    with ju.File(fname, gain_file=gain_file, pedestal_file=pedestal_file) as juf:
        images = juf[:nshots]
        pulse_ids = juf["pulse_id"][:nshots].T[0]

    images_roi1 = crop_roi(images, roi1)
    images_roi2 = crop_roi(images, roi2)
    images_roi3 = crop_roi(images, roi3)
    images_roi4 = crop_roi(images, roi4)
    
    return images_roi1, images_roi2, images_roi3, images_roi4, pulse_ids

########################################################################################

def read_and_crop_jf(channel_jf, roi1=None, roi2=None, roi3=None, roi4=None, batch_size=100):
    images_roi1 = [] # performance here can be improved by using a np array and fill it, see ch.apply_in_batches()
    images_roi2 = []
    images_roi3 = []
    images_roi4 = []
    
    for indices, batch in channel_jf.in_batches(batch_size):
        images_roi1.extend(crop_roi(batch, roi1))
        images_roi2.extend(crop_roi(batch, roi2))
        images_roi3.extend(crop_roi(batch, roi3))
        images_roi4.extend(crop_roi(batch, roi4))
        
    images_roi1 = np.asarray(images_roi1)
    images_roi2 = np.asarray(images_roi2)
    images_roi3 = np.asarray(images_roi3)
    images_roi4 = np.asarray(images_roi4)
    
    return images_roi1, images_roi2, images_roi3, images_roi4






# ##    Next: 3 functions to load COMPACT BS data

def check_file_and_data(filename, nshots=None):
    exists = os.path.isfile(filename)
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)
        checkData = data["SAR-CVME-TIFALL5:EvtSet/is_data_present"][:nshots]
        dataOK = checkData.all()
        
        combined = exists & dataOK
        
        return combined

########################################################################################

def check_files_and_data(data, nshots=None):
    #with SFDataFiles(filenames) as data:
    try:
        checkData = data["SAR-CVME-TIFALL5:EvtSet"]._group["is_data_present"][:nshots] #FIX!
    except KeyError:
        return False
    dataOK = checkData.all()
    return dataOK

########################################################################################

def check_channels(data, asked_channels, name):
    asked_channels = set(asked_channels)
    inters = asked_channels.intersection(data.names)
    check = (inters == asked_channels)
    if not check:
        print_line()
        cprint(f"asked channels \"{name}\" has unknown channels:", set(asked_channels) - set(data.names), color="blue")
        print_line()
        asked_channels = list(inters)
    return list(asked_channels)

########################################################################################

def load_data_compact(channel_list, data):
    #with SFDataFiles(datafiles) as data:#, SFDataFile(filename_camera) as data_camera:
    channel_list = check_channels(data, channel_list, "channels")

    channel_list_complete = [channel_Events] + channel_list

    subset = data[channel_list_complete]
    subset.print_stats(show_complete=True)
    subset.drop_missing()

    Event_code = subset[channel_Events].data
    FEL_raw  = Event_code[:,13] #Event 13: changed from 12 on June 22
    Ppicker = Event_code[:,200]    

    #FEL = np.logical_and(FEL_raw, np.logical_not(Ppicker))
    FEL = FEL_raw

    index_light = FEL == 1

    Deltap = (1 / FEL.mean()).round().astype(int) #Get the FEL rep rate from the Event code
    print ('FEL rep rate is {} Hz'.format(100 / Deltap))

    result = {}
    for ch in channel_list_complete:
        dat = subset[ch].data
        pids = subset[ch].pids[index_light]
        ch_out   = dat[index_light]
        result[ch] = ch_out

    return result, pids

########################################################################################

def load_data_compact_JF(channel_list, data, roi1, roi2, roi3, roi4):
    #with SFDataFiles(datafiles) as data:#, SFDataFile(filename_camera) as data_camera:
    channel_list = check_channels(data, channel_list, "channels")

    channel_list_complete = [channel_Events] + channel_list

    subset = data[channel_list_complete]
    subset.print_stats(show_complete=True)
    subset.drop_missing()

    Event_code = subset[channel_Events].data
    FEL = Event_code[:,13] #Event 13: changed from 12 on June 22

    index_light = FEL == 1

    Deltap = (1 / FEL.mean()).round().astype(int) #Get the FEL rep rate from the Event code
    print ('FEL rep rate is {} Hz'.format(100 / Deltap))

    result = {}
    for chname in channel_list_complete:
        ch = subset[chname]
        
        if "JF" in chname: # or something more clever here!
            data_imgs1, data_imgs2, data_imgs3, data_imgs4 = read_and_crop_jf(ch, roi1, roi2, roi3, roi4)
            result["JFroi1"]=data_imgs1[index_light]
            result["JFroi2"]=data_imgs2[index_light]
            result["JFroi3"]=data_imgs3[index_light]
            result["JFroi4"]=data_imgs4[index_light]
        else:
            result[chname] = ch.data[index_light]

    return result

########################################################################################

def load_data_compact_FEL_pump(channels_pump_unpump, channels_pump, data):  
    #with SFDataFiles(datafiles) as data:
    channels_pump_unpump = check_channels(data, channels_pump_unpump, "pump unpump")
    channels_pump = check_channels(data, channels_pump, "pump")

    channels_unpump = channels_pump_unpump

    subset_unpump = data[channels_pump_unpump]
    subset_unpump.print_stats(show_complete=True)
    subset_unpump.drop_missing()

    Event_code = subset_unpump[channel_Events].data

    FEL_raw  = Event_code[:,13] #Event 13: changed from 12 on June 22
    Ppicker  = Event_code[:,200] 
    Laser    = Event_code[:,18]
    Darkshot = Event_code[:,21]

    #FEL = np.logical_and(FEL_raw, Ppicker)
    FEL = FEL_raw

    if Darkshot.mean()==0:
        laser_reprate = Laser.mean().round().astype(int)
    else:
        laser_reprate = (Laser.mean() / Darkshot.mean() - 1).round().astype(int)

    index_dark_before = np.append([True], np.logical_not(Darkshot))[:-1]
    index_light       = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot), index_dark_before))
    index_dark        = np.logical_and.reduce((np.logical_not(FEL), Laser, np.logical_not(Darkshot), index_dark_before))

    #print(np.shape(index_dark), np.shape(index_light))

    Deltap_FEL = (1 / FEL.mean()).round().astype(int) #Get the FEL rep rate from the Event code
    FEL_reprate = 100 / Deltap_FEL
    print ('Pump rep rate (FEL) is {} Hz'.format(FEL_reprate))

    index_probe = np.logical_and.reduce((Laser, np.logical_not(Darkshot)))

    Deltap_laser = (1 / index_probe.mean()).round().astype(int) #Get the laser rep rate from the Event code
    print ('Probe rep rate (laser) is {} Hz'.format(100 / Deltap_laser))

    result_unpump ={}
    for ch in channels_unpump:
        result_unpump[ch] = subset_unpump[ch].data[index_dark]

    actual_pids_unpump = subset_unpump[channel_Events].pids[index_dark]
    
    #with SFDataFiles(datafiles) as data:
        
    subset_pump = data[channels_pump]
    #subset_pump.print_stats(show_complete=True)
    subset_pump.drop_missing()

    result_pump = {}
    for ch in channels_pump:
        result_pump[ch] = subset_pump[ch].data

    actual_pids_pump = subset_pump[channel_Events].pids
        
    wanted_pids_pump = actual_pids_unpump + Deltap_laser
    final_pids_pump, ind_pump, _ = np.intersect1d(actual_pids_pump, wanted_pids_pump, return_indices=True)
    final_pids_unpump = final_pids_pump - Deltap_laser
    _, _, ind_unpump = np.intersect1d(final_pids_unpump, actual_pids_unpump, return_indices=True)
    
    for ch in channels_pump:
        result_pump[ch] = result_pump[ch][ind_pump]
    
    for ch in channels_unpump:
        result_unpump[ch] = result_unpump[ch][ind_unpump]
    
    ppdata = namedtuple("PPData", ["pump", "unpump"])
    result_pp = {}
    shared_channels = set(channels_pump).intersection(channels_unpump)
    for ch in shared_channels:
        result_pp[ch] = ppdata(pump=result_pump[ch], unpump=result_unpump[ch])
                
#    return result_pp, result_pump, FEL_reprate, laser_reprate
    return result_pp, result_pump, final_pids_pump, final_pids_unpump

########################################################################################

def load_data_compact_laser_pump(channels_pump_unpump, channels_FEL, data):
    #with SFDataFiles(datafiles) as data:
    channels_pump_unpump = check_channels(data, channels_pump_unpump, "pump unpump")
    channels_FEL = check_channels(data, channels_FEL, "FEL")

    subset_FEL = data[channels_FEL]
    subset_FEL.print_stats(show_complete=True)
    
#     subset_FEL.drop_missing()
    
    Event_code = subset_FEL[channel_Events].data
    FEL = Event_code[:,13] #Event 13: changed from 12 on June 22
    
    Deltap_FEL = (1 / FEL.mean()).round().astype(int) #Get the FEL rep rate from the Event code
    FEL_reprate = 100 / Deltap_FEL
    print ('Probe rep rate (FEL) is {} Hz'.format(FEL_reprate))
    
    subset_FEL.drop_missing()

    Event_code = subset_FEL[channel_Events].data
    
    FEL_raw  = Event_code[:,13] #Event 13: changed from 12 on June 22
    Ppicker  = Event_code[:,200]
    Laser    = Event_code[:,18]
    Darkshot = Event_code[:,21]

    #FEL = np.logical_and(FEL_raw, np.logical_not(Ppicker))
    FEL = FEL_raw

    if Darkshot.mean()==0:
        laser_reprate = (1 / Laser.mean() - 1).round().astype(int)
        index_light = np.logical_and.reduce((FEL, Laser))
        index_dark  = np.logical_and.reduce((FEL, np.logical_not(Laser)))
    else:
        laser_reprate = (Laser.mean() / Darkshot.mean() - 1).round().astype(int)

        index_light = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))
        index_dark = np.logical_and.reduce((FEL, Laser, Darkshot))

    #index_probe = np.logical_and.reduce((Laser, np.logical_not(Darkshot)))

    #Deltap_FEL = (1 / FEL.mean()).round().astype(int) #Get the FEL rep rate from the Event code
    #FEL_reprate = 100 / Deltap_FEL
    #print ('Probe rep rate (FEL) is {} Hz'.format(FEL_reprate))

    print ('Pump scheme is {}:1'.format(laser_reprate))

    result_pp = {}
    for ch in channels_pump_unpump:
        ch_pump   = subset_FEL[ch].data[index_light]
        pids_pump   = subset_FEL[ch].pids[index_light]

        ch_unpump = subset_FEL[ch].data[index_dark]
        pids_unpump = subset_FEL[ch].pids[index_dark]

        correct_pids_pump   = pids_unpump + Deltap_FEL
        final_pids, indPump, indUnPump = np.intersect1d(pids_pump, correct_pids_pump, return_indices=True)

        if (((100 / Deltap_FEL) / laser_reprate) == FEL_reprate):
            ch_pump   = ch_pump[indPump]
            ch_unpump = ch_unpump[indUnPump]
            pids_pump=pids_pump[indPump]
            pids_unpump=pids_unpump[indUnPump]

        ppdata = namedtuple("PPData", ["pump", "unpump"])
        result_pp[ch] = ppdata(pump=ch_pump, unpump=ch_unpump)


    result_FEL = {}
    for ch in channels_FEL:
        result_FEL[ch] = subset_FEL[ch].data 

    print ("Loaded {} pump and {} unpump shots".format(len(ch_pump), len(ch_unpump)))

    return result_pp, result_FEL, pids_pump, pids_unpump

########################################################################################

def load_data_compact_laser_pump_JF(channels_pump_unpump, channels_FEL, data, roi1=None, roi2=None, roi3=None, roi4=None):
    #with SFDataFiles(datafiles) as data:
    channels_pump_unpump = check_channels(data, channels_pump_unpump, "pump unpump")
    channels_FEL = check_channels(data, channels_FEL, "FEL")

    subset_FEL = data[channels_FEL]
    subset_FEL.print_stats(show_complete=True)
    
    Event_code = subset_FEL[channel_Events].data
    FEL = Event_code[:,13] #Event 13: changed from 12 on June 22
    
    Deltap_FEL = (1 / FEL.mean()).round().astype(int) #Get the FEL rep rate from the Event code
    FEL_reprate = 100 / Deltap_FEL
    print ('Probe rep rate (FEL) is {} Hz'.format(FEL_reprate))
    
    subset_FEL.drop_missing()

    Event_code = subset_FEL[channel_Events].data

    FEL      = Event_code[:,13] #Event 13: changed from 12 on June 22
    Laser    = Event_code[:,18]
    Darkshot = Event_code[:,21]

    if Darkshot.mean()==0:
        laser_reprate = (1 / Laser.mean() - 1).round().astype(int)
        index_light = np.logical_and.reduce((FEL, Laser))
        index_dark  = np.logical_and.reduce((FEL, np.logical_not(Laser)))
    else:
        laser_reprate = (Laser.mean() / Darkshot.mean() - 1).round().astype(int)

        index_light = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))
        index_dark = np.logical_and.reduce((FEL, Laser, Darkshot))

    #index_probe = np.logical_and.reduce((Laser, np.logical_not(Darkshot)))

    #Deltap_FEL = (1 / FEL.mean()).round().astype(int) #Get the FEL rep rate from the Event code
    #FEL_reprate = 100 / Deltap_FEL
    #print ('Probe rep rate (FEL) is {} Hz'.format(FEL_reprate))

    print ('Pump scheme is {}:1'.format(laser_reprate))

    result_pp = {}
    
    for chname in channels_pump_unpump:
        ch = subset_FEL[chname]
        
        pids_pump = ch.pids[index_light]
        pids_unpump = ch.pids[index_dark]
        correct_pids_pump = pids_unpump + Deltap_FEL
        final_pids, indPump, indUnPump = np.intersect1d(pids_pump, correct_pids_pump, return_indices=True)

        pids_pump=pids_pump[indPump]
        pids_unpump=pids_unpump[indUnPump]
        
        ppdata = namedtuple("PPData", ["pump", "unpump"])  
        
        if "JF" in chname: # or something more clever here!
            fel_data_imgs1, fel_data_imgs2, fel_data_imgs3, fel_data_imgs4 = read_and_crop_jf(ch, roi1, roi2, roi3, roi4)
            
            fel_data_imgs1_pump   = fel_data_imgs1[index_light][indPump]
            fel_data_imgs1_unpump = fel_data_imgs1[index_dark][indUnPump]
            fel_data_imgs2_pump   = fel_data_imgs2[index_light][indPump]
            fel_data_imgs2_unpump = fel_data_imgs2[index_dark][indUnPump]
            fel_data_imgs3_pump   = fel_data_imgs3[index_light][indPump]
            fel_data_imgs3_unpump = fel_data_imgs3[index_dark][indUnPump]
            fel_data_imgs4_pump   = fel_data_imgs4[index_light][indPump]
            fel_data_imgs4_unpump = fel_data_imgs4[index_dark][indUnPump]
            result_pp["JFroi1"] = ppdata(pump=fel_data_imgs1_pump, unpump=fel_data_imgs1_unpump)
            result_pp["JFroi2"] = ppdata(pump=fel_data_imgs2_pump, unpump=fel_data_imgs2_unpump)
            result_pp["JFroi3"] = ppdata(pump=fel_data_imgs3_pump, unpump=fel_data_imgs3_unpump)
            result_pp["JFroi4"] = ppdata(pump=fel_data_imgs4_pump, unpump=fel_data_imgs4_unpump)
        else:
            fel_data = ch.data
            ch_pump   = fel_data[index_light][indPump]
            ch_unpump = fel_data[index_dark][indUnPump]
            result_pp[chname] = ppdata(pump=ch_pump, unpump=ch_unpump)
            
        #result_pp[ch] = ppdata(pump=ch_pump, unpump=ch_unpump)

    result_FEL = {}
    for chname in channels_FEL:
        ch = subset_FEL[chname]
        if "JF" in chname: # or something more clever here!
            fel_data_imgs1, fel_data_imgs2, fel_data_imgs3, fel_data_imgs4 = read_and_crop_jf(ch, roi1, roi2, roi3, roi4)
            result_FEL["JFroi1"]=fel_data_imgs1
            result_FEL["JFroi2"]=fel_data_imgs2
            result_FEL["JFroi3"]=fel_data_imgs3
            result_FEL["JFroi4"]=fel_data_imgs4
        else:
            result_FEL[chname] = ch.data
 

    print ("Loaded {} pump and {} unpump shots".format(len(ch_pump), len(ch_unpump)))

    return result_pp, result_FEL, pids_pump, pids_unpump

########################################################################################

def load_data_compact_laser_pump_JF_noPair(channels_pump_unpump, channels_FEL, data, roi1=None, roi2=None, roi3=None, roi4=None):
    #with SFDataFiles(datafiles) as data:
    channels_pump_unpump = check_channels(data, channels_pump_unpump, "pump unpump")
    channels_FEL = check_channels(data, channels_FEL, "FEL")

    subset_FEL = data[channels_FEL]
    subset_FEL.print_stats(show_complete=True)
    
    Event_code = subset_FEL[channel_Events].data
    FEL = Event_code[:,13] #Event 13: changed from 12 on June 22
    
    Deltap_FEL = (1 / FEL.mean()).round().astype(int) #Get the FEL rep rate from the Event code
    FEL_reprate = 100 / Deltap_FEL
    print ('Probe rep rate (FEL) is {} Hz'.format(FEL_reprate))
    
    subset_FEL.drop_missing()

    Event_code = subset_FEL[channel_Events].data

    FEL      = Event_code[:,13] #Event 13: changed from 12 on June 22
    Laser    = Event_code[:,18]
    Darkshot = Event_code[:,21]

    if Darkshot.mean()==0:
        laser_reprate = (1 / Laser.mean() - 1).round().astype(int)
        index_light = np.logical_and.reduce((FEL, Laser))
        index_dark  = np.logical_and.reduce((FEL, np.logical_not(Laser)))
    else:
        laser_reprate = (Laser.mean() / Darkshot.mean() - 1).round().astype(int)

        index_light = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))
        index_dark = np.logical_and.reduce((FEL, Laser, Darkshot))

    print ('Pump scheme is {}:1'.format(laser_reprate))

    result_pp = {}
    
    for chname in channels_pump_unpump:
        ch = subset_FEL[chname]
        
        pids_pump = ch.pids[index_light]
        pids_unpump = ch.pids[index_dark]
        
        ppdata = namedtuple("PPData", ["pump", "unpump"])  
        
        if "JF" in chname: # or something more clever here!
            fel_data_imgs1, fel_data_imgs2, fel_data_imgs3, fel_data_imgs4 = read_and_crop_jf(ch, roi1, roi2, roi3, roi4)
            
            fel_data_imgs1_pump   = fel_data_imgs1[index_light]
            fel_data_imgs1_unpump = fel_data_imgs1[index_dark]
            fel_data_imgs2_pump   = fel_data_imgs2[index_light]
            fel_data_imgs2_unpump = fel_data_imgs2[index_dark]
            fel_data_imgs3_pump   = fel_data_imgs3[index_light]
            fel_data_imgs3_unpump = fel_data_imgs3[index_dark]
            fel_data_imgs4_pump   = fel_data_imgs4[index_light]
            fel_data_imgs4_unpump = fel_data_imgs4[index_dark]
            result_pp["JFroi1"] = ppdata(pump=fel_data_imgs1_pump, unpump=fel_data_imgs1_unpump)
            result_pp["JFroi2"] = ppdata(pump=fel_data_imgs2_pump, unpump=fel_data_imgs2_unpump)
            result_pp["JFroi3"] = ppdata(pump=fel_data_imgs3_pump, unpump=fel_data_imgs3_unpump)
            result_pp["JFroi4"] = ppdata(pump=fel_data_imgs4_pump, unpump=fel_data_imgs4_unpump)
        else:
            fel_data = ch.data
            ch_pump   = fel_data[index_light]
            ch_unpump = fel_data[index_dark]
            result_pp[chname] = ppdata(pump=ch_pump, unpump=ch_unpump)
            
        #result_pp[ch] = ppdata(pump=ch_pump, unpump=ch_unpump)

    result_FEL = {}
    for chname in channels_FEL:
        ch = subset_FEL[chname]
        if "JF" in chname: # or something more clever here!
            fel_data_imgs1, fel_data_imgs2, fel_data_imgs3, fel_data_imgs4 = read_and_crop_jf(ch, roi1, roi2, roi3, roi4)
            result_FEL["JFroi1"]=fel_data_imgs1
            result_FEL["JFroi2"]=fel_data_imgs2
            result_FEL["JFroi3"]=fel_data_imgs3
            result_FEL["JFroi4"]=fel_data_imgs4
        else:
            result_FEL[chname] = ch.data
 

    print ("Loaded {} pump and {} unpump shots".format(len(ch_pump), len(ch_unpump)))

    return result_pp, result_FEL, pids_pump, pids_unpump

########################################################################################

def load_data_compact_pump_probe_JF(channels_pump_unpump, channels_FEL, data, roi1=None, roi2=None, roi3=None, roi4=None):
    
    channels_pump_unpump = check_channels(data, channels_pump_unpump, "pump unpump")
    channels_FEL = check_channels(data, channels_FEL, "FEL")

    subset_FEL = data[channels_FEL]
    subset_FEL.print_stats(show_complete=True)
    
    Event_code = subset_FEL[channel_Events].data
    FEL = Event_code[:,13] #Event 13: changed from 12 on June 22
    
    Deltap_FEL = (1 / FEL.mean()).round().astype(int) #Get the FEL rep rate from the Event code
    FEL_reprate = 100 / Deltap_FEL
    print ('Probe rep rate (FEL) is {} Hz'.format(FEL_reprate))
    
    subset_FEL.drop_missing()

    Event_code = subset_FEL[channel_Events].data

    FEL      = Event_code[:,13] #Event 13: changed from 12 on June 22
    Laser    = Event_code[:,18]
    Darkshot = Event_code[:,21]

    if Darkshot.mean()==0:
        laser_reprate = (1 / Laser.mean() - 1).round().astype(int)
        index_light = np.logical_and.reduce((FEL, Laser))
        index_dark  = np.logical_and.reduce((FEL, np.logical_not(Laser)))
    else:
        laser_reprate = (Laser.mean() / Darkshot.mean() - 1).round().astype(int)
        index_light = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))
        index_dark = np.logical_and.reduce((FEL, Laser, Darkshot))

    print ('Laser rep rate is {} Hz (delayed or dark)'.format(100 / laser_reprate))
    print ('Pump scheme is {}:1'.format(laser_reprate - 1))

    result_pp = {}
    
    for chname in channels_pump_unpump:
        ch = subset_FEL[chname]
        
        pids_pump = ch.pids[index_light]
        pids_unpump = ch.pids[index_dark]
        correct_pids_pump = pids_unpump + Deltap_FEL
        final_pids, indPump, indUnPump = np.intersect1d(pids_pump, correct_pids_pump, return_indices=True)

        pids_pump=pids_pump[indPump]
        pids_unpump=pids_unpump[indUnPump]
        
        ppdata = namedtuple("PPData", ["pump", "unpump"])  
        
        if "JF" in chname: # or something more clever here!
            fel_data_imgs1, fel_data_imgs2, fel_data_imgs3, fel_data_imgs4 = read_and_crop_jf(ch, roi1, roi2, roi3, roi4)
            
            fel_data_imgs1_pump   = fel_data_imgs1[index_light][indPump]
            fel_data_imgs1_unpump = fel_data_imgs1[index_dark][indUnPump]
            fel_data_imgs2_pump   = fel_data_imgs2[index_light][indPump]
            fel_data_imgs2_unpump = fel_data_imgs2[index_dark][indUnPump]
            fel_data_imgs3_pump   = fel_data_imgs3[index_light][indPump]
            fel_data_imgs3_unpump = fel_data_imgs3[index_dark][indUnPump]
            fel_data_imgs4_pump   = fel_data_imgs4[index_light][indPump]
            fel_data_imgs4_unpump = fel_data_imgs4[index_dark][indUnPump]
            result_pp["JFroi1"] = ppdata(pump=fel_data_imgs1_pump, unpump=fel_data_imgs1_unpump)
            result_pp["JFroi2"] = ppdata(pump=fel_data_imgs2_pump, unpump=fel_data_imgs2_unpump)
            result_pp["JFroi3"] = ppdata(pump=fel_data_imgs3_pump, unpump=fel_data_imgs3_unpump)
            result_pp["JFroi4"] = ppdata(pump=fel_data_imgs4_pump, unpump=fel_data_imgs4_unpump)
        else:
            fel_data = ch.data
            ch_pump   = fel_data[index_light][indPump]
            ch_unpump = fel_data[index_dark][indUnPump]
            result_pp[chname] = ppdata(pump=ch_pump, unpump=ch_unpump)
            
        #result_pp[ch] = ppdata(pump=ch_pump, unpump=ch_unpump)

    result_FEL = {}
    for chname in channels_FEL:
        ch = subset_FEL[chname]
        if "JF" in chname: # or something more clever here!
            fel_data_imgs1, fel_data_imgs2, fel_data_imgs3, fel_data_imgs4 = read_and_crop_jf(ch, roi1, roi2, roi3, roi4)
            result_FEL["JFroi1"]=fel_data_imgs1
            result_FEL["JFroi2"]=fel_data_imgs2
            result_FEL["JFroi3"]=fel_data_imgs3
            result_FEL["JFroi4"]=fel_data_imgs4
        else:
            result_FEL[chname] = ch.data
 

    print ("Loaded {} pump and {} unpump shots".format(len(ch_pump), len(ch_unpump)))

    return result_pp, result_FEL, pids_pump, pids_unpump

########################################################################################

def load_data_compact_pump_probe(channels_pump_unpump, channels_FEL, data):
   
    channels_pump_unpump = check_channels(data, channels_pump_unpump, "pump unpump")
    channels_FEL = check_channels(data, channels_FEL, "FEL")

    subset_FEL = data[channels_FEL]
    subset_FEL.print_stats(show_complete=True)
    
    Event_code = subset_FEL[channel_Events].data
    FEL = Event_code[:,13] #Event 13: changed from 12 on June 22
    
    Deltap_FEL = (1 / FEL.mean()).round().astype(int) #Get the FEL rep rate from the Event code
    FEL_reprate = 100 / Deltap_FEL
    print ('FEL rep rate is {} Hz'.format(FEL_reprate))
    
    subset_FEL.drop_missing()

    Event_code = subset_FEL[channel_Events].data
    
    FEL_raw  = Event_code[:,13] #Event 13: changed from 12 on June 22
    Ppicker  = Event_code[:,200]
    Laser    = Event_code[:,18]
    Darkshot = Event_code[:,21]

    #FEL = np.logical_and(FEL_raw, np.logical_not(Ppicker))
    FEL = FEL_raw

    if Darkshot.mean()==0:
        laser_reprate = (1 / Laser.mean()).round().astype(int)
        index_light = np.logical_and.reduce((FEL, Laser))
        index_dark  = np.logical_and.reduce((FEL, np.logical_not(Laser)))
    else:
        laser_reprate = (Laser.mean() / Darkshot.mean()).round().astype(int) 
        index_light = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))
        index_dark = np.logical_and.reduce((FEL, Laser, Darkshot))

    print ('Laser rep rate is {} Hz (delayed or dark)'.format(100 / laser_reprate))
    print ('Pump scheme is {}:1'.format(laser_reprate - 1))

    result_pp = {}
    for ch in channels_pump_unpump:
        ch_pump   = subset_FEL[ch].data[index_light]
        pids_pump   = subset_FEL[ch].pids[index_light]

        ch_unpump = subset_FEL[ch].data[index_dark]
        pids_unpump = subset_FEL[ch].pids[index_dark]

        correct_pids_pump   = pids_unpump + Deltap_FEL
        final_pids, indPump, indUnPump = np.intersect1d(pids_pump, correct_pids_pump, return_indices=True)
        
        pids_pump=pids_pump[indPump]
        pids_unpump=pids_unpump[indUnPump]

        if (((100 / Deltap_FEL) / (laser_reprate - 1)) == FEL_reprate):
            ch_pump   = ch_pump[indPump]
            ch_unpump = ch_unpump[indUnPump] 

        ppdata = namedtuple("PPData", ["pump", "unpump"])
        result_pp[ch] = ppdata(pump=ch_pump, unpump=ch_unpump)


    result_FEL = {}
    for ch in channels_FEL:
        result_FEL[ch] = subset_FEL[ch].data 

    print ("Loaded {} pump and {} unpump shots".format(len(ch_pump), len(ch_unpump)))

    return result_pp, result_FEL, pids_pump, pids_unpump

########################################################################################
########################################################################################
########################################################################################

def load_YAG_events2(filename, modulo = 2, nshots=None):
    
    (index_light, index_dark), ratioPump_FEL, ratioProbe_laser = load_reprates_FEL_pump(filename, nshots)
    modulo_int = ratioPump_FEL * ratioProbe_laser
    
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)
        pulse_ids = data[channel_BS_pulse_ids][:nshots]
        nshots = _get_modulo(pulse_ids,modulo_int)

        LaserDiode_pump = data[channel_LaserDiode]['data'][:nshots][index_light].ravel()
        LaserDiode_pump = _average(LaserDiode_pump, modulo_int -1)
        LaserDiode_unpump = data[channel_LaserDiode]['data'][:nshots][index_dark].ravel()
        LaserDiode_unpump = _average(LaserDiode_unpump, modulo_int -1)

        LaserRefDiode_pump = data[channel_Laser_refDiode]['data'][:nshots][index_light].ravel()
        LaserRefDiode_pump = _average(LaserRefDiode_pump, modulo_int -1)
        LaserRefDiode_unpump = data[channel_Laser_refDiode]['data'][:nshots][index_dark].ravel()
        LaserRefDiode_unpump = _average(LaserRefDiode_unpump, modulo_int -1)
        
        pulse_ids = data[channel_BS_pulse_ids][:nshots][index_light].ravel()

        IzeroFEL = data[channel_Izero]['data'][:nshots][index_light].ravel()
        IzeroFEL = _average(IzeroFEL, modulo_int -1)

        #PIPS = data[channel_PIPS_trans][:nshots][index_light]
        PIPS = data[channel_LaserDiode]['data'][:nshots][index_light].ravel()

        Delay = data[channel_delay]['data'][:nshots][index_dark]
        #Delay = BS_file[channel_laser_pitch][:][index_dark]

        #BAM = BS_file[channel_BAM][:][index_light]
        
        print ("Pump/umpump arrays have {} shots each".format(len(LaserDiode_pump), len(LaserDiode_unpump)))
        return _cut_to_shortest_length(LaserDiode_pump, LaserDiode_unpump, LaserRefDiode_pump, LaserRefDiode_unpump, IzeroFEL, PIPS, Delay, pulse_ids)

def load_YAG_events(filename, modulo = 2, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:nshots]
        nshots = _get_modulo(pulse_ids,modulo)

        FEL = data[channel_Events]['data'][:nshots,48]
        Laser = data[channel_Events]['data'][:nshots,18]
        Darkshot = data[channel_Events]['data'][:nshots,21]
        
        index_dark_before = np.append([True], np.logical_not(Darkshot))[:-1]
        index_light = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot), index_dark_before))
        index_dark = np.logical_and.reduce((np.logical_not(FEL), Laser, np.logical_not(Darkshot), index_dark_before))
        
        LaserDiode_pump = data[channel_LaserDiode]['data'][:nshots][index_light].ravel()
        LaserDiode_pump = _average(LaserDiode_pump, modulo -1)
        LaserDiode_unpump = data[channel_LaserDiode]['data'][:nshots][index_dark].ravel()
        LaserDiode_unpump = _average(LaserDiode_unpump, modulo -1)

        LaserRefDiode_pump = data[channel_Laser_refDiode]['data'][:nshots][index_light].ravel()
        LaserRefDiode_pump = _average(LaserRefDiode_pump, modulo -1)
        LaserRefDiode_unpump = data[channel_Laser_refDiode]['data'][:nshots][index_dark].ravel()
        LaserRefDiode_unpump = _average(LaserRefDiode_unpump, modulo -1)
        
        pulse_ids = data[channel_BS_pulse_ids][:nshots][index_light].ravel()

        IzeroFEL = data[channel_Izero]['data'][:nshots][index_light].ravel()
        IzeroFEL = _average(IzeroFEL, modulo -1)

        #PIPS = data[channel_PIPS_trans][:nshots][index_light]
        PIPS = data[channel_LaserDiode]['data'][:nshots][index_light].ravel()

        Delay = data[channel_delay]['data'][:nshots][index_dark]
        #Delay = BS_file[channel_laser_pitch][:][index_dark]

        #BAM = BS_file[channel_BAM][:][index_light]
        
        print ("Pump/umpump arrays have {} shots each".format(len(LaserDiode_pump), len(LaserDiode_unpump)))
        return _cut_to_shortest_length(LaserDiode_pump, LaserDiode_unpump, LaserRefDiode_pump, LaserRefDiode_unpump, IzeroFEL, PIPS, Delay, pulse_ids)


def load_YAG_pulseID(filename, reprateFEL, repratelaser):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        reprate_FEL, reprate_laser = _make_reprates_on_off(pulse_ids, reprateFEL, repratelaser)

        LaserDiode_pump = BS_file[channel_LaserDiode]['data'][:][reprate_FEL]
        LaserDiode_unpump = BS_file[channel_LaserDiode]['data'][:][reprate_laser]
        LaserRefDiode_pump = BS_file[channel_Laser_refDiode]['data'][:][reprate_FEL]
        LaserRefDiode_unpump = BS_file[channel_Laser_refDiode]['data'][:][reprate_laser]
        IzeroFEL = BS_file[channel_Izero]['data'][:][reprate_FEL]
        PIPS = BS_file[channel_PIPS_trans]['data'][:][reprate_FEL]

        Delay = BS_file[channel_delay]['data'][:][reprate_laser]
        #Delay = BS_file[channel_laser_pitch][:][index_unpump]

        #BAM = BS_file[channel_BAM][:][reprate_FEL]

    return LaserDiode_pump, LaserDiode_unpump, LaserRefDiode_pump, LaserRefDiode_unpump, IzeroFEL, PIPS, Delay, pulse_ids



# ##    Next: 2 functions to load pump-probe XAS data (energy-delay) (events/pulseIDs)

def load_PumpProbe_events2(filename, channel_variable, modulo=2, nshots=None):
    
    (index_light, index_dark), ratioPump_laser, ratioProbe_FEL = load_reprates_laser_pump(filename, nshots)
    modulo_int = ratioPump_laser * ratioProbe_FEL
    
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)
        pulse_ids = BS_file[channel_BS_pulse_ids][:nshots]
        nshots = _get_modulo(pulse_ids,modulo_int)
                
        DataFluo_pump = BS_file[channel_PIPS_fluo]['data'][:nshots][index_light].ravel()
        DataFluo_pump = _average(DataFluo_pump, ratioPump_laser - 1)
        DataFluo_unpump = BS_file[channel_PIPS_fluo]['data'][:nshots][index_dark].ravel()
        
        DataTrans_pump = BS_file[channel_PIPS_trans]['data'][:nshots][index_light].ravel()
        DataTrans_pump = _average(DataTrans_pump, ratioPump_laser - 1)
        DataTrans_unpump = BS_file[channel_PIPS_trans]['data'][:nshots][index_dark].ravel()
        
        IzeroFEL_pump = BS_file[channel_Izero]['data'][:nshots][index_light].ravel()
        IzeroFEL_pump = _average(IzeroFEL_pump, ratioPump_laser - 1)
        IzeroFEL_unpump = BS_file[channel_Izero]['data'][:nshots][index_dark].ravel()
        
        Variable = BS_file[channel_variable]['data'][:nshots][index_dark]
        
        print ("Pump/umpump arrays have {} shots each".format(len(DataFluo_pump), len(DataFluo_unpump)))
             
    return _cut_to_shortest_length(DataFluo_pump, DataFluo_unpump, IzeroFEL_pump, IzeroFEL_unpump, Variable, DataTrans_pump, DataTrans_unpump, pulse_ids)

def load_PumpProbe_events(filename, channel_variable, modulo=2, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)
        
        pulse_ids = BS_file[channel_BS_pulse_ids][:nshots]
        nshots = _get_modulo(pulse_ids,modulo)
        
        FEL = BS_file[channel_Events]['data'][:nshots,48]
        Laser = BS_file[channel_Events]['data'][:nshots,18]
        Darkshot = BS_file[channel_Events]['data'][:nshots,21]
        
        index_light = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))
        index_dark = np.logical_and.reduce((FEL, Laser, Darkshot))
 #       print (index_light, index_dark)
                
        DataFluo_pump = BS_file[channel_PIPS_fluo]['data'][:nshots][index_light].ravel()
        DataFluo_pump = _average(DataFluo_pump, modulo - 1)
        DataFluo_unpump = BS_file[channel_PIPS_fluo]['data'][:nshots][index_dark].ravel()
        
        DataTrans_pump = BS_file[channel_PIPS_trans]['data'][:nshots][index_light].ravel()
        DataTrans_pump = _average(DataTrans_pump, modulo - 1)
        DataTrans_unpump = BS_file[channel_PIPS_trans]['data'][:nshots][index_dark].ravel()
        
        IzeroFEL_pump = BS_file[channel_Izero]['data'][:nshots][index_light].ravel()
        IzeroFEL_pump = _average(IzeroFEL_pump, modulo - 1)
        IzeroFEL_unpump = BS_file[channel_Izero]['data'][:nshots][index_dark].ravel()
        
        Variable = BS_file[channel_variable]['data'][:nshots][index_dark]
        
        print ("Pump/umpump arrays have {} shots each".format(len(DataFluo_pump), len(DataFluo_unpump)))
             
    return _cut_to_shortest_length(DataFluo_pump, DataFluo_unpump, IzeroFEL_pump, IzeroFEL_unpump, Variable, DataTrans_pump, DataTrans_unpump, pulse_ids)


def load_PumpProbe_pulseID(filename, channel_variable, reprateFEL, repratelaser):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        reprate_FEL, reprate_laser = _make_reprates_on_off(pulse_ids, reprateFEL, repratelaser)

        DataFluo_pump = BS_file[channel_PIPS_fluo]['data'][:][reprate_laser]
        DataFluo_unpump = BS_file[channel_PIPS_fluo]['data'][:][reprate_FEL]

        DataTrans_pump = BS_file[channel_PIPS_trans]['data'][:][reprate_laser]
        DataTrans_unpump = BS_file[channel_PIPS_trans]['data'][:][reprate_FEL]

        IzeroFEL_pump = BS_file[channel_Izero]['data'][:][reprate_laser]
        IzeroFEL_unpump = BS_file[channel_Izero]['data'][:][reprate_FEL]

        Variable = BS_file[channel_variable]['data'][:][reprate_FEL]

    return DataFluo_pump, DataFluo_unpump, IzeroFEL_pump, IzeroFEL_unpump, Variable, DataTrans_pump, DataTrans_unpump


def load_laserIntensity(filename):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        FEL = BS_file[channel_Events]['data'][:,48]
        Laser = BS_file[channel_Events]['data'][:,18]
        Darkshot = BS_file[channel_Events]['data'][:,21]
        Jungfrau = BS_file[channel_Events]['data'][:,40]

        index_light = np.logical_and(Jungfrau,Laser,np.logical_not(Darkshot))

        DataLaser = BS_file[channel_LaserDiode_DIAG]['data'][:][index_light]

        PulseIDs = pulse_ids[:][index_light]

    return DataLaser, PulseIDs


def load_FEL_scans(filename, channel_variable, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:nshots]

        FEL = data[channel_Events]['data'][:nshots,48]
        index_light = FEL == 1

        DataFEL_t  = data[channel_PIPS_trans]['data'][:nshots][index_light]
        DataFEL_f  = data[channel_PIPS_fluo]['data'][:nshots][index_light]
        Izero      = data[channel_Izero]['data'][:nshots][index_light]
        Laser      = data[channel_LaserDiode]['data'][:nshots][index_light]
        Variable   = data[channel_variable]['data'][:nshots][index_light]

        PulseIDs = pulse_ids[:nshots][index_light]

    return DataFEL_t, DataFEL_f, Izero, Laser, Variable, PulseIDs


def load_FEL_scans_pulseID(filename, channel_variable, reprateFEL, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:nshots]

        reprate_FEL = _make_reprates_on(pulse_ids, reprateFEL)

        DataFEL_t  = data[channel_PIPS_trans]['data'][:nshots][reprate_FEL]
        DataFEL_f  = data[channel_PIPS_fluo]['data'][:nshots][reprate_FEL]
        Izero      = data[channel_Izero]['data'][:nshots][reprate_FEL]
        Laser      = data[channel_LaserDiode]['data'][:nshots][reprate_FEL]
        Variable   = data[channel_variable]['data'][:nshots][reprate_FEL]

        PulseIDs = pulse_ids[:nshots][reprate_FEL]

    return DataFEL_t, DataFEL_f, Izero, Laser, Variable, PulseIDs


def load_FEL_pp_pulseID(filename, channel_variable, reprateFEL, repratelaser, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:nshots]

        reprate_FEL, reprate_laser = _make_reprates_on_off(pulse_ids, reprateFEL, repratelaser)

        IzeroFEL_pump = data[channel_Izero]['data'][:nshots][reprate_laser]
        IzeroFEL_unpump = data[channel_Izero]['data'][:nshots][reprate_FEL]
        Variable = data[channel_variable]['data'][:nshots][reprate_FEL]

        PulseIDs = pulse_ids[:nshots][reprate_FEL]

    return IzeroFEL_pump, IzeroFEL_unpump, Variable, PulseIDs



def load_laser_scans(filename):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:]

        Laser = BS_file[channel_Events]['data'][:,18]
        index_light = Laser == 1

        DataLaser = BS_file[channel_LaserDiode]['data'][:][index_light]
        Position = BS_file[channel_position]['data'][:][index_light]

        PulseIDs = pulse_ids[:][index_light]

    return DataLaser, Position, PulseIDs


def load_single_channel(filename, channel, eventCode):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        condition_array = data[channel_Events][:,eventCode]
        condition = condition_array == 1
        
        DataBS = data[channel]['data'][:][condition]
        PulseIDs = data[channel_BS_pulse_ids][:][condition]

    return DataBS, PulseIDs

def load_single_channel_pulseID(filename, channel, reprate):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:]

        condition = _make_reprates_on(pulse_ids, reprate)

        DataBS = data[channel]['data'][:][condition]
        PulseIDs = data[channel_BS_pulse_ids][:][condition]

    return DataBS, PulseIDs

def load_single_channel_pp_pulseID(filename, channel, reprateFEL, repratelaser):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:]

        reprate_FEL, reprate_laser = _make_reprates_on_off(pulse_ids, reprateFEL, repratelaser)

        DataBS_ON = data[channel]['data'][:][reprate_laser]
        DataBS_OFF = data[channel]['data'][:][reprate_FEL]
        PulseIDs_ON = data[channel_BS_pulse_ids][:][reprate_laser]
        PulseIDs_OFF = data[channel_BS_pulse_ids][:][reprate_FEL]

    return DataBS_ON, DataBS_OFF, PulseIDs_ON, PulseIDs_OFF


def load_PSSS_data_from_scans_pulseID(filename, channel_variable, reprateFEL, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        data = _get_data(BS_file)

        pulse_ids = data[channel_BS_pulse_ids][:nshots]

        reprate_FEL = _make_reprates_on(pulse_ids, reprateFEL)

        PSSS_center  = data[channel_PSSS_center]['data'][:nshots][reprate_FEL]
        PSSS_fwhm    = data[channel_PSSS_fwhm]['data'][:nshots][reprate_FEL]
        PSSS_x       = data[channel_PSSS_x]['data'][:nshots][reprate_FEL]
        PSSS_y       = data[channel_PSSS_y]['data'][:nshots][reprate_FEL]

        PulseIDs = pulse_ids[:nshots][reprate_FEL]

    return PSSS_center, PSSS_fwhm, PSSS_x, PSSS_y, PulseIDs

def load_reprates_FEL_pump(filename, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:nshots]
        FEL = BS_file[channel_Events]['data'][:nshots,48]
        Laser = BS_file[channel_Events]['data'][:nshots,18]
        Darkshot = BS_file[channel_Events]['data'][:nshots,21]
        
        #Laser is the probe:
        ratioProbe_laser = int(np.rint(Laser.sum()/len(Laser)))
        
        #FEL is the pump:
        ratioPump_FEL = int(np.rint(len(FEL)/FEL.sum()))
                
        modulo_int = ratioPump_FEL * ratioProbe_laser
        
        nshots = _get_modulo(pulse_ids, modulo_int)
        
        pulse_ids = BS_file[channel_BS_pulse_ids][:nshots]
        FEL = BS_file[channel_Events]['data'][:nshots,48]
        Laser = BS_file[channel_Events]['data'][:nshots,18]
        Darkshot = BS_file[channel_Events]['data'][:nshots,21]
          
        index_dark_before = np.append([True], np.logical_not(Darkshot))[:-1]
        index_light = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot), index_dark_before))
        index_dark = np.logical_and.reduce((np.logical_not(FEL), Laser, np.logical_not(Darkshot), index_dark_before))
              
        return _cut_to_shortest_length(index_light, index_dark), ratioPump_FEL, ratioProbe_laser


def load_reprates_laser_pump(filename, nshots=None):
    with h5py.File(filename, 'r') as BS_file:
        BS_file = _get_data(BS_file)

        pulse_ids = BS_file[channel_BS_pulse_ids][:nshots]
        FEL = BS_file[channel_Events]['data'][:nshots,48]
        Laser = BS_file[channel_Events]['data'][:nshots,18]
        Darkshot = BS_file[channel_Events]['data'][:nshots,21]
        
        #FEL is the probe:
        ratioProbe_FEL = int(np.rint(len(FEL)/FEL.sum()))
        
        #Laser is the pump:
        if (Darkshot.sum() == 0):
            ratioPump_laser = int(np.rint(Laser.sum()/len(Laser)))
        else:
            ratioPump_laser = int(np.rint(Laser.sum()/Darkshot.sum()))
                    
        modulo_int = ratioPump_laser * ratioProbe_FEL
       
        nshots = _get_modulo(pulse_ids, modulo_int)

        pulse_ids = BS_file[channel_BS_pulse_ids][:nshots]
        FEL = BS_file[channel_Events]['data'][:nshots,48]
        Laser = BS_file[channel_Events]['data'][:nshots,18]
        Darkshot = BS_file[channel_Events]['data'][:nshots,21]
                 
        index_light = np.logical_and.reduce((FEL, Laser, np.logical_not(Darkshot)))
        index_dark = np.logical_and.reduce((FEL, Laser, Darkshot))
        
        #print (len(FEL), len(index_light), len(index_dark))
        
        return _cut_to_shortest_length(index_light, index_dark), ratioPump_laser, ratioProbe_FEL

