import numpy as np
import json, glob
import os, math
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from IPython.display import clear_output, display
from datetime import datetime
from scipy.stats.stats import pearsonr
from scipy.signal import find_peaks
from scipy import ndimage
import itertools
from collections import defaultdict

from alvra_tools import clock
from alvra_tools.load_data import *
from alvra_tools.channels import *
from alvra_tools.utils import *

######################################

def plot_tool_static(spectrum, x_axis, bin_):
    
    spectrum_rebin  = bin_sum(spectrum,  bin_)
    x_axis_rebin = bin_mean(x_axis, bin_)
    
    spectrum_err  = np.sqrt(abs(spectrum_rebin))

    low_err = spectrum_rebin - spectrum_err
    high_err = spectrum_rebin + spectrum_err
    return x_axis_rebin, spectrum_rebin, low_err, high_err

######################################

def plot_tool(spectra_ON, spectra_OFF, x_axis, bin_):
    
    spectra_on_rebin  = bin_sum(spectra_ON,  bin_)
    spectra_off_rebin = bin_sum(spectra_OFF, bin_)
    x_axis_rebin = bin_mean(x_axis, bin_)
    
    spectra_on_err  = np.sqrt(abs(spectra_on_rebin))
    spectra_off_err = np.sqrt(abs(spectra_off_rebin))

    low_err = (spectra_on_rebin - spectra_off_rebin)-np.sqrt(spectra_on_err**2+spectra_off_err**2)
    high_err = (spectra_on_rebin - spectra_off_rebin)+np.sqrt(spectra_on_err**2+spectra_off_err**2)
    return x_axis_rebin, spectra_on_rebin, spectra_off_rebin, low_err, high_err

######################################

def energy_calib(image2D, roi, energyaxis, lowlim, highlim):
    energyrange = np.arange(np.shape(image2D)[0])
    energyaxis = energyaxis[lowlim:highlim]
    maxpos = []
    for index in energyrange:
        maxpos.append(np.argmax(image2D[index,:]) + roi[0])
    maxpos = np.asarray(maxpos[lowlim:highlim])
    
    m,b = np.polyfit(maxpos, energyaxis, 1)

    print ('m = {} eV/pixel'.format(m))
    print ('b = {} eV'.format (b))
    
    return (m,b, maxpos)

######################################

def energy_loss(RIXS_2D, xaxis, energy):
    indexinter = np.arange(np.shape(RIXS_2D)[0])
    interp_axis = np.arange((xaxis - energy[-1])[-1], (xaxis - energy[0])[0], xaxis[0]-xaxis[1])

    RIXS_interp = []

    for index in indexinter:
        RIXS_interp.append(np.interp(interp_axis, -(xaxis - energy[index]), np.array(RIXS_2D)[index,:]))
        
    return (interp_axis, RIXS_interp)

######################################

def line_rectifier(image2D_roi, binsize, pixelstart):
    if (image2D_roi.shape[0] % binsize != 0):
        print ('Rebin {} rows to {}'.format((image2D_roi.shape[0]) , image2D_roi.shape[0]//binsize))
        lastindex = (image2D_roi.shape[0] // binsize) * binsize
        image2D_roi = image2D_roi[:lastindex,:]
    image2D_roi_reshaped = rebin2D(image2D_roi, (image2D_roi.shape[0]//binsize, image2D_roi.shape[1]))
    maxposX = np.argmax(image2D_roi_reshaped, axis=1)
    maxposX = np.int16(np.interp(np.arange(np.shape(image2D_roi)[0]), np.arange(len(maxposX))*binsize, maxposX))
    image2D_roi_corrected = []
    for index in range(len(maxposX)):
        temp = np.roll(image2D_roi[index,:], pixelstart - maxposX[index])
        image2D_roi_corrected.append(temp)
    image2D_roi_corrected = np.asarray(image2D_roi_corrected)
    return image2D_roi_corrected

######################################

def get_angle_rotation(ROIkey, roi2D, fitinf, fitsup, increment, liminf, limsup):
    from scipy import ndimage
    from lmfit.models import PseudoVoigtModel, VoigtModel, LorentzianModel
    mod = PseudoVoigtModel()
    angle = np.arange(fitinf, fitsup+increment, increment)
    width = []
    for ang in angle:
        roi_rot = ndimage.rotate(roi2D, ang, axes=(0,1), reshape=False)
        line = np.average(roi_rot, axis = 0)
        line2fit = line[liminf:limsup] - np.average(line[0:20])
        axis2fit = np.arange(0, len(line2fit))
        center = np.argmax(line2fit)
        
        pars = mod.guess(line2fit, x=axis2fit)
        init = mod.eval(pars, x=axis2fit)
        out = mod.fit(line2fit, center=center, x=axis2fit)

        width.append(out.params.get('fwhm').value)
    width = np.asarray(width)
    fit_par = np.poly1d(np.polyfit(angle, width, 6))
    fitangle = np.arange(angle[0], angle[-1], 0.01)
    f = fit_par(fitangle)
    
    return angle[np.argmin(width)], fitangle[np.argmin(f)]

######################################

def clean_ROI_names (channels_ROI):
    for ROIname in channels_ROI:
        if "bkg" in ROIname:
            channels_ROI.remove(ROIname)
    return channels_ROI

######################################

def edge_removal(module_edge, roi_removal, array):
    index_edge = module_edge - roi_removal[0]
    array_input = array.copy()
    print (array[index_edge-1:index_edge+3])
    array[index_edge] = array[index_edge-1]/2
    array[index_edge-1] = array[index_edge-1]/2

    array[index_edge+1] = array[index_edge+2]/2
    array[index_edge+2] = array[index_edge+2]/2 
    print (array[index_edge-1:index_edge+3])
    return array, array_input

######################################

def XES_static_ROIs(scan, channels_list, thr_low, thr_high, index=0, angle_rot=defaultdict(int), del_bkg=True):
    s = scan[index]
    angle_rot=defaultdict(int, angle_rot)
    detector = "JF02T09V03"
#    channels_ROI = add_ROI_channels(s, detector)

    channels_ROI = Get_ROI_names(s, detector)
    if del_bkg:
        channels_ROI = clean_ROI_names(channels_ROI)
    channels_list = channels_list + channels_ROI
    
    check_files_and_data(s)
    check = get_filesize_diff(s)
    if check:
        clear_output(wait=True)
        filename = scan.files[index][0].split('/')[-1].split('.')[0]
        print ('Processing: {}'.format(scan.fname.split('/')[-3]))
        print ('Step {} of {}: filename {}'.format(index+1, len(scan.files), filename))
	     
        results, _ = load_data_compact(channels_list, s)
	    
        thresholded = {}
        averaged = {}
        spectrum = {}
        tags = []
        
        for roi in channels_ROI:
            data = results[roi]
            thr  = threshold(data, thr_low, thr_high)
            if angle_rot[roi] != 0:
                thr = ndimage.rotate(thr, angle_rot[roi], axes=(1,2), reshape=False)
            avg  = np.average(thr, axis = 0)
            #if angle_rot[roi] != 0:
            #    avg = ndimage.rotate(avg, angle_rot[roi], axes=(0,1), reshape=False)
            spec = avg.sum(axis=0)
            
            tag = roi#.split(':')[-1]
            
            thresholded[tag] = thr
            averaged[tag] = avg
            spectrum[tag] = spec
            tags.append(tag)
	    
    meta = results["meta"]

    return(spectrum, averaged, thresholded, tags, meta)

######################################

def XES_PumpProbe_ROIs(scan, channels_list, thr_low, thr_high, index=0, angle_rot=defaultdict(int), del_bkg=True):
    clock_int = clock.Clock()
    angle_rot=defaultdict(int, angle_rot)
    s = scan[index]
    channels_ROI = Get_ROI_names(s, "JF02T09V03")
    if del_bkg:
        channels_ROI = clean_ROI_names(channels_ROI)
    channels_pp = [channel_Events] + channels_list + channels_ROI
    channels_all = channels_pp
    step = scan[index]

    check_files_and_data(step)
    check = get_filesize_diff(step)  
    if check:
        clear_output(wait=True)
        filename = scan.files[index][0].split('/')[-1].split('.')[0]        
        print ('Processing: {}'.format(scan.fname.split('/')[-3]))
        print ('Step {} of {}: filename {}'.format(index+1, len(scan.files), filename))

        resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)

        thresholded_on = {}
        averaged_on = {}
        spectrum_on = {}
		
        thresholded_off = {}
        averaged_off = {}
        spectrum_off = {}

        tags = []
		
        for roi in channels_ROI:
            data_on = resultsPP[roi].pump
            data_off = resultsPP[roi].unpump

            thr_on  = threshold(data_on, thr_low, thr_high)
            if angle_rot[roi] != 0:
                thr_on = ndimage.rotate(thr_on, angle_rot[roi], axes=(1,2), reshape=False)
            avg_on  = np.average(thr_on, axis = 0)
            #if angle_rot[roi] != 0:
            #    avg_on = ndimage.rotate(avg_on, angle_rot[roi], axes=(0,1), reshape=False)
            spec_on = avg_on.sum(axis=0)

            thr_off  = threshold(data_off, thr_low, thr_high)
            if angle_rot[roi] != 0:
                thr_off = ndimage.rotate(thr_off, angle_rot[roi], axes=(1,2), reshape=False)
            avg_off  = np.average(thr_off, axis = 0)
            #if angle_rot[roi] != 0:
            #    avg_off = ndimage.rotate(avg_off, angle_rot[roi], axes=(0,1), reshape=False)
            spec_off = avg_off.sum(axis=0)
		    
            tag = roi#.split(':')[-1]
    
            thresholded_on[tag] = thr_on
            averaged_on[tag] = avg_on
            spectrum_on[tag] = spec_on
		    
            thresholded_off[tag] = thr_off
            averaged_off[tag] = avg_off
            spectrum_off[tag] = spec_off
		
            tags.append(tag)
    print ("Took {} seconds for the previous step".format(clock_int.tick()))
    meta = resultsPP["meta"]
    return(spectrum_on, spectrum_off, averaged_on, averaged_off, thresholded_on, thresholded_off, tags, meta)

######################################

def XES_delayscan_ROIs(scan, channels_list, thr_low, thr_high, angle_rot=defaultdict(int), del_bkg=True):
    angle_rot=defaultdict(int, angle_rot)
    clock_int = clock.Clock()
    s = scan[0]
    channels_ROI = Get_ROI_names(s, "JF02T09V03")
    if del_bkg:
        channels_ROI = clean_ROI_names(channels_ROI)
    channels_pp = [channel_Events] + channels_list + channels_ROI
    channels_all = channels_pp

    if ' as delay' in scan.parameters['name'][0]:
        print ('Scan is done with the stage in fs')
        Delay_fs = scan.readbacks
        Delay_mm = fs2mm(scan.readbacks,0)
    else:
        print ('Scan is done with the stage in mm')
        Delay_fs = mm2fs(scan.readbacks,0)
        Delay_mm = scan.readbacks
	    
    spectra_on = []
    spectra_off = []
    spectra_shots_on = []
    spectra_shots_off = []
    thresholdeds_on = []
    thresholdeds_off = []

    for i, step in enumerate(scan):
	    
        check_files_and_data(step)
        check = get_filesize_diff(step)  
        if check:
            clear_output(wait=True)
            filename = scan.files[i][0].split('/')[-1].split('.')[0]
            print ("Took {} seconds for the previous step".format(clock_int.tick()))
            print ('Processing: {}'.format(scan.fname.split('/')[-3]))
            print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))

            resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)
		
            thresholded_on = {}
            averaged_on = {}
            spectrum_on = {}
            spectrum_shots_on = {}
		
            thresholded_off = {}
            averaged_off = {}
            spectrum_off = {}
            spectrum_shots_off = {}

            tags = []
		
            for roi in channels_ROI:
                data_on = resultsPP[roi].pump
                data_off = resultsPP[roi].unpump
		    
                thr_on  = threshold(data_on, thr_low, thr_high)
                if angle_rot[roi] != 0:
                    thr_on = ndimage.rotate(thr_on, angle_rot[roi], axes=(1,2), reshape=False)
                avg_on  = np.average(thr_on, axis = 0)
                #if angle_rot[roi] != 0:
                #    avg_on = ndimage.rotate(avg_on, angle_rot[roi], axes=(0,1), reshape=False)
                spec_shots_on = thr_on.sum(axis=1)
                spec_on = avg_on.sum(axis=0)
		    
                thr_off  = threshold(data_off, thr_low, thr_high)
                if angle_rot[roi] != 0:
                    thr_off = ndimage.rotate(thr_off, angle_rot[roi], axes=(1,2), reshape=False)
                avg_off  = np.average(thr_off, axis = 0)
                #if angle_rot[roi] != 0:
                #    avg_off = ndimage.rotate(avg_off, angle_rot[roi], axes=(0,1), reshape=False)
                spec_shots_off = thr_off.sum(axis=1)
                spec_off = avg_off.sum(axis=0)
		    
                tag = roi#.split(':')[-1]
    
                thresholded_on[tag] = thr_on
                averaged_on[tag] = avg_on
                spectrum_on[tag] = spec_on
                spectrum_shots_on[tag] = spec_shots_on
		    
                thresholded_off[tag] = thr_off
                averaged_off[tag] = avg_off
                spectrum_off[tag] = spec_off
                spectrum_shots_off[tag] = spec_shots_off
		
                tags.append(tag)

            spectra_on.append(spectrum_on)
            spectra_off.append(spectrum_off)
            spectra_shots_on.append(spectrum_shots_on)
            spectra_shots_off.append(spectrum_shots_off)
            thresholdeds_on.append(thresholded_on)
            thresholdeds_off.append(thresholded_off)

        if i==0:
            meta = resultsPP["meta"]
    
    return(spectra_on, spectra_off, spectra_shots_on, spectra_shots_off, thresholdeds_on, thresholdeds_off, tags, Delay_fs, Delay_mm, meta)
    
######################################

TT_PSEN126 = [channel_PSEN126_signal, channel_PSEN126_bkg, channel_PSEN126_arrTimes, channel_PSEN126_arrTimesAmp, channel_PSEN126_peaks, channel_PSEN126_edges]

def XES_delayscan_TT_ROIs(scan, channels_list, TT, channel_delay_motor, timezero_mm, thr_low, thr_high, angle_rot=defaultdict(int), del_bkg=True):
    angle_rot=defaultdict(int, angle_rot)
    clock_int = clock.Clock()
    s = scan[0]
    channels_ROI = Get_ROI_names(s, "JF02T09V03")
    if del_bkg:
        channels_ROI = clean_ROI_names(channels_ROI)
    channels_pp = [channel_Events, channel_delay_motor] + channels_list + channels_ROI + TT
    channels_all = channels_pp

    if ' as delay' in scan.parameters['name'][0]:
        print ('Scan is done with the stage in fs')
        Delay_fs = scan.readbacks
        Delay_mm = fs2mm(scan.readbacks,0)
    else:
        print ('Scan is done with the stage in mm')
        Delay_fs = mm2fs(scan.readbacks,0)
        Delay_mm = scan.readbacks

    Delay_fs_stage = []
    arrTimes_scan = []
    arrTimesAmp_scan = []
    Delays_fs_scan = []
	    
    spectra_shots_on = []
    spectra_shots_off = []
    thresholdeds_on = []
    thresholdeds_off = []

    for i, step in enumerate(scan):
	    
        check_files_and_data(step)
        check = get_filesize_diff(step)  
        if check:
            clear_output(wait=True)
            filename = scan.files[i][0].split('/')[-1].split('.')[0]
            print ("Took {} seconds for the previous step".format(clock_int.tick()))
            print ('Processing: {}'.format(scan.fname.split('/')[-3]))
            print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))

            resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)

            delay_shot = resultsPP[channel_delay_motor].pump
            delay_shot_fs = mm2fs(delay_shot, timezero_mm)
            Delay_fs_stage.append(delay_shot_fs.mean())

            arrTimes = resultsPP[channel_PSEN126_arrTimes].pump
            arrTimesAmp = resultsPP[channel_PSEN126_arrTimesAmp].pump
            sigtraces = resultsPP[channel_PSEN126_edges].pump
            peaktraces = resultsPP[channel_PSEN126_peaks].pump

            spectrum_shots_on = {}
            spectrum_shots_off = {}
            thresholded_on = {}
            thresholded_off = {}

            tags = []
		
            for roi in channels_ROI:
                data_on = resultsPP[roi].pump
                data_off = resultsPP[roi].unpump
		    
                thr_on  = threshold(data_on, thr_low, thr_high)
                if angle_rot[roi] != 0:
                    thr_on = ndimage.rotate(thr_on, angle_rot[roi], axes=(1,2), reshape=False)
                spec_shots_on = thr_on.sum(axis=1)
                   
                thr_off  = threshold(data_off, thr_low, thr_high)
                if angle_rot[roi] != 0:
                    thr_off = ndimage.rotate(thr_off, angle_rot[roi], axes=(1,2), reshape=False)
                spec_shots_off = thr_off.sum(axis=1)
                    
                tag = roi#.split(':')[-1]
    
                spectrum_shots_on[tag] = spec_shots_on
                spectrum_shots_off[tag] = spec_shots_off
                thresholded_on[tag] = thr_on
                thresholded_off[tag] = thr_off
		
                tags.append(tag)

            spectra_shots_on.append(spectrum_shots_on)
            spectra_shots_off.append(spectrum_shots_off)
            thresholdeds_on.append(thresholded_on)
            thresholdeds_off.append(thresholded_off)
        
        Delays_fs_scan.extend(delay_shot_fs)
        arrTimes_scan.extend(arrTimes)

        if i==0:
            meta = resultsPP["meta"]

    Delays_corr_scan = np.asarray(Delays_fs_scan) + np.asarray(arrTimes_scan)
    
    return Delays_fs_scan, Delays_corr_scan, spectra_shots_on, spectra_shots_off, thresholdeds_on, thresholdeds_off, tags, Delay_fs, Delay_mm, meta

######################################

def RIXS_PumpProbe_ROIs(scan, channels_list, thr_low, thr_high, angle_rot=defaultdict(int)):
    angle_rot=defaultdict(int, angle_rot)

    s = scan[0]
    channels_ROI = Get_ROI_names(s, "JF02T09V03")
    channels_pp = [channel_Events] + channels_list + channels_ROI
    channels_all = channels_pp

    Energy_eV = scan.readbacks
	    
    spectra_on = []
    spectra_off = []
    spectra_shots_on = []
    spectra_shots_off = []
    thresholdeds_on = []
    thresholdeds_off = []

    for i, step in enumerate(scan):
	    
        check_files_and_data(step)
        check = get_filesize_diff(step)  
        if check:
            clear_output(wait=True)
            filename = scan.files[i][0].split('/')[-1].split('.')[0]
            print ('Processing: {}'.format(scan.fname.split('/')[-3]))
            print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))

            resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)
		
            thresholded_on = {}
            averaged_on = {}
            spectrum_on = {}
            spectrum_shots_on = {}
		
            thresholded_off = {}
            averaged_off = {}
            spectrum_off = {}
            spectrum_shots_off = {}

            tags = []
		
            for roi in channels_ROI:
                data_on = resultsPP[roi].pump
                data_off = resultsPP[roi].unpump
		    
                thr_on  = threshold(data_on, thr_low, thr_high)
                if angle_rot[roi] != 0:
                    thr_on = ndimage.rotate(thr_on, angle_rot[roi], axes=(1,2), reshape=False)
                avg_on  = np.average(thr_on, axis = 0)
                spec_shots_on = thr_on.sum(axis=1)
                spec_on = avg_on.sum(axis=0)
		    
                thr_off  = threshold(data_off, thr_low, thr_high)
                if angle_rot[roi] != 0:
                    thr_off = ndimage.rotate(thr_off, angle_rot[roi], axes=(1,2), reshape=False)
                avg_off  = np.average(thr_off, axis = 0)
                spec_shots_off = thr_off.sum(axis=1)
                spec_off = avg_off.sum(axis=0)
		    
                tag = roi#.split(':')[-1]
    
                thresholded_on[tag] = thr_on
                averaged_on[tag] = avg_on
                spectrum_on[tag] = spec_on
                spectrum_shots_on[tag] = spec_shots_on
		    
                thresholded_off[tag] = thr_off
                averaged_off[tag] = avg_off
                spectrum_off[tag] = spec_off
                spectrum_shots_off[tag] = spec_shots_off
		
                tags.append(tag)

            spectra_on.append(spectrum_on)
            spectra_off.append(spectrum_off)
            spectra_shots_on.append(spectrum_shots_on)
            spectra_shots_off.append(spectrum_shots_off)
            thresholdeds_on.append(thresholded_on)
            thresholdeds_off.append(thresholded_off)

        if i==0:
            meta = resultsPP["meta"]
    
    return(spectra_on, spectra_off, spectra_shots_on, spectra_shots_off, thresholdeds_on, thresholdeds_off, tags, Energy_eV, meta)


######################################
#########--Old functions--############
######################################

def XES_static_full(fname, pgroup, thr_low, thr_high, nshots):
    clock_int = clock.Clock()

    images,_ = load_JF_static_batches(fname, pgroup, nshots=nshots)

    print ('Loaded {} images'.format(images.shape[0]))
    print ('Now summing up {} images...'.format(images.shape[0]))

    image_sum = np.sum(images, axis=0)
    image_avg = np.mean(images, axis=0)

    images_thr = threshold(images, thr_low, thr_high)

    image_sum_thr = np.sum(images_thr, axis=0)
    image_avg_thr = np.mean(images_thr, axis=0) 

    print ("It took", clock_int.tick(), "seconds to process this file")
    return image_sum, image_sum_thr, image_avg, image_avg_thr

######################################

def XES_pp_full(fname, pgroup, thr_low, thr_high, nshots):
    clock_int = clock.Clock()

    images_on, images_off,_,_ = load_JF_pp_batches(fname, pgroup, nshots=nshots)

    print ('Loaded {} images ON, {} images OFF'.format(images_on.shape[0],images_off.shape[0]))
    print ('Now summing up {} images ON&OFF...'.format(images_on.shape[0]))

    image_on_sum = np.sum(images_on, axis=0)
    image_on_avg = np.mean(images_on, axis=0)

    image_off_sum = np.sum(images_off, axis=0)
    image_off_avg = np.mean(images_off, axis=0)

    images_on_thr = threshold(images_on, thr_low, thr_high)
    images_off_thr = threshold(images_off, thr_low, thr_high)

    image_on_sum_thr = np.sum(images_on_thr, axis=0)
    image_on_avg_thr = np.mean(images_on_thr, axis=0) 

    image_off_sum_thr = np.sum(images_off_thr, axis=0)
    image_off_avg_thr = np.mean(images_off_thr, axis=0) 

    print ("It took", clock_int.tick(), "seconds to process this file")
    return image_on_sum, image_on_sum_thr, image_on_avg, image_on_avg_thr, image_off_sum, image_off_sum_thr, image_off_avg, image_off_avg_thr

######################################

def XES_static_4ROIs(fname, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize):
    clock_int = clock.Clock()
    
    roiarray = [roi1, roi2, roi3, roi4]

    spec_roi1 = 0
    spec_roi2 = 0
    spec_roi3 = 0
    spec_roi4 = 0

    print("Processing file %s" % (fname.split('/')[-1]))

    imgs_roi1, imgs_roi2, imgs_roi3, imgs_roi4, pids = \
    load_and_crop_JF_static_batches(fname, pgroup, roi1, roi2, roi3, roi4, nshots=nshots)

    imgs_roi1_thr = threshold(imgs_roi1, thr_low, thr_high)
    imgs_roi2_thr = threshold(imgs_roi2, thr_low, thr_high)
    imgs_roi3_thr = threshold(imgs_roi3, thr_low, thr_high)
    imgs_roi4_thr = threshold(imgs_roi4, thr_low, thr_high)

    imgs_roi1_thr_sum = np.average(imgs_roi1_thr, axis = 0)
    imgs_roi2_thr_sum = np.average(imgs_roi2_thr, axis = 0)
    imgs_roi3_thr_sum = np.average(imgs_roi3_thr, axis = 0)
    imgs_roi4_thr_sum = np.average(imgs_roi4_thr, axis = 0)

    image_array = [imgs_roi1_thr_sum, imgs_roi2_thr_sum, imgs_roi3_thr_sum, imgs_roi4_thr_sum]
    correct_array = []

    for index in range(len(image_array)):
        if correctFlag[index]:
            print ('{} will be corrected'.format(roiarray[index]))
            maxvalue = np.max(image_array[index].sum(axis=0))
            refpxl = np.array(find_peaks(image_array[index].sum(axis=0), height=maxvalue/2))[0][0]
            #refpxl = np.argmax(image_array[index].sum(axis=0))
            imgs_correct = line_rectifier(image_array[index], binsize, refpxl)
        else:
            print ('{} will remain as it is'.format(roiarray[index]))
            imgs_correct = image_array[index]
        correct_array.append(imgs_correct)

    spec_roi1  = correct_array[0].sum(axis = 0)
    spec_roi2  = correct_array[1].sum(axis = 0)
    spec_roi3  = correct_array[2].sum(axis = 0)
    spec_roi4  = correct_array[3].sum(axis = 0)

    print ("It took", clock_int.tick(), "seconds to process this file")
    return spec_roi1, spec_roi2, spec_roi3, spec_roi4, pids

######################################

def XES_PumpProbe_4ROIs(fname, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize):
    clock_int = clock.Clock()

    roiarray = [roi1, roi2, roi3, roi4]

    spec_roi1_ON = 0
    spec_roi2_ON = 0
    spec_roi3_ON = 0
    spec_roi4_ON = 0

    spec_roi1_OFF = 0
    spec_roi2_OFF = 0
    spec_roi3_OFF = 0
    spec_roi4_OFF = 0

    print("Processing file %s" % (fname.split('/')[-1]))

    imgs_on_roi1, imgs_on_roi2, imgs_on_roi3, imgs_on_roi4, pids_on, \
    imgs_off_roi1, imgs_off_roi2, imgs_off_roi3, imgs_off_roi4, pids_off = \
    load_and_crop_JF_pp_batches_4rois(fname, pgroup, roi1, roi2, roi3, roi4, nshots=nshots)

    imgs_on_roi1_thr = threshold(imgs_on_roi1, thr_low, thr_high)
    imgs_on_roi2_thr = threshold(imgs_on_roi2, thr_low, thr_high)
    imgs_on_roi3_thr = threshold(imgs_on_roi3, thr_low, thr_high)
    imgs_on_roi4_thr = threshold(imgs_on_roi4, thr_low, thr_high)

    imgs_on_roi1_thr_sum = np.average(imgs_on_roi1_thr, axis = 0)
    imgs_on_roi2_thr_sum = np.average(imgs_on_roi2_thr, axis = 0)
    imgs_on_roi3_thr_sum = np.average(imgs_on_roi3_thr, axis = 0)
    imgs_on_roi4_thr_sum = np.average(imgs_on_roi4_thr, axis = 0)

    imgs_off_roi1_thr = threshold(imgs_off_roi1, thr_low, thr_high)
    imgs_off_roi2_thr = threshold(imgs_off_roi2, thr_low, thr_high)
    imgs_off_roi3_thr = threshold(imgs_off_roi3, thr_low, thr_high)
    imgs_off_roi4_thr = threshold(imgs_off_roi4, thr_low, thr_high)

    imgs_off_roi1_thr_sum = np.average(imgs_off_roi1_thr, axis = 0)
    imgs_off_roi2_thr_sum = np.average(imgs_off_roi2_thr, axis = 0)
    imgs_off_roi3_thr_sum = np.average(imgs_off_roi3_thr, axis = 0)
    imgs_off_roi4_thr_sum = np.average(imgs_off_roi4_thr, axis = 0)

    image_on_array = [imgs_on_roi1_thr_sum, imgs_on_roi2_thr_sum, imgs_on_roi3_thr_sum, imgs_on_roi4_thr_sum]
    image_off_array = [imgs_off_roi1_thr_sum, imgs_off_roi2_thr_sum, imgs_off_roi3_thr_sum, imgs_off_roi4_thr_sum]

    correct_array_on = []
    correct_array_off = []    

    for index in range(len(image_on_array)):
        if correctFlag[index]:
            print ('{} will be corrected'.format(roiarray[index]))
            image_on_off = image_on_array[index] + image_off_array[index]
            maxvalue = np.max(image_on_off.sum(axis=0))
            #maxvalue_off = np.max(image_off_array[index].sum(axis=0))
            refpxl = np.array(find_peaks(image_on_off.sum(axis=0), height=maxvalue/2))[0][0]
            #refpxl_off = np.array(find_peaks(image_off_array[index].sum(axis=0), height=maxvalue_off/2))[0][0]
            
            imgs_correct_on = line_rectifier(image_on_array[index], binsize, refpxl)
            imgs_correct_off = line_rectifier(image_off_array[index], binsize, refpxl)
        else:
            print ('{} will remain as it is'.format(roiarray[index]))
            imgs_correct_on = image_on_array[index]
            imgs_correct_off = image_off_array[index]
        correct_array_on.append(imgs_correct_on)
        correct_array_off.append(imgs_correct_off)

    spec_roi1_ON  = correct_array_on[0].sum(axis = 0)
    spec_roi2_ON  = correct_array_on[1].sum(axis = 0)
    spec_roi3_ON  = correct_array_on[2].sum(axis = 0)
    spec_roi4_ON  = correct_array_on[3].sum(axis = 0)

    spec_roi1_OFF  = correct_array_off[0].sum(axis = 0)
    spec_roi2_OFF  = correct_array_off[1].sum(axis = 0)
    spec_roi3_OFF  = correct_array_off[2].sum(axis = 0)
    spec_roi4_OFF  = correct_array_off[3].sum(axis = 0)
    print ('Loaded {} images ON, {} images OFF'.format(imgs_on_roi1.shape[0],imgs_off_roi1.shape[0]))
    print ("It took", clock_int.tick(), "seconds to process this file")
    return spec_roi1_ON, spec_roi2_ON, spec_roi3_ON, spec_roi4_ON, pids_on, spec_roi1_OFF, spec_roi2_OFF, spec_roi3_OFF, spec_roi4_OFF, pids_off

######################################

def XES_delayscan_4ROIs(scan, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize, nsteps=None):
    clock_int = clock.Clock()

#    from sfdata import SFScanInfo
#    scan = SFScanInfo(json_file)

    if ' as delay' in scan.parameters['name'][0]:
        print ('Scan is done with the stage in fs')
        Delay_fs = scan.readbacks
        Delay_mm = fs2mm(scan.readbacks,0)
    else:
        print ('Scan is done with the stage in mm')
        Delay_fs = mm2fs(scan.readbacks,0)
        Delay_mm = scan.readbacks

    XES_roi1_ON = []
    XES_roi2_ON = []
    XES_roi3_ON = []
    XES_roi4_ON = []

    XES_roi1_OFF = []
    XES_roi2_OFF = []
    XES_roi3_OFF = []
    XES_roi4_OFF = []

    for i, step in enumerate(scan[:nsteps]):

        JF_file = [f for f in step if "JF02" in f][0]
        ff = scan.files[i][0].split('/')[-1].split('.')[0]
        print("File {} out of {}: {}".format(i+1, int(len(scan.files) if nsteps is None else nsteps), ff))

        spec_roi1_ON, spec_roi2_ON, spec_roi3_ON, spec_roi4_ON, pids_on, \
        spec_roi1_OFF, spec_roi2_OFF, spec_roi3_OFF, spec_roi4_OFF, pids_off = \
        XES_PumpProbe_4ROIs(JF_file, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize)

        XES_roi1_ON.append(spec_roi1_ON)
        XES_roi2_ON.append(spec_roi2_ON)
        XES_roi3_ON.append(spec_roi3_ON)
        XES_roi4_ON.append(spec_roi4_ON)

        XES_roi1_OFF.append(spec_roi1_OFF)
        XES_roi2_OFF.append(spec_roi2_OFF)
        XES_roi3_OFF.append(spec_roi3_OFF)
        XES_roi4_OFF.append(spec_roi4_OFF)

        clear_output(wait=True)
        print ("It took", clock_int.tick(), "seconds to process this file")

    XES_roi1_ON = np.asarray(XES_roi1_ON)
    XES_roi2_ON = np.asarray(XES_roi2_ON)
    XES_roi3_ON = np.asarray(XES_roi3_ON)
    XES_roi4_ON = np.asarray(XES_roi4_ON)

    XES_roi1_OFF = np.asarray(XES_roi1_OFF)
    XES_roi2_OFF = np.asarray(XES_roi2_OFF)
    XES_roi3_OFF = np.asarray(XES_roi3_OFF)
    XES_roi4_OFF = np.asarray(XES_roi4_OFF)

    print ("\nJob done! It took", clock_int.tock(), "seconds to process", len(scan.files), "file(s)")
    return XES_roi1_ON, XES_roi2_ON, XES_roi3_ON, XES_roi4_ON, pids_on, XES_roi1_OFF, XES_roi2_OFF, XES_roi3_OFF, XES_roi4_OFF, pids_off, Delay_fs, Delay_mm


######################################

def XES_PumpProbe_4ROIs_sfdata(SFDfile, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize):
    channels_pp = [channel_Events, 'JF02T09V03']
    channels_all = channels_pp
    
    clock_int = clock.Clock()

    roiarray = [roi1, roi2, roi3, roi4]

    spec_roi1_ON = 0
    spec_roi2_ON = 0
    spec_roi3_ON = 0
    spec_roi4_ON = 0

    spec_roi1_OFF = 0
    spec_roi2_OFF = 0
    spec_roi3_OFF = 0
    spec_roi4_OFF = 0

    #print("Processing file {}".format(scan.files[0][0].split('/')[-1].split('.')[0]))

    results_pp, result_FEL, pids_on, pids_off = load_data_compact_laser_pump_JF_noPair(channels_pp, channels_all, SFDfile, roi1, roi2, roi3, roi4)

    imgs_on_roi1 = results_pp["JFroi1"].pump
    imgs_on_roi2 = results_pp["JFroi2"].pump
    imgs_on_roi3 = results_pp["JFroi3"].pump
    imgs_on_roi4 = results_pp["JFroi4"].pump

    imgs_off_roi1 = results_pp["JFroi1"].unpump
    imgs_off_roi2 = results_pp["JFroi2"].unpump
    imgs_off_roi3 = results_pp["JFroi3"].unpump
    imgs_off_roi4 = results_pp["JFroi4"].unpump

    imgs_on_roi1_thr = threshold(imgs_on_roi1, thr_low, thr_high)
    imgs_on_roi2_thr = threshold(imgs_on_roi2, thr_low, thr_high)
    imgs_on_roi3_thr = threshold(imgs_on_roi3, thr_low, thr_high)
    imgs_on_roi4_thr = threshold(imgs_on_roi4, thr_low, thr_high)

    imgs_on_roi1_thr_sum = np.average(imgs_on_roi1_thr, axis = 0)
    imgs_on_roi2_thr_sum = np.average(imgs_on_roi2_thr, axis = 0)
    imgs_on_roi3_thr_sum = np.average(imgs_on_roi3_thr, axis = 0)
    imgs_on_roi4_thr_sum = np.average(imgs_on_roi4_thr, axis = 0)

    imgs_off_roi1_thr = threshold(imgs_off_roi1, thr_low, thr_high)
    imgs_off_roi2_thr = threshold(imgs_off_roi2, thr_low, thr_high)
    imgs_off_roi3_thr = threshold(imgs_off_roi3, thr_low, thr_high)
    imgs_off_roi4_thr = threshold(imgs_off_roi4, thr_low, thr_high)

    imgs_off_roi1_thr_sum = np.average(imgs_off_roi1_thr, axis = 0)
    imgs_off_roi2_thr_sum = np.average(imgs_off_roi2_thr, axis = 0)
    imgs_off_roi3_thr_sum = np.average(imgs_off_roi3_thr, axis = 0)
    imgs_off_roi4_thr_sum = np.average(imgs_off_roi4_thr, axis = 0)

    image_on_array = [imgs_on_roi1_thr_sum, imgs_on_roi2_thr_sum, imgs_on_roi3_thr_sum, imgs_on_roi4_thr_sum]
    image_off_array = [imgs_off_roi1_thr_sum, imgs_off_roi2_thr_sum, imgs_off_roi3_thr_sum, imgs_off_roi4_thr_sum]

    correct_array_on = []
    correct_array_off = []    

    for index in range(len(image_on_array)):
        if correctFlag[index]:
            print ('{} will be corrected'.format(roiarray[index]))
            image_on_off = image_on_array[index] + image_off_array[index]
            maxvalue = np.max(image_on_off.sum(axis=0))
            #maxvalue_off = np.max(image_off_array[index].sum(axis=0))
            refpxl = np.array(find_peaks(image_on_off.sum(axis=0), height=maxvalue/2))[0][0]
            #refpxl_off = np.array(find_peaks(image_off_array[index].sum(axis=0), height=maxvalue_off/2))[0][0]
            
            imgs_correct_on = line_rectifier(image_on_array[index], binsize, refpxl)
            imgs_correct_off = line_rectifier(image_off_array[index], binsize, refpxl)
        else:
            print ('{} will remain as it is'.format(roiarray[index]))
            imgs_correct_on = image_on_array[index]
            imgs_correct_off = image_off_array[index]
        correct_array_on.append(imgs_correct_on)
        correct_array_off.append(imgs_correct_off)

    spec_roi1_ON  = correct_array_on[0].sum(axis = 0)
    spec_roi2_ON  = correct_array_on[1].sum(axis = 0)
    spec_roi3_ON  = correct_array_on[2].sum(axis = 0)
    spec_roi4_ON  = correct_array_on[3].sum(axis = 0)

    spec_roi1_OFF  = correct_array_off[0].sum(axis = 0)
    spec_roi2_OFF  = correct_array_off[1].sum(axis = 0)
    spec_roi3_OFF  = correct_array_off[2].sum(axis = 0)
    spec_roi4_OFF  = correct_array_off[3].sum(axis = 0)
    print ('Loaded {} images ON, {} images OFF'.format(imgs_on_roi1.shape[0],imgs_off_roi1.shape[0]))
    print ("It took", clock_int.tick(), "seconds to process this file")
    return spec_roi1_ON, spec_roi2_ON, spec_roi3_ON, spec_roi4_ON, pids_on, spec_roi1_OFF, spec_roi2_OFF, spec_roi3_OFF, spec_roi4_OFF, pids_off

######################################

def XES_delayscan_4ROIs_sfdata(scan, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize, nsteps=None):
    clock_int = clock.Clock()

#    from sfdata import SFScanInfo
#    scan = SFScanInfo(json_file)

    if ' as delay' in scan.parameters['name'][0]:
        print ('Scan is done with the stage in fs')
        Delay_fs = scan.readbacks
        Delay_mm = fs2mm(scan.readbacks,0)
    else:
        print ('Scan is done with the stage in mm')
        Delay_fs = mm2fs(scan.readbacks,0)
        Delay_mm = scan.readbacks

    XES_roi1_ON = []
    XES_roi2_ON = []
    XES_roi3_ON = []
    XES_roi4_ON = []

    XES_roi1_OFF = []
    XES_roi2_OFF = []
    XES_roi3_OFF = []
    XES_roi4_OFF = []

    for i, step in enumerate(scan[:nsteps]):
#    for i, step in enumerate(scan):

        JF_file = [f for f in step if "JF02" in f][0]
        ff = scan.files[i][0].split('/')[-1].split('.')[0]
        print("File {} out of {}: {}".format(i+1, int(len(scan.files) if nsteps is None else nsteps), ff))

        spec_roi1_ON, spec_roi2_ON, spec_roi3_ON, spec_roi4_ON, pids_on, \
        spec_roi1_OFF, spec_roi2_OFF, spec_roi3_OFF, spec_roi4_OFF, pids_off = \
        XES_PumpProbe_4ROIs_sfdata(step, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize)

        XES_roi1_ON.append(spec_roi1_ON)
        XES_roi2_ON.append(spec_roi2_ON)
        XES_roi3_ON.append(spec_roi3_ON)
        XES_roi4_ON.append(spec_roi4_ON)

        XES_roi1_OFF.append(spec_roi1_OFF)
        XES_roi2_OFF.append(spec_roi2_OFF)
        XES_roi3_OFF.append(spec_roi3_OFF)
        XES_roi4_OFF.append(spec_roi4_OFF)

        clear_output(wait=True)
        print ("It took", clock_int.tick(), "seconds to process this file")

    XES_roi1_ON = np.asarray(XES_roi1_ON)
    XES_roi2_ON = np.asarray(XES_roi2_ON)
    XES_roi3_ON = np.asarray(XES_roi3_ON)
    XES_roi4_ON = np.asarray(XES_roi4_ON)

    XES_roi1_OFF = np.asarray(XES_roi1_OFF)
    XES_roi2_OFF = np.asarray(XES_roi2_OFF)
    XES_roi3_OFF = np.asarray(XES_roi3_OFF)
    XES_roi4_OFF = np.asarray(XES_roi4_OFF)

    print ("\nJob done! It took", clock_int.tock(), "seconds to process", int(len(scan.files) if nsteps is None else nsteps), "file(s)")
    return XES_roi1_ON, XES_roi2_ON, XES_roi3_ON, XES_roi4_ON, pids_on, XES_roi1_OFF, XES_roi2_OFF, XES_roi3_OFF, XES_roi4_OFF, pids_off, Delay_fs, Delay_mm

######################################

TT_PSEN126 = [channel_PSEN126_signal, channel_PSEN126_bkg, channel_PSEN126_arrTimes, channel_PSEN126_arrTimesAmp, channel_PSEN126_peaks, channel_PSEN126_edges]

def XES_delayscan_TT_4ROIs(scan, pgroup, TT, channel_delay_motor, timezero_mm, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, nsteps=None):
    clock_int = clock.Clock()
    channels_pp = [channel_Events, channel_delay_motor, 'JF02T09V03'] + TT
    channels_all = channels_pp

#    from sfdata import SFScanInfo
#    scan = SFScanInfo(json_file)

    if ' as delay' in scan.parameters['name'][0]:
        print ('Scan is done with the stage in fs')
        Delay_fs = scan.readbacks
        Delay_mm = fs2mm(scan.readbacks,0)
    else:
        print ('Scan is done with the stage in mm')
        Delay_fs = mm2fs(scan.readbacks,0)
        Delay_mm = scan.readbacks

    Delay_fs_stage = []
    arrTimes_scan = []
    arrTimesAmp_scan = []
    Delays_fs_scan = []

    XES_roi1_ON = []
    XES_roi2_ON = []
    XES_roi3_ON = []
    XES_roi4_ON = []

    XES_roi1_OFF = []
    XES_roi2_OFF = []
    XES_roi3_OFF = []
    XES_roi4_OFF = []

    for i, step in enumerate(scan[:nsteps]):

        JF_file = [f for f in step if "JF02" in f][0]
        ff = scan.files[i][0].split('/')[-1].split('.')[0]
        print("File {} out of {}: {}".format(i+1, int(len(scan.files) if nsteps is None else nsteps), ff))

        resultsPP,_,_,_ = load_data_compact_laser_pump_JF_noPair(channels_pp, channels_all, step, roi1, roi2, roi3, roi4)

        delay_shot = resultsPP[channel_delay_motor].pump
        delay_shot_fs = mm2fs(delay_shot, timezero_mm)
        Delay_fs_stage.append(delay_shot_fs.mean())

        arrTimes = resultsPP[channel_PSEN126_arrTimes].pump
        arrTimesAmp = resultsPP[channel_PSEN126_arrTimesAmp].pump
        sigtraces = resultsPP[channel_PSEN126_edges].pump
        peaktraces = resultsPP[channel_PSEN126_peaks].pump

        XES_shot_roi1_ON_thr = threshold(resultsPP["JFroi1"].pump, thr_low, thr_high)
        XES_shot_roi2_ON_thr = threshold(resultsPP["JFroi2"].pump, thr_low, thr_high)
        XES_shot_roi3_ON_thr = threshold(resultsPP["JFroi3"].pump, thr_low, thr_high)
        XES_shot_roi4_ON_thr = threshold(resultsPP["JFroi4"].pump, thr_low, thr_high)

        XES_shot_roi1_OFF_thr = threshold(resultsPP["JFroi1"].unpump, thr_low, thr_high)
        XES_shot_roi2_OFF_thr = threshold(resultsPP["JFroi2"].unpump, thr_low, thr_high)
        XES_shot_roi3_OFF_thr = threshold(resultsPP["JFroi3"].unpump, thr_low, thr_high)
        XES_shot_roi4_OFF_thr = threshold(resultsPP["JFroi4"].unpump, thr_low, thr_high)
           
        spec_roi1_ON = XES_shot_roi1_ON_thr.sum(axis=1)
        spec_roi2_ON = XES_shot_roi2_ON_thr.sum(axis=1)
        spec_roi3_ON = XES_shot_roi3_ON_thr.sum(axis=1)
        spec_roi4_ON = XES_shot_roi4_ON_thr.sum(axis=1)

        spec_roi1_OFF = XES_shot_roi1_OFF_thr.sum(axis=1)
        spec_roi2_OFF = XES_shot_roi2_OFF_thr.sum(axis=1)
        spec_roi3_OFF = XES_shot_roi3_OFF_thr.sum(axis=1)
        spec_roi4_OFF = XES_shot_roi4_OFF_thr.sum(axis=1)

        XES_roi1_ON.append(spec_roi1_ON)
        XES_roi2_ON.append(spec_roi2_ON)
        XES_roi3_ON.append(spec_roi3_ON)
        XES_roi4_ON.append(spec_roi4_ON)

        XES_roi1_OFF.append(spec_roi1_OFF)
        XES_roi2_OFF.append(spec_roi2_OFF)
        XES_roi3_OFF.append(spec_roi3_OFF)
        XES_roi4_OFF.append(spec_roi4_OFF)

        Delays_fs_scan.append(delay_shot_fs)
        arrTimes_scan.append(arrTimes)

        clear_output(wait=True)
        print ("It took", clock_int.tick(), "seconds to process this file")

    #Delay_mm = Delay_mm[:np.shape(Pump_probe)[0]]
    #Delay_fs = Delay_fs[:np.shape(Pump_probe)[0]]

    Delays_fs_scan = np.asarray(list(itertools.chain.from_iterable(Delays_fs_scan)))
    arrTimes_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_scan)))

    Delays_corr_scan = Delays_fs_scan + arrTimes_scan

    XES_roi1_ON = np.asarray(XES_roi1_ON)
    XES_roi2_ON = np.asarray(XES_roi2_ON)
    XES_roi3_ON = np.asarray(XES_roi3_ON)
    XES_roi4_ON = np.asarray(XES_roi4_ON)
    
    XES_roi1_OFF = np.asarray(XES_roi1_OFF)
    XES_roi2_OFF = np.asarray(XES_roi2_OFF)
    XES_roi3_OFF = np.asarray(XES_roi3_OFF)
    XES_roi4_OFF = np.asarray(XES_roi4_OFF)

    #XES_roi1_ON = np.asarray(list(itertools.chain.from_iterable(XES_roi1_ON)))
    #XES_roi2_ON = np.asarray(list(itertools.chain.from_iterable(XES_roi2_ON)))
    #XES_roi3_ON = np.asarray(list(itertools.chain.from_iterable(XES_roi3_ON)))
    #XES_roi4_ON = np.asarray(list(itertools.chain.from_iterable(XES_roi4_ON)))
    
    #XES_roi1_OFF = np.asarray(list(itertools.chain.from_iterable(XES_roi1_OFF)))
    #XES_roi2_OFF = np.asarray(list(itertools.chain.from_iterable(XES_roi2_OFF)))
    #XES_roi3_OFF = np.asarray(list(itertools.chain.from_iterable(XES_roi3_OFF)))
    #XES_roi4_OFF = np.asarray(list(itertools.chain.from_iterable(XES_roi4_OFF)))

    print ("\nJob done! It took", clock_int.tock(), "seconds to process", int(len(scan.files) if nsteps is None else nsteps), "file(s)")
    return Delays_fs_scan, Delays_corr_scan, XES_roi1_ON, XES_roi2_ON, XES_roi3_ON, XES_roi4_ON, XES_roi1_OFF, XES_roi2_OFF, XES_roi3_OFF, XES_roi4_OFF, Delay_fs, Delay_mm

######################################

def XES_delayscan_TT_reduced(scan, pgroup, TT, channel_delay_motor, timezero_mm, thr_low, thr_high, nshots, nsteps=None):
    from collections import defaultdict

    clock_int = clock.Clock()
    channels_pp = [channel_Events, channel_delay_motor, 'JF02T09V03'] + TT
    channels_all = channels_pp

    if ' as delay' in scan.parameters['name'][0]:
        print ('Scan is done with the stage in fs')
        Delay_fs = scan.readbacks
        Delay_mm = fs2mm(scan.readbacks,0)
    else:
        print ('Scan is done with the stage in mm')
        Delay_fs = mm2fs(scan.readbacks,0)
        Delay_mm = scan.readbacks

    Delay_fs_stage = []
    arrTimes_scan = []
    arrTimesAmp_scan = []
    Delays_fs_scan = []

    XES_spectra_on = defaultdict(list)
    XES_spectra_off = defaultdict(list)

    for i, step in enumerate(scan[:nsteps]):

        ff = scan.files[i][0].split('/')[-1].split('.')[0]
        print("File {} out of {}: {}".format(i+1, int(len(scan.files) if nsteps is None else nsteps), ff))

        subset = step[channels_all]
        subset.print_stats(show_complete=True)
        subset.drop_missing()
        valid_idx = subset['JF02T09V03'].valid

        Event_code = subset[channel_Events].data

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

        delay_shot = step[channel_delay_motor][index_light]
        delay_shot_fs = mm2fs(delay_shot, timezero_mm)
        Delay_fs_stage.append(delay_shot_fs.mean())

        arrTimes = step[channel_PSEN126_arrTimes][index_light]
        arrTimesAmp = step[channel_PSEN126_arrTimesAmp][index_light]
        sigtraces = step[channel_PSEN126_edges][index_light]
        peaktraces = step[channel_PSEN126_peaks][index_light]

        spec_roi_on = 8*[0]
        spec_roi_off = 8*[0]
        roi_name = []	
	
        for nroi in range(8):
            imgs = step['JF02T09V03'].juf[f'data_roi_{nroi}'][valid_idx]

            imgs_on = imgs[index_light]
            imgs_on = threshold(imgs_on, thr_low, thr_high)
            spec_roi_on[nroi] = imgs_on.sum(axis=1)
        
            imgs_off = imgs[index_dark]
            imgs_off = threshold(imgs_off, thr_low, thr_high)
            spec_roi_off[nroi] = imgs_off.sum(axis=1)
        
            roi_name.append(f'roi_{nroi}')

        for nroi,roi in enumerate(roi_name):
            XES_spectra_on[roi].append(spec_roi_on[nroi])
            XES_spectra_off[roi].append(spec_roi_off[nroi])

        Delays_fs_scan.append(delay_shot_fs)
        arrTimes_scan.append(arrTimes)

        clear_output(wait=True)
        print ("It took", clock_int.tick(), "seconds to process this file")

    Delays_fs_scan = np.asarray(list(itertools.chain.from_iterable(Delays_fs_scan)))
    arrTimes_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_scan)))
    Delays_corr_scan = Delays_fs_scan + arrTimes_scan

    print ("\nJob done! It took", clock_int.tock(), "seconds to process", int(len(scan.files) if nsteps is None else nsteps), "file(s)")
    return Delays_fs_scan, Delays_corr_scan, XES_spectra_on, XES_spectra_off, Delay_fs, Delay_mm

######################################

def RIXS_static_4ROIs(json_file, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize):
    clock_int = clock.Clock()
    from sfdata import SFScanInfo
    scan = SFScanInfo(json_file)
    Energy_eV = scan.readbacks

    RIXS_roi1 = []
    RIXS_roi2 = []
    RIXS_roi3 = []
    RIXS_roi4 = []

    for i, step in enumerate(scan.files):
        
        JF_file = [f for f in step if "JF02" in f][0]
  
        print("File {} out of {}:".format(i+1, len(scan.files)))
        
        spec_roi1, spec_roi2, spec_roi3, spec_roi4, pids = \
        XES_static_4ROIs(JF_file, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize)

        RIXS_roi1.append(spec_roi1)
        RIXS_roi2.append(spec_roi2)
        RIXS_roi3.append(spec_roi3)
        RIXS_roi4.append(spec_roi4)

        clear_output(wait=True)
        print ("It took", clock_int.tick(), "seconds to process this file")
   
    RIXS_roi1 = np.asarray(RIXS_roi1)
    RIXS_roi2 = np.asarray(RIXS_roi2)
    RIXS_roi3 = np.asarray(RIXS_roi3)
    RIXS_roi4 = np.asarray(RIXS_roi4)

    print ("\nJob done! It took", clock_int.tock(), "seconds to process", len(scan.files), "file(s)")
    return RIXS_roi1, RIXS_roi2, RIXS_roi3, RIXS_roi4, Energy_eV, pids
    
######################################

def XES_scan_4ROIs_sfdata(scan, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize, nsteps=None):
    clock_int = clock.Clock()

#    from sfdata import SFScanInfo
#    scan = SFScanInfo(json_file)

    Adjustable = scan.readbacks

    XES_roi1_ON = []
    XES_roi2_ON = []
    XES_roi3_ON = []
    XES_roi4_ON = []

    XES_roi1_OFF = []
    XES_roi2_OFF = []
    XES_roi3_OFF = []
    XES_roi4_OFF = []

    for i, step in enumerate(scan[:nsteps]):

        ff = scan.files[0][0].split('/')[-1].split('.')[0]
        print("File {} out of {}: {}".format(i+1, int(len(scan.files) if nsteps is None else nsteps), ff))

        spec_roi1_ON, spec_roi2_ON, spec_roi3_ON, spec_roi4_ON, pids_on, \
        spec_roi1_OFF, spec_roi2_OFF, spec_roi3_OFF, spec_roi4_OFF, pids_off = \
        XES_PumpProbe_4ROIs_sfdata(step, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize)

        XES_roi1_ON.append(spec_roi1_ON)
        XES_roi2_ON.append(spec_roi2_ON)
        XES_roi3_ON.append(spec_roi3_ON)
        XES_roi4_ON.append(spec_roi4_ON)

        XES_roi1_OFF.append(spec_roi1_OFF)
        XES_roi2_OFF.append(spec_roi2_OFF)
        XES_roi3_OFF.append(spec_roi3_OFF)
        XES_roi4_OFF.append(spec_roi4_OFF)

        clear_output(wait=True)
        print ("It took", clock_int.tick(), "seconds to process this file")

    XES_roi1_ON = np.asarray(XES_roi1_ON)
    XES_roi2_ON = np.asarray(XES_roi2_ON)
    XES_roi3_ON = np.asarray(XES_roi3_ON)
    XES_roi4_ON = np.asarray(XES_roi4_ON)

    XES_roi1_OFF = np.asarray(XES_roi1_OFF)
    XES_roi2_OFF = np.asarray(XES_roi2_OFF)
    XES_roi3_OFF = np.asarray(XES_roi3_OFF)
    XES_roi4_OFF = np.asarray(XES_roi4_OFF)

    print ("\nJob done! It took", clock_int.tock(), "seconds to process", int(len(scan.files) if nsteps is None else nsteps), "file(s)")
    return XES_roi1_ON, XES_roi2_ON, XES_roi3_ON, XES_roi4_ON, pids_on, XES_roi1_OFF, XES_roi2_OFF, XES_roi3_OFF, XES_roi4_OFF, pids_off, Adjustable

######################################

def RIXS_PumpProbe_4ROIs(json_file, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize):
    clock_int = clock.Clock()
    from sfdata import SFScanInfo
    scan = SFScanInfo(json_file)
    Energy_eV = scan.readbacks

    RIXS_roi1_ON = []
    RIXS_roi2_ON = []
    RIXS_roi3_ON = []
    RIXS_roi4_ON = []

    RIXS_roi1_OFF = []
    RIXS_roi2_OFF = []
    RIXS_roi3_OFF = []
    RIXS_roi4_OFF = []

    for i, step in enumerate(scan.files):
        JF_file = [f for f in step if "JF02" in f][0]
  
        print("File {} out of {}:".format(i+1, len(scan.files)))

        spec_roi1_ON, spec_roi2_ON, spec_roi3_ON, spec_roi4_ON, pids_on, \
        spec_roi1_OFF, spec_roi2_OFF, spec_roi3_OFF, spec_roi4_OFF, pids_off = \
        XES_PumpProbe_4ROIs(JF_file, pgroup, roi1, roi2, roi3, roi4, thr_low, thr_high, nshots, correctFlag, binsize)

        RIXS_roi1_ON.append(spec_roi1_ON)
        RIXS_roi2_ON.append(spec_roi2_ON)
        RIXS_roi3_ON.append(spec_roi3_ON)
        RIXS_roi4_ON.append(spec_roi4_ON)

        RIXS_roi1_OFF.append(spec_roi1_OFF)
        RIXS_roi2_OFF.append(spec_roi2_OFF)
        RIXS_roi3_OFF.append(spec_roi3_OFF)
        RIXS_roi4_OFF.append(spec_roi4_OFF)

        clear_output(wait=True)
        print ("It took", clock_int.tick(), "seconds to process this file")
   
    RIXS_roi1_ON = np.asarray(RIXS_roi1_ON)
    RIXS_roi2_ON = np.asarray(RIXS_roi2_ON)
    RIXS_roi3_ON = np.asarray(RIXS_roi3_ON)
    RIXS_roi4_ON = np.asarray(RIXS_roi4_ON)

    RIXS_roi1_OFF = np.asarray(RIXS_roi1_OFF)
    RIXS_roi2_OFF = np.asarray(RIXS_roi2_OFF)
    RIXS_roi3_OFF = np.asarray(RIXS_roi3_OFF)
    RIXS_roi4_OFF = np.asarray(RIXS_roi4_OFF)

    print ("\nJob done! It took", clock_int.tock(), "seconds to process", len(scan.files), "file(s)")
    return RIXS_roi1_ON, RIXS_roi2_ON, RIXS_roi3_ON, RIXS_roi4_ON, pids_on, RIXS_roi1_OFF, RIXS_roi2_OFF, RIXS_roi3_OFF, RIXS_roi4_OFF, pids_off, Energy_eV

######################################
######################################

def save_data_XES_timescans(reducedir, run_name, delaymm, delayfs, spec_array_ON, spec_array_OFF):

    np.save(reducedir+run_name+'/delays_fs.npy', delayfs)
    np.save(reducedir+run_name+'/delays_mm.npy', delaymm)

    for i, spectrum in enumerate(spec_array_ON):
        np.save(reducedir+run_name+'/spectrum_roi{}_ON.npy'.format(i+1), spectrum)

    for i, spectrum in enumerate(spec_array_OFF):
        np.save(reducedir+run_name+'/spectrum_roi{}_OFF.npy'.format(i+1), spectrum)
    
######################################

def save_data_XES(reducedir, run_name, spec_array_ON, spec_array_OFF):

    for i, spectrum in enumerate(spec_array_ON):
        np.save(reducedir+run_name+'/spectrum_roi{}_ON.npy'.format(i+1), spectrum)

    for i, spectrum in enumerate(spec_array_OFF):
        np.save(reducedir+run_name+'/spectrum_roi{}_OFF.npy'.format(i+1), spectrum)

######################################

def save_data_XES_ROIs(reducedir, run_name, s_on, s_off, rois, meta):
 
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "spectra_on": s_on, 
                                    "spectra_off" : s_off, 
                                    "ROIs" : rois, 
                                    "meta" : meta}
   
    np.save(reducedir+run_name+'/run_array', run_array)

######################################

def save_data_XES_timescans_ROIs(reducedir, run_name, s_on, s_off, rois, delaymm, delayfs, meta):

    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "spectra_on": s_on, 
                                    "spectra_off" : s_off, 
                                    "ROIs" : rois,
                                    "Delay_mm" : delaymm,
                                    "Delay_fs" : delayfs,
                                    "meta" : meta}
   
    np.save(reducedir+run_name+'/run_array', run_array)

######################################

def save_data_XES_timescans_ROIs_TT(reducedir, run_name, all_s_on, all_s_off, thrs_on, thrs_off, rois, all_delays_stage, all_delays_corr, meta, runlist):

    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "all_spectra_shots_on": all_s_on, 
                                    "all_spectra_shots_off" : all_s_off, 
                                    "all_thresholds_on" : thrs_on,
                                    "all_thresholds_off" : thrs_off,
                                    "ROIs" : rois,
                                    "all_delays_fs_scan" : all_delays_stage,
                                    "all_delays_corr_scan" : all_delays_corr,
                                    "meta" : meta,
                                    "runlist" : runlist}
   
    np.save(reducedir+run_name+'/run_array', run_array)
    
######################################
    
def save_data_XES_timescans_ROIs_TT_stack(reducedir, run_name, all_s_on, all_s_off, rois, all_delays_stage, all_delays_corr, meta, runlist):

    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "all_spectra_shots_on": all_s_on, 
                                    "all_spectra_shots_off" : all_s_off, 
                                    "ROIs" : rois,
                                    "all_delays_fs_scan" : all_delays_stage,
                                    "all_delays_corr_scan" : all_delays_corr,
                                    "meta" : meta,
                                    "runlist" : runlist}
   
    np.save(reducedir+run_name+'/run_array', run_array)

######################################

def save_data_RIXS_ROIs(reducedir, run_name, s_on, s_off, rois, energy, meta):

    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "spectra_on": s_on, 
                                    "spectra_off" : s_off, 
                                    "ROIs" : rois,
                                    "Energy_eV" : energy,
                                    "meta" : meta}
   
    np.save(reducedir+run_name+'/run_array', run_array)



 
