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
import itertools

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

    for i, step in enumerate(scan.files[:nsteps]):

        JF_file = [f for f in step if "JF02" in f][0]

        print("File {} out of {}:".format(i+1, int(len(scan.files) if nsteps is None else nsteps)))

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


 
    


 
