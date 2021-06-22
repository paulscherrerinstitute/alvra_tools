import numpy as np
import json, glob
import os, math
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from IPython.display import clear_output, display
from ipyfilechooser import FileChooser
from datetime import datetime
import itertools

from alvra_tools.load_data import *
from alvra_tools.channels import *
from alvra_tools.timing_tool import *
from alvra_tools.utils import *


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


# +
def TT_statistics_scan(json_file, target, calibration):
    channel_list_pp = [channel_Events, channel_PSEN_signal, channel_PSEN_bkg]
    channel_list_all = channel_list_pp
    
    from sfdata import SFScanInfo
    scan = SFScanInfo(json_file)
    
    arrTimes_scan = []
    arrTimesAmp_scan = []
    
    for i, step in enumerate(scan):
        check_files_and_data(step)
        clear_output(wait=True)
        filename = scan.files[i][0].split('/')[-1].split('.')[0]
        print ('Processing: {}'.format(json_file.split('/')[-1]))
        print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
        
        resultsPP, results_FEL, _, _ = load_data_compact_FEL_pump(channel_list_pp, channel_list_all, step)
        
        sig = resultsPP[channel_PSEN_signal].pump
        back = resultsPP[channel_PSEN_bkg].pump
        
        bkg_files = find_backgrounds(step.fnames[0],'/scratch')
        print ("File recorded at {}".format(datetime.fromtimestamp(bkg_files[2])))
        print (bkg_files[0])
        print (bkg_files[1])
        background_from_fit = np.loadtxt(bkg_files[0])
        peakback = np.loadtxt(bkg_files[1])
        
        arrTimes, arrTimesAmp, sigtraces, peaktraces = arrivalTimes(target, calibration, back, sig, background_from_fit, peakback)
        
        arrTimes_scan.append(arrTimes)
        arrTimesAmp_scan.append(arrTimesAmp)
        
    arrTimes_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_scan)))
    arrTimesAmp_scan = np.asarray(list(itertools.chain.from_iterable(arrTimesAmp_scan)))
    
    return arrTimes, arrTimesAmp, arrTimes_scan, arrTimesAmp_scan, peaktraces
               
    
    
# -

def Two_TT_statistics_scan(json_file, target_1, calibration_1, target_2, calibration_2):
    channel_list_pp = [channel_Events, channel_PSEN_signal, channel_PSEN_bkg, channel_cam125_signal]
    channel_list_all = channel_list_pp
    
    from sfdata import SFScanInfo
    scan = SFScanInfo(json_file)
    
    arrTimes_1_scan = []
    arrTimesAmp_1_scan = []
    arrTimes_2_scan = []
    arrTimesAmp_2_scan = []
    
    for i, step in enumerate(scan):
        check_files_and_data(step)
        clear_output(wait=True)
        filename = scan.files[i][0].split('/')[-1].split('.')[0]
        print ('Processing: {}'.format(json_file.split('/')[-1]))
        print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
        
        resultsPP, results_FEL, _, _ = load_data_compact_FEL_pump(channel_list_pp, channel_list_all, step)
        
        sig_1 = resultsPP[channel_PSEN_signal].pump
        back_1 = resultsPP[channel_PSEN_bkg].pump
        sig_2 = resultsPP[channel_cam125_signal].pump
        back_2 = resultsPP[channel_cam125_signal].unpump 
        
        bkg_files = find_backgrounds(step.fnames[0],'/scratch')
        print ("File recorded at {}".format(datetime.fromtimestamp(bkg_files[2])))
        print (bkg_files[0])
        print (bkg_files[1])
        background_from_fit_1 = np.loadtxt(bkg_files[0])
        peakback_1 = np.loadtxt(bkg_files[1])
        _, _, peakback_2 = edge(target_2, back_2[0], back_2[0], 1, 0)
        
        arrTimes_1, arrTimesAmp_1, sigtraces_1, peaktraces_1 = arrivalTimes(target_1, calibration_1, back_1, sig_1, background_from_fit_1, peakback_1)
        arrTimes_2, arrTimesAmp_2, sigtraces_2, peaktraces_2 = arrivalTimes(target_2, calibration_2, back_2, sig_2, 1, peakback_2)
                
        arrTimes_1_scan.append(arrTimes_1)
        arrTimesAmp_1_scan.append(arrTimesAmp_1)
        arrTimes_2_scan.append(arrTimes_2)
        arrTimesAmp_2_scan.append(arrTimesAmp_2)
        
    arrTimes_1_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_1_scan)))
    arrTimesAmp_1_scan = np.asarray(list(itertools.chain.from_iterable(arrTimesAmp_1_scan)))
    arrTimes_2_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_2_scan)))
    arrTimesAmp_2_scan = np.asarray(list(itertools.chain.from_iterable(arrTimesAmp_2_scan)))
    
    
    return arrTimes_1, arrTimesAmp_1, arrTimes_1_scan, arrTimesAmp_1_scan, peaktraces_1,\
arrTimes_2, arrTimesAmp_2, arrTimes_2_scan, arrTimesAmp_2_scan, peaktraces_2


def YAG_scan_noTT(json_file, quantile):
    channel_list_pp = [channel_Events, channel_LaserDiode, channel_Laser_refDiode]
    channel_list_all = channel_list_pp

    ########################################################################

    from sfdata import SFScanInfo
    scan = SFScanInfo(json_file)

    if ' as delay' in scan.parameters['name'][0]:
    	print ('Scan is done with the stage in fs')
    	Delay_fs = scan.readbacks
    	Delay_mm = fs2mm(scan.readbacks,0)
    else:
    	print ('Scan is done with the stage in mm')
    	Delay_fs = mm2fs(scan.readbacks,0)
    	Delay_mm = scan.readbacks
    
    Pump_probe_all = []
    Pump_probe = []
    Pump_probe_avg =[]
    Pump_probe_std = []
    Pump_probe_std_err = []

    for i, step in enumerate(scan):
        check_files_and_data(step)
        clear_output(wait=True)
        filename = scan.files[i][0].split('/')[-1].split('.')[0]
        print ('Processing: {}'.format(json_file.split('/')[-1]))
        print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
        resultsPP, results_FEL, pids_pump, pids_unpump = load_data_compact_FEL_pump(channel_list_pp, channel_list_all, step)


        Laser_pump = resultsPP[channel_LaserDiode].pump
        Laser_ref_pump = resultsPP[channel_Laser_refDiode].pump
        Laser_unpump = resultsPP[channel_LaserDiode].unpump
        Laser_ref_unpump = resultsPP[channel_Laser_refDiode].unpump

        Laser_diff = -np.log10((Laser_pump) / (Laser_unpump))

        Pump_probe_all.append(Laser_diff)
        df_pump_probe = pd.DataFrame(Laser_diff)
        Pump_probe.append(np.nanquantile(df_pump_probe, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

        Pump_probe_std.append(np.nanmean(Laser_diff))
        Pump_probe_std_err.append(np.nanstd(Laser_diff))#/np.sqrt(len(Laser_diff)))

        Pump = np.median(Laser_pump)
        Unpump = np.median(Laser_unpump)
        Pump_probe_avg.append(-np.log10((Pump) / (Unpump)))

    Delay_mm = Delay_mm[:np.shape(Pump_probe)[0]]
    Delay_fs = Delay_fs[:np.shape(Pump_probe)[0]]
    
    Pump_probe_all = np.asarray(Pump_probe_all)
    Pump_probe = np.asarray(Pump_probe)
    Pump_probe_avg = np.asarray(Pump_probe_avg)
    Pump_probe_std = np.asarray(Pump_probe_std)
    Pump_probe_std_err = np.asarray(Pump_probe_std_err)
    
    minlen = min(len(i) for i in Pump_probe_all)
    Pump_probe_all = cut(Pump_probe_all, minlen)    

    print ("Quantile range = {}".format(0.5 - quantile/2), 0.5 + quantile/2)
    print ("Loaded {} files, size of the arrays = {}".format(len(scan.files), len(Pump_probe)))
    print ("Shape of pump probe data is {}".format(Pump_probe_all.shape))
    
    return Delay_mm,Delay_fs,Pump_probe,Pump_probe_all,Pump_probe_std,Pump_probe_std_err,Pump_probe_avg


def YAG_scan_one_TT(json_file, channel_delay_motor, timezero_mm, quantile, target, calibration, filterTime=2000, filterAmp=0):
    
    channel_list_pp = [channel_Events, channel_LaserDiode, channel_Laser_refDiode, channel_delay_motor,
                      channel_PSEN_signal, channel_PSEN_bkg]
    channel_list_all = channel_list_pp

    ########################################################################

    from sfdata import SFScanInfo
    scan = SFScanInfo(json_file)
    
    Delay_fs_stage = []
    Pump_probe = []
    Pump_probe_scan = []
    arrTimes_scan = []
    arrTimesAmp_scan = []
    Delays_fs_scan = []
    
    for i, step in enumerate(scan):
        check_files_and_data(step)
        clear_output(wait=True)
        filename = scan.files[i][0].split('/')[-1].split('.')[0]
        print ('Processing: {}'.format(json_file.split('/')[-1]))
        print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
        
        resultsPP, results_FEL, _, _ = load_data_compact_FEL_pump(channel_list_pp, channel_list_all, step)

        Laser_pump = resultsPP[channel_LaserDiode].pump
        Laser_ref_pump = resultsPP[channel_Laser_refDiode].pump
        Laser_unpump = resultsPP[channel_LaserDiode].unpump
        Laser_ref_unpump = resultsPP[channel_Laser_refDiode].unpump
        sig = resultsPP[channel_PSEN_signal].pump
        back = resultsPP[channel_PSEN_bkg].pump
        
        delay_shot = resultsPP[channel_delay_motor].pump
        delay_shot_fs = mm2fs(delay_shot, timezero_mm)
        Delay_fs_stage.append(delay_shot_fs.mean())

        Laser_diff = -np.log10((Laser_pump) / (Laser_unpump))
        
        bkg_files = find_backgrounds(step.fnames[0],'/scratch')
        print ("File recorded at {}".format(datetime.fromtimestamp(bkg_files[2])))
        print (bkg_files[0])
        print (bkg_files[1])
        background_from_fit = np.loadtxt(bkg_files[0])
        peakback = np.loadtxt(bkg_files[1])
        
        arrTimes, arrTimesAmp, sigtraces, peaktraces = arrivalTimes(target, calibration, back, sig, background_from_fit, peakback)
        
        index = (np.asarray(arrTimes) < filterTime) & (np.asarray(arrTimesAmp) > filterAmp)
        
        Delays_fs_scan.append(delay_shot_fs[index])
        arrTimes_scan.append(arrTimes[index])
        arrTimesAmp_scan.append(arrTimesAmp[index])
        Pump_probe_scan.append(Laser_diff[index])
        
        df_pump_probe = pd.DataFrame(Laser_diff)
        Pump_probe.append(np.nanquantile(df_pump_probe, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))
    
    Delays_fs_scan = np.asarray(list(itertools.chain.from_iterable(Delays_fs_scan)))
    arrTimes_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_scan)))
    Pump_probe_scan = np.asarray(list(itertools.chain.from_iterable(Pump_probe_scan)))
    arrTimesAmp_scan = np.asarray(list(itertools.chain.from_iterable(arrTimesAmp_scan)))
    
    Delays_fs_scan = np.asarray(Delays_fs_scan)
    arrTimes_scan = np.asarray(arrTimes_scan)
    Pump_probe_scan = np.asarray(Pump_probe_scan)
    arrTimesAmp_scan = np.asarray(arrTimesAmp_scan)
    Pump_probe = np.asarray(Pump_probe)
    
    Delays_corr_scan = Delays_fs_scan + arrTimes_scan#((p0 - np.array(edgePos))*px2fs)
    
    print ("Quantile range = {}".format(0.5 - quantile/2), 0.5 + quantile/2)
    print ("Loaded {} files, size of the arrays = {}".format(len(scan.files), len(Pump_probe)))
    
    return Delay_fs_stage, Delays_corr_scan,Pump_probe,Pump_probe_scan


def YAG_scan_two_TT(json_file, channel_delay_motor, timezero_mm, quantile, 
                    target_1, calibration_1, target_2, calibration_2,
                    filterTime_1=2000, filterAmp_1=0, filterTime_2=2000, filterAmp_2=0):
    
    channel_list_pp = [channel_Events, channel_LaserDiode, channel_Laser_refDiode, channel_delay_motor,
                      channel_PSEN_signal, channel_PSEN_bkg,
                      channel_cam125_signal]
    channel_list_all = channel_list_pp

    ########################################################################

    from sfdata import SFScanInfo
    scan = SFScanInfo(json_file)
    
    Delay_fs_stage = []
    Pump_probe = []
    Pump_probe_scan = []
    arrTimes_1_scan = []
    arrTimesAmp_1_scan = []
    arrTimes_2_scan = []
    arrTimesAmp_2_scan = []
    Delays_fs_scan = []
    
    for i, step in enumerate(scan):
        check_files_and_data(step)
        clear_output(wait=True)
        filename = scan.files[i][0].split('/')[-1].split('.')[0]
        print ('Processing: {}'.format(json_file.split('/')[-1]))
        print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
        
        resultsPP, results_FEL, _, _ = load_data_compact_FEL_pump(channel_list_pp, channel_list_all, step)

        Laser_pump = resultsPP[channel_LaserDiode].pump
        Laser_ref_pump = resultsPP[channel_Laser_refDiode].pump
        Laser_unpump = resultsPP[channel_LaserDiode].unpump
        Laser_ref_unpump = resultsPP[channel_Laser_refDiode].unpump
        sig_1 = resultsPP[channel_PSEN_signal].pump
        back_1 = resultsPP[channel_PSEN_bkg].pump
        sig_2 = resultsPP[channel_cam125_signal].pump
        back_2 = resultsPP[channel_cam125_signal].unpump
        
        delay_shot = resultsPP[channel_delay_motor].pump
        delay_shot_fs = mm2fs(delay_shot, timezero_mm)
        Delay_fs_stage.append(delay_shot_fs.mean())

        Laser_diff = -np.log10((Laser_pump) / (Laser_unpump))
        
        bkg_files = find_backgrounds(step.fnames[0],'/scratch')
        print ("File recorded at {}".format(datetime.fromtimestamp(bkg_files[2])))
        print (bkg_files[0])
        print (bkg_files[1])
        background_from_fit_1 = np.loadtxt(bkg_files[0])
        peakback_1 = np.loadtxt(bkg_files[1])
        _, _, peakback_2 = edge(target_2, back_2[0], back_2[0], 1, 0)
            
        arrTimes_1, arrTimesAmp_1, sigtraces_1, peaktraces_1 = arrivalTimes(target_1, calibration_1, back_1, sig_1, background_from_fit_1, peakback_1)
        arrTimes_2, arrTimesAmp_2, sigtraces_2, peaktraces_2 = arrivalTimes(target_2, calibration_2, back_2, sig_2, 1, peakback_2)
        
        index_1 = (np.asarray(arrTimes_1) < filterTime_1) & (np.asarray(arrTimesAmp_1) > filterAmp_1)
        index_2 = (np.asarray(arrTimes_2) < filterTime_2) & (np.asarray(arrTimesAmp_2) > filterAmp_2)
        
        index = np.logical_and.reduce((index_1, index_2))
        
        Delays_fs_scan.append(delay_shot_fs[index])
        arrTimes_1_scan.append(arrTimes_1[index])
        arrTimesAmp_1_scan.append(arrTimesAmp_1[index])
        arrTimes_2_scan.append(arrTimes_2[index] + 850)
        arrTimesAmp_2_scan.append(arrTimesAmp_2[index])
        Pump_probe_scan.append(Laser_diff[index])
        
        df_pump_probe = pd.DataFrame(Laser_diff)
        Pump_probe.append(np.nanquantile(df_pump_probe, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))
    
    Delays_fs_scan = np.asarray(list(itertools.chain.from_iterable(Delays_fs_scan)))
    arrTimes_1_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_1_scan)))
    arrTimesAmp_1_scan = np.asarray(list(itertools.chain.from_iterable(arrTimesAmp_1_scan)))
    arrTimes_2_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_2_scan)))
    arrTimesAmp_2_scan = np.asarray(list(itertools.chain.from_iterable(arrTimesAmp_2_scan)))
    
    Pump_probe_scan = np.asarray(list(itertools.chain.from_iterable(Pump_probe_scan)))
    
    Delays_fs_scan = np.asarray(Delays_fs_scan)
    arrTimes_1_scan = np.asarray(arrTimes_1_scan)
    arrTimesAmp_1_scan = np.asarray(arrTimesAmp_1_scan)
    arrTimes_2_scan = np.asarray(arrTimes_2_scan)
    arrTimesAmp_2_scan = np.asarray(arrTimesAmp_2_scan)
    Pump_probe_scan = np.asarray(Pump_probe_scan)
    Pump_probe = np.asarray(Pump_probe)
    
    Delays_corr_1_scan = Delays_fs_scan + arrTimes_1_scan#((p0 - np.array(edgePos))*px2fs)
    Delays_corr_2_scan = Delays_fs_scan + arrTimes_2_scan#((p0 - np.array(edgePos))*px2fs)
    
    print ("Quantile range = {}".format(0.5 - quantile/2), 0.5 + quantile/2)
    print ("Loaded {} files, size of the arrays = {}".format(len(scan.files), len(Pump_probe)))
    
    return Delay_fs_stage, Delays_corr_1_scan, Delays_corr_2_scan, Pump_probe, Pump_probe_scan
