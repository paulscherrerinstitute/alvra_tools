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

TT_PSEN119 = [channel_PSEN119_signal, channel_PSEN119_bkg] 
TT_PSEN124 = [channel_PSEN125_signal]
TT_PSEN126 = [channel_PSEN125_signal, channel_PSEN125_bkg, channel_PSEN125_arrTimes, channel_PSEN125_arrTimesAmp, channel_PSEN125_peaks, channel_PSEN125_edges]

#########################################################

def TT_statistics_scan(scan, TT, target, calibration):
    channel_list_pp = [channel_Events] + TT
    channel_list_all = channel_list_pp
    
    #from sfdata import SFScanInfo
    #scan = SFScanInfo(json_file)
    
    arrTimes_scan = []
    arrTimesAmp_scan = []
    
    for i, step in enumerate(scan):
        check_files_and_data(step)
        check = get_filesize_diff(step)
        if check:
		clear_output(wait=True)
		filename = scan.files[i][0].split('/')[-1].split('.')[0]
		print ('Processing: {}'.format(scan.fname.split('/')[-3]))
		print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
		
		resultsPP, results_FEL, _, _ = load_data_compact_pump_probe(channel_list_pp, channel_list_all, step)

		arrTimes, arrTimesAmp, sigtraces, peaktraces = get_arrTimes(resultsPP, TT, target, calibration)
		
		arrTimes_scan.append(arrTimes)
		arrTimesAmp_scan.append(arrTimesAmp)
		
    arrTimes_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_scan)))
    arrTimesAmp_scan = np.asarray(list(itertools.chain.from_iterable(arrTimesAmp_scan)))

    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(arrTimes_scan), len(scan)))
    
    return arrTimes, arrTimesAmp, arrTimes_scan, arrTimesAmp_scan, peaktraces
                 
#########################################################

def Two_TT_statistics_scan(scan, TT1, TT2, target_1, calibration_1, target_2, calibration_2):
    channel_list_pp = [channel_Events, channel_PSEN_signal, channel_PSEN_bkg, channel_cam125_signal]
    channel_list_all = channel_list_pp
    
    #from sfdata import SFScanInfo
    #scan = SFScanInfo(json_file)
    
    arrTimes_1_scan = []
    arrTimesAmp_1_scan = []
    arrTimes_2_scan = []
    arrTimesAmp_2_scan = []
    
    for i, step in enumerate(scan):
        check_files_and_data(step)
        check = get_filesize_diff(step)
        if check:
		clear_output(wait=True)
		filename = scan.files[i][0].split('/')[-1].split('.')[0]
		print ('Processing: {}'.format(scan.fname.split('/')[-3]))
		print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
		
		resultsPP, results_FEL, _, _ = load_data_compact_pump_probe(channel_list_pp, channel_list_all, step)

		arrTimes_1, arrTimesAmp_1, sigtraces_1, peaktraces_1 = get_arrTimes(resultsPP, TT1, target_1, calibration_1)
		arrTimes_2, arrTimesAmp_2, sigtraces_2, peaktraces_2 = get_arrTimes(resultsPP, TT2, target_2, calibration_2)        

		arrTimes_1_scan.append(arrTimes_1)
		arrTimesAmp_1_scan.append(arrTimesAmp_1)
		arrTimes_2_scan.append(arrTimes_2)
		arrTimesAmp_2_scan.append(arrTimesAmp_2)
        
    arrTimes_1_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_1_scan)))
    arrTimesAmp_1_scan = np.asarray(list(itertools.chain.from_iterable(arrTimesAmp_1_scan)))
    arrTimes_2_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_2_scan)))
    arrTimesAmp_2_scan = np.asarray(list(itertools.chain.from_iterable(arrTimesAmp_2_scan)))
    
    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(arrTimes_scan), len(scan)))
    
    return arrTimes_1, arrTimesAmp_1, arrTimes_1_scan, arrTimesAmp_1_scan, peaktraces_1,\
arrTimes_2, arrTimesAmp_2, arrTimes_2_scan, arrTimesAmp_2_scan, peaktraces_2

#########################################################

def YAG_scan_noTT(scan, quantile):
    channel_list_pp = [channel_Events, channel_LaserDiode, channel_Laser_refDiode]
    channel_list_all = channel_list_pp

    #from sfdata import SFScanInfo
    #scan = SFScanInfo(json_file)

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
        check = get_filesize_diff(step)
        if check:
		clear_output(wait=True)
		filename = scan.files[i][0].split('/')[-1].split('.')[0]
		print ('Processing: {}'.format(scan.fname.split('/')[-3]))
		print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
		resultsPP, results_FEL, pids_pump, pids_unpump = load_data_compact_pump_probe(channel_list_pp, channel_list_all, step)


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
    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(Delay_mm), len(scan)))
    
    return Delay_mm,Delay_fs,Pump_probe,Pump_probe_all,Pump_probe_std,Pump_probe_std_err,Pump_probe_avg

#########################################################

def YAG_scan_one_TT(scan, TT, channel_delay_motor, timezero_mm, quantile, target, calibration, filterTime=2000, filterAmp=0):
    
    channel_list_pp = [channel_Events, channel_LaserDiode, channel_Laser_refDiode, channel_delay_motor] + TT
    channel_list_all = channel_list_pp

    #from sfdata import SFScanInfo
    #scan = SFScanInfo(json_file)
   
    if ' as delay' in scan.parameters['name'][0]:
        print ('Scan is done with the stage in fs')
        Delay_fs = scan.readbacks
        Delay_mm = fs2mm(scan.readbacks,0)
    else:
        print ('Scan is done with the stage in mm')
        Delay_fs = mm2fs(scan.readbacks,0)
        Delay_mm = scan.readbacks
 
    Delay_fs_stage = []
    Pump_probe = []
    Pump_probe_scan = []
    arrTimes_scan = []
    arrTimesAmp_scan = []
    Delays_fs_scan = []
    
    for i, step in enumerate(scan):
        check_files_and_data(step)
        check = get_filesize_diff(step)
        if check:
		clear_output(wait=True)
		filename = scan.files[i][0].split('/')[-1].split('.')[0]
		print ('Processing: {}'.format(scan.fname.split('/')[-3]))
		print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
		
		resultsPP, results_FEL, _, _ = load_data_compact_pump_probe(channel_list_pp, channel_list_all, step)

		Laser_pump = resultsPP[channel_LaserDiode].pump
		Laser_ref_pump = resultsPP[channel_Laser_refDiode].pump
		Laser_unpump = resultsPP[channel_LaserDiode].unpump
		Laser_ref_unpump = resultsPP[channel_Laser_refDiode].unpump
		
		delay_shot = resultsPP[channel_delay_motor].pump
		delay_shot_fs = mm2fs(delay_shot, timezero_mm)
		Delay_fs_stage.append(delay_shot_fs.mean())

		Laser_diff = -np.log10((Laser_pump) / (Laser_unpump))

		arrTimes, arrTimesAmp, sigtraces, peaktraces = get_arrTimes(resultsPP, step, TT, target, calibration)
		
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
    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(Pump_probe), len(scan)))
    
    return Delays_fs_scan, Delays_corr_scan,Pump_probe,Pump_probe_scan

#########################################################

def YAG_scan_one_TT_bs(scan, TT, channel_delay_motor, timezero_mm, quantile, filterTime=2000, filterAmp=0):
    
    channel_list_pp = [channel_Events, channel_LaserDiode, channel_Laser_refDiode, channel_delay_motor] + TT
    channel_list_all = channel_list_pp

    #from sfdata import SFScanInfo
    #scan = SFScanInfo(json_file)
   
    if ' as delay' in scan.parameters['name'][0]:
        print ('Scan is done with the stage in fs')
        Delay_fs = scan.readbacks
        Delay_mm = fs2mm(scan.readbacks,0)
    else:
        print ('Scan is done with the stage in mm')
        Delay_fs = mm2fs(scan.readbacks,0)
        Delay_mm = scan.readbacks
 
    Delay_fs_stage = []
    Pump_probe = []
    Pump_probe_scan = []
    arrTimes_scan = []
    arrTimesAmp_scan = []
    Delays_fs_scan = []
    
    for i, step in enumerate(scan):
        check_files_and_data(step)
        check = get_filesize_diff(step)
        if check:
		clear_output(wait=True)
		filename = scan.files[i][0].split('/')[-1].split('.')[0]
		print ('Processing: {}'.format(scan.fname.split('/')[-3]))
		print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
		
		resultsPP, results_FEL, _, _ = load_data_compact_pump_probe(channel_list_pp, channel_list_all, step)

		Laser_pump = resultsPP[channel_LaserDiode].pump
		Laser_ref_pump = resultsPP[channel_Laser_refDiode].pump
		Laser_unpump = resultsPP[channel_LaserDiode].unpump
		Laser_ref_unpump = resultsPP[channel_Laser_refDiode].unpump
		
		delay_shot = resultsPP[channel_delay_motor].pump
		delay_shot_fs = mm2fs(delay_shot, timezero_mm)
		Delay_fs_stage.append(delay_shot_fs.mean())

		Laser_diff = -np.log10((Laser_pump) / (Laser_unpump))

		arrTimes = resultsPP[channel_PSEN125_arrTimes].pump
		arrTimesAmp = resultsPP[channel_PSEN125_arrTimesAmp].pump
		sigtraces = resultsPP[channel_PSEN125_edges].pump
		peaktraces = resultsPP[channel_PSEN125_peaks].pump

		#arrTimes, arrTimesAmp, sigtraces, peaktraces = get_arrTimes(resultsPP, step, TT, target, calibration)
		
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
    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(Pump_probe), len(scan)))
    
    return Delays_fs_scan, Delays_corr_scan,Pump_probe,Pump_probe_scan

#########################################################

def YAG_scan_two_TT(scan, TT1, TT2, channel_delay_motor, timezero_mm, quantile, 
                    target_1, calibration_1, target_2, calibration_2,
                    filterTime_1=2000, filterAmp_1=0, filterTime_2=2000, filterAmp_2=0):
    
    channel_list_pp = [channel_Events, channel_LaserDiode, channel_Laser_refDiode, channel_delay_motor] + TT1 + TT2
                     
    channel_list_all = channel_list_pp

    #from sfdata import SFScanInfo
    #scan = SFScanInfo(json_file)
    
    if ' as delay' in scan.parameters['name'][0]:
        print ('Scan is done with the stage in fs')
        Delay_fs_stage = scan.readbacks
        Delay_mm = fs2mm(scan.readbacks,0)
    else:
        print ('Scan is done with the stage in mm')
        Delay_fs_stage = mm2fs(scan.readbacks,0)
        Delay_mm = scan.readbacks
    
    pids_pump_scan = [] 
    Pump_probe = []
    Pump_probe_scan = []
    arrTimes_1_scan = []
    arrTimesAmp_1_scan = []
    arrTimes_2_scan = []
    arrTimesAmp_2_scan = []
    Delays_fs_scan = []
    Izero_scan = []
    
    for i, step in enumerate(scan):
        check_files_and_data(step)
        check = get_filesize_diff(step)
        if check:
		clear_output(wait=True)
		filename = scan.files[i][0].split('/')[-1].split('.')[0]
		print ('Processing: {}'.format(scan.fname.split('/')[-3]))
		print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
		
		resultsPP, results_FEL,_, _ = load_data_compact_pump_probe(channel_list_pp, channel_list_all, step)
		
		Izero = resultsPP[channel_Izero122].pump
		Laser_pump = resultsPP[channel_LaserDiode].pump
		Laser_ref_pump = resultsPP[channel_Laser_refDiode].pump
		Laser_unpump = resultsPP[channel_LaserDiode].unpump
		Laser_ref_unpump = resultsPP[channel_Laser_refDiode].unpump
		
		delay_shot = resultsPP[channel_delay_motor].pump
		delay_shot_fs = mm2fs(delay_shot, timezero_mm)

		Laser_diff = -np.log10((Laser_pump) / (Laser_unpump))

		arrTimes_1, arrTimesAmp_1, sigtraces_1, peaktraces_1 = get_arrTimes(resultsPP, TT1, target_1, calibration_1)
		arrTimes_2, arrTimesAmp_2, sigtraces_2, peaktraces_2 = get_arrTimes(resultsPP, TT2, target_2, calibration_2)             
		
		print ("arrTimes M2={}, arrTimes M1={}".format(np.shape(arrTimes_1), np.shape(arrTimes_2)))
	     
		index_1 = (np.asarray(arrTimes_1) < filterTime_1) & (np.asarray(arrTimesAmp_1) > filterAmp_1)
		index_2 = (np.asarray(arrTimes_2) < filterTime_2) & (np.asarray(arrTimesAmp_2) > filterAmp_2)
		
		index = np.logical_and.reduce((index_1, index_2))
		
		#pids_pump_scan.append(pids_pump[index]) 
		Delays_fs_scan.append(delay_shot_fs[index])
		arrTimes_1_scan.append(arrTimes_1[index])
		arrTimesAmp_1_scan.append(arrTimesAmp_1[index])
		arrTimes_2_scan.append(arrTimes_2[index])
		arrTimesAmp_2_scan.append(arrTimesAmp_2[index])
		Pump_probe_scan.append(Laser_diff[index])
		Izero_scan.append(Izero[index])        

		df_pump_probe = pd.DataFrame(Laser_diff)
		Pump_probe.append(np.nanquantile(df_pump_probe, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))
    
    #pids_pump_scan = np.asarray(list(itertools.chain.from_iterable(pids_pump_scan)))
    Delays_fs_scan = np.asarray(list(itertools.chain.from_iterable(Delays_fs_scan)))
    arrTimes_1_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_1_scan)))
    arrTimesAmp_1_scan = np.asarray(list(itertools.chain.from_iterable(arrTimesAmp_1_scan)))
    arrTimes_2_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_2_scan)))
    arrTimesAmp_2_scan = np.asarray(list(itertools.chain.from_iterable(arrTimesAmp_2_scan)))
    
    Pump_probe_scan = np.asarray(list(itertools.chain.from_iterable(Pump_probe_scan)))
    Izero_scan = np.asarray(list(itertools.chain.from_iterable(Izero_scan)))    

    #pids_pump_scan = np.asarray(pids_pump_scan)
    Delays_fs_scan = np.asarray(Delays_fs_scan)
    arrTimes_1_scan = np.asarray(arrTimes_1_scan)
    arrTimesAmp_1_scan = np.asarray(arrTimesAmp_1_scan)
    arrTimes_2_scan = np.asarray(arrTimes_2_scan)
    arrTimesAmp_2_scan = np.asarray(arrTimesAmp_2_scan)
    Pump_probe_scan = np.asarray(Pump_probe_scan)
    Izero_scan = np.asarray(Izero_scan)
    Pump_probe = np.asarray(Pump_probe)
    
    Delays_corr_1_scan = Delays_fs_scan + arrTimes_1_scan#((p0 - np.array(edgePos))*px2fs)
    Delays_corr_2_scan = Delays_fs_scan + arrTimes_2_scan#((p0 - np.array(edgePos))*px2fs)
    
    print ("Quantile range = {}".format(0.5 - quantile/2), 0.5 + quantile/2)
    print ("Loaded {} files, size of the arrays = {}".format(len(scan.files), len(Pump_probe)))
    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(Pump_probe), len(scan)))
    
    return Delay_fs_stage, Delays_fs_scan, Delays_corr_1_scan, Delays_corr_2_scan, Pump_probe, Pump_probe_scan, Izero_scan, arrTimesAmp_1_scan, arrTimesAmp_2_scan, pids_pump_scan

#########################################################

def save_run_array_YAG(reducedir, run_name, Delmm, Delfs, PP, PPall, PPavg):
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "Delay_mm": Delmm,
                                    "Delay_fs": Delfs,
                                    "Pump_probe": PP,
                                    "Pump_probe_all": PPall,
                                    "Pump_probe_avg": PPavg}

    np.save(reducedir+run_name+'/run_array', run_array)

#########################################################

def save_run_array_YAG_TT(reducedir, run_name, Delfs, Delcorr, PP, PPscan):
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "Delay_fs": Delfs,
                                    "Delay_corr": Delcorr,
                                    "Pump_probe": PP,
                                    "Pump_probe_scan": PPscan}

    np.save(reducedir+run_name+'/run_array', run_array)


