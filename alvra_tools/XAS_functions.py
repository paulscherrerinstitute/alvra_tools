import numpy as np
import json, glob
import os, math
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from IPython.display import clear_output, display
from datetime import datetime
from scipy.stats.stats import pearsonr
import itertools

from alvra_tools.load_data import *
from alvra_tools.channels import *
from alvra_tools.utils import *
from alvra_tools.timing_tool import *

######################################

#def get_timezero_NBS(json_file):
#    from sfdata import SFScanInfo
#    scan = SFScanInfo(json_file)
#    fn = scan.files[0][0].replace('.BSDATA.h5','*').replace('.PVCHANNELS.h5','*').replace('.CAMERAS.h5','*').replace('.*JF*.h5','*')
#    with SFDataFiles(fn) as sfd:
#        ch = sfd['SARES11-CVME-EVR0:DUMMY_PV2_NBS']
#        t0mm = ch.data[0]
#    return t0mm

######################################

def correlation_filter_static(FluoData, Izero, quantile):
    
    FluoData_norm = FluoData / Izero

    qnt_low = np.nanquantile(FluoData_norm, 0.5 - quantile/2)
    qnt_high = np.nanquantile(FluoData_norm, 0.5 + quantile/2)

    condition_low = FluoData_norm > qnt_low
    condition_high = FluoData_norm < qnt_high

    correlation_filter = condition_low & condition_high
    
    FluoData_filter = FluoData[correlation_filter]
    Izero_filter = Izero[correlation_filter]

    print ('{} shots out of {} survived'.format(np.shape(FluoData_filter), np.shape(FluoData)))

    return (FluoData_filter, Izero_filter)

######################################

def correlation_filter(FluoData_pump, FluoData_unpump, Izero_pump, Izero_unpump, quantile):
    
    FluoData_pump_norm = FluoData_pump / Izero_pump
    FluoData_unpump_norm = FluoData_unpump / Izero_unpump

    qnt_low_pump = np.nanquantile(FluoData_pump_norm, 0.5 - quantile/2)
    qnt_high_pump = np.nanquantile(FluoData_pump_norm, 0.5 + quantile/2)
    qnt_low_unpump = np.nanquantile(FluoData_unpump_norm, 0.5 - quantile/2)
    qnt_high_unpump = np.nanquantile(FluoData_unpump_norm, 0.5 + quantile/2)

    condition_pump_low = FluoData_pump_norm > qnt_low_pump
    condition_pump_high = FluoData_pump_norm < qnt_high_pump
    condition_unpump_low = FluoData_unpump_norm > qnt_low_unpump
    condition_unpump_high = FluoData_unpump_norm < qnt_high_unpump

    correlation_filter = condition_pump_low & condition_pump_high & condition_unpump_low & condition_unpump_high
    
    FluoData_pump_filter = FluoData_pump[correlation_filter]
    FluoData_unpump_filter = FluoData_unpump[correlation_filter]
    Izero_pump_filter = Izero_pump[correlation_filter]
    Izero_unpump_filter = Izero_unpump[correlation_filter]

    #FluoData_pump_filter_norm = FluoData_pump_filter / Izero_pump_filter
    #FluoData_unpump_filter_norm = FluoData_unpump_filter / Izero_unpump_filter

    print ('{} shots out of {} survived'.format(np.shape(FluoData_pump_filter), np.shape(FluoData_pump)))

    return (FluoData_pump_filter, FluoData_unpump_filter, Izero_pump_filter, Izero_unpump_filter)

######################################

def correlation_filter_TT(FluoData_pump, FluoData_unpump, Izero_pump, Izero_unpump, arrTimes, delay_fs, quantile):
    
    FluoData_pump_norm = FluoData_pump / Izero_pump
    FluoData_unpump_norm = FluoData_unpump / Izero_unpump

    qnt_low_pump = np.nanquantile(FluoData_pump_norm, 0.5 - quantile/2)
    qnt_high_pump = np.nanquantile(FluoData_pump_norm, 0.5 + quantile/2)
    qnt_low_unpump = np.nanquantile(FluoData_unpump_norm, 0.5 - quantile/2)
    qnt_high_unpump = np.nanquantile(FluoData_unpump_norm, 0.5 + quantile/2)

    condition_pump_low = FluoData_pump_norm > qnt_low_pump
    condition_pump_high = FluoData_pump_norm < qnt_high_pump
    condition_unpump_low = FluoData_unpump_norm > qnt_low_unpump
    condition_unpump_high = FluoData_unpump_norm < qnt_high_unpump

    correlation_filter = condition_pump_low & condition_pump_high & condition_unpump_low & condition_unpump_high
    
    FluoData_pump_filter = FluoData_pump[correlation_filter]
    FluoData_unpump_filter = FluoData_unpump[correlation_filter]
    Izero_pump_filter = Izero_pump[correlation_filter]
    Izero_unpump_filter = Izero_unpump[correlation_filter]
    arrTimes_filter = arrTimes[correlation_filter]
    delay_fs_filter = delay_fs[correlation_filter]

    #FluoData_pump_filter_norm = FluoData_pump_filter / Izero_pump_filter
    #FluoData_unpump_filter_norm = FluoData_unpump_filter / Izero_unpump_filter

    print ('{} shots out of {} survived'.format(np.shape(FluoData_pump_filter), np.shape(FluoData_pump)))

    return (FluoData_pump_filter, FluoData_unpump_filter, Izero_pump_filter, Izero_unpump_filter, arrTimes_filter, delay_fs_filter)

######################################

def Get_correlation_from_scan_static(scan, index, diode, Izero, quantile):
    channels = [channel_Events, diode, Izero]

    data = scan[index]
    results,_ = load_data_compact(channels, data)
    data.close()
    
    clear_output(wait=True)
    
    Fluo = results[diode]
    IzeroFEL = results[Izero]
    
    Fluo_filter, Izero_filter = correlation_filter_static(Fluo, IzeroFEL, quantile)

    return (Fluo, IzeroFEL, Fluo_filter, Izero_filter)
    
######################################

def Get_correlation_from_scan(scan, index, diode, Izero, quantile):
    channels_pp = [channel_Events, diode, Izero]
    channels_all = channels_pp

    data = scan[index]
    results,_,_,_ = load_data_compact_pump_probe(channels_pp, channels_all, data)
    data.close()
    
    clear_output(wait=True)
    
    Fluo_pump = results[diode].pump
    Fluo_unpump = results[diode].unpump
    Izero_pump = results[Izero].pump
    Izero_unpump = results[Izero].unpump
    
    Fluo_pump_filter, Fluo_unpump_filter, Izero_pump_filter, Izero_unpump_filter = correlation_filter(Fluo_pump, Fluo_unpump, Izero_pump, Izero_unpump, quantile)

    return (Fluo_pump, Fluo_unpump, Izero_pump, Izero_unpump, Fluo_pump_filter, Fluo_unpump_filter, Izero_pump_filter, Izero_unpump_filter)
    
######################################

def XAS_scan_1diode_static(scan, diode, Izero, quantile):
    channels = [channel_Events, diode, Izero]

#    from sfdata import SFScanInfo
#    scan = SFScanInfo(json_file)

    Adjustable = scan.readbacks

    DataFluo = []
    IzeroFEL = []
 
    correlation = []

    for i, step in enumerate(scan):
       check_files_and_data(step)
       clear_output(wait=True)
       filename = scan.files[i][0].split('/')[-1].split('.')[0]
       print ('Processing: {}'.format(scan.fname.split('/')[-3]))
       print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))

       results,_ = load_data_compact(channels, step)
    
       Fluo_shot = results[diode]
       IzeroFEL_shot = results[Izero]
    
       ######################################
       ### filter Diode1 data
       ######################################

       Diode_shot_filter, Izero_filter = correlation_filter_static(Fluo_shot, IzeroFEL_shot, quantile)
       Diode_shot_filter = Diode_shot_filter / Izero_filter

       ######################################
       ### make dataframes Diode1
       ######################################

       df_fluo = pd.DataFrame(Diode_shot_filter)
       DataFluo.append(np.nanquantile(df_fluo, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       correlation.append(pearsonr(IzeroFEL_shot,Fluo_shot)[0])
       IzeroFEL.append(np.mean(IzeroFEL_shot))
	
       print ('Step {} of {}: Processed {}'.format(i+1, len(scan.files), filename))
       print ("correlation Diode (all shots) = {}".format(pearsonr(IzeroFEL_shot,Fluo_shot)[0]))
    
    Adjustable = Adjustable[:np.shape(DataFluo)[0]]
    
    DataFluo = np.asarray(DataFluo)
    IzeroFEL = np.asarray(Izero)
    correlation = np.asarray(correlation)

    return (DataFluo, IzeroFEL, correlation, Adjustable)

######################################

def XAS_scan_1diode(scan, diode, Izero, quantile):
    channels_pp = [channel_Events, diode, Izero]
    channels_all = channels_pp    

#    from sfdata import SFScanInfo
#    scan = SFScanInfo(json_file)

    Adjustable = scan.readbacks

    DataFluo_pump = []
    DataFluo_unpump = []
    Pump_probe = []

    Izero_pump = []
    Izero_unpump = []

    correlation = []
    goodshots = []

    for i, step in enumerate(scan):
       check_files_and_data(step)
       clear_output(wait=True)
       filename = scan.files[i][0].split('/')[-1].split('.')[0]
       print ('Processing: {}'.format(scan.fname.split('/')[-3]))
       print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))

       resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)
    
       IzeroFEL_pump_shot = resultsPP[Izero].pump
       IzeroFEL_unpump_shot = resultsPP[Izero].unpump
       Fluo_pump_shot = resultsPP[diode].pump
       Fluo_unpump_shot = resultsPP[diode].unpump
    
       ######################################
       ### filter Diode1 data
       ######################################
    
       Diode_pump_shot_filter, Diode_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter = correlation_filter(Fluo_pump_shot, Fluo_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, quantile)
       Diode_pump_shot_filter = Diode_pump_shot_filter / Izero_pump_filter
       Diode_unpump_shot_filter = Diode_unpump_shot_filter / Izero_unpump_filter

       Pump_probe_shot = Diode_pump_shot_filter - Diode_unpump_shot_filter

       ######################################
       ### make dataframes Diode1
       ######################################

       df_pump = pd.DataFrame(Diode_pump_shot_filter)
       DataFluo_pump.append(np.nanquantile(df_pump, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_unpump = pd.DataFrame(Diode_unpump_shot_filter)
       DataFluo_unpump.append(np.nanquantile(df_unpump, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_pump_probe_APD1 = pd.DataFrame(Pump_probe_shot)
       Pump_probe.append(np.nanquantile(df_pump_probe_APD1, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))
       
       goodshots.append(len(Pump_probe_shot))
       correlation.append(pearsonr(IzeroFEL_pump_shot,Fluo_pump_shot)[0])
       Izero_pump.append(np.mean(IzeroFEL_pump_shot))
       Izero_unpump.append(np.mean(IzeroFEL_unpump_shot))
	
       print ('Step {} of {}: Processed {}'.format(i+1, len(scan.files), filename))
       print ("correlation Diode (all shots) = {}".format(pearsonr(IzeroFEL_pump_shot,Fluo_pump_shot)[0]))
    
    Adjustable = Adjustable[:np.shape(Pump_probe)[0]]
    
    DataFluo_pump = np.asarray(DataFluo_pump)
    DataFluo_unpump = np.asarray(DataFluo_unpump)
    Pump_probe = np.asarray(Pump_probe)

    Izero_pump = np.asarray(Izero_pump)
    Izero_unpump = np.asarray(Izero_unpump)
    correlation = np.asarray(correlation)

    return (DataFluo_pump, DataFluo_unpump, Pump_probe, Izero_pump, Izero_unpump, correlation, Adjustable, goodshots)

######################################

def XAS_scan_2diodes_static(scan, diode1, diode2, Izero, quantile):
    channels = [channel_Events, diode1, diode2, Izero]

#    from sfdata import SFScanInfo
#    scan = SFScanInfo(json_file)

    Adjustable = scan.readbacks

    DataFluo1 = []
    DataFluo2 = []
    IzeroFEL = []
 
    correlation1 = []
    correlation2 = []

    for i, step in enumerate(scan):
       check_files_and_data(step)
       clear_output(wait=True)
       filename = scan.files[i][0].split('/')[-1].split('.')[0]
       print ('Processing: {}'.format(scan.fname.split('/')[-3]))
       print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))

       results,_ = load_data_compact(channels, step)
    
       Fluo_shot1 = results[diode1]
       Fluo_shot2 = results[diode2]
       IzeroFEL_shot = results[Izero]
    
       ######################################
       ### filter Diode1 data
       ######################################

       Diode1_shot_filter, Izero_filter = correlation_filter_static(Fluo_shot1, IzeroFEL_shot, quantile)
       Diode1_shot_filter = Diode1_shot_filter / Izero_filter

       ######################################
       ### filter Diode2 data
       ######################################

       Diode2_shot_filter, Izero_filter = correlation_filter_static(Fluo_shot2, IzeroFEL_shot, quantile)
       Diode2_shot_filter = Diode2_shot_filter / Izero_filter

       ######################################
       ### make dataframes Diode1
       ######################################

       df_fluo1 = pd.DataFrame(Diode1_shot_filter)
       DataFluo1.append(np.nanquantile(df_fluo1, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       ######################################
       ### make dataframes Diode2
       ######################################

       df_fluo2 = pd.DataFrame(Diode2_shot_filter)
       DataFluo2.append(np.nanquantile(df_fluo2, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       correlation1.append(pearsonr(IzeroFEL_shot,Fluo_shot1)[0])
       correlation2.append(pearsonr(IzeroFEL_shot,Fluo_shot2)[0])
       IzeroFEL.append(np.mean(IzeroFEL_shot))
	
       print ('Step {} of {}: Processed {}'.format(i+1, len(scan.files), filename))
       print ("correlation Diode1 (all shots) = {}".format(pearsonr(IzeroFEL_shot,Fluo_shot1)[0]))
       print ("correlation Diode2 (all shots) = {}".format(pearsonr(IzeroFEL_shot,Fluo_shot2)[0]))
    
    Adjustable = Adjustable[:np.shape(DataFluo1)[0]]
    
    DataFluo1 = np.asarray(DataFluo1)
    DataFluo2 = np.asarray(DataFluo2)
    IzeroFEL = np.asarray(Izero)
    correlation1 = np.asarray(correlation1)
    correlation2 = np.asarray(correlation2)

    return (DataFluo1, DataFluo2, IzeroFEL, correlation1, correlation2, Adjustable)

######################################

def XAS_scan_2diodes(scan, diode1, diode2, Izero, quantile):
    channels_pp = [channel_Events, diode1, diode2, Izero]
    channels_all = channels_pp   

#    from sfdata import SFScanInfo
#    scan = SFScanInfo(json_file)

    Adjustable = scan.readbacks

    DataFluo1_pump = []
    DataFluo1_unpump = []
    Pump_probe1 = []

    DataFluo2_pump = []
    DataFluo2_unpump = []
    Pump_probe2 = []

    Izero_pump = []
    Izero_unpump = []

    correlation1 = []
    correlation2 = []

    goodshots1 = []
    goodshots2 = []

    for i, step in enumerate(scan):
       check_files_and_data(step)
       clear_output(wait=True)
       filename = scan.files[i][0].split('/')[-1].split('.')[0]
       print ('Processing: {}'.format(scan.fname.split('/')[-3]))
       print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))

       resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)
    
       IzeroFEL_pump_shot = resultsPP[Izero].pump
       IzeroFEL_unpump_shot = resultsPP[Izero].unpump
       Fluo1_pump_shot = resultsPP[diode1].pump
       Fluo1_unpump_shot = resultsPP[diode1].unpump
       Fluo2_pump_shot = resultsPP[diode2].pump
       Fluo2_unpump_shot = resultsPP[diode2].unpump
    
       ######################################
       ### filter Diode1 data
       ######################################
    
       Diode1_pump_shot_filter, Diode1_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter = correlation_filter(Fluo1_pump_shot, Fluo1_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, quantile)
       Diode1_pump_shot_filter = Diode1_pump_shot_filter / Izero_pump_filter
       Diode1_unpump_shot_filter = Diode1_unpump_shot_filter / Izero_unpump_filter
       
       Pump_probe_1_shot = Diode1_pump_shot_filter - Diode1_unpump_shot_filter

       ######################################
       ### filter Diode2 data
       ######################################
    
       Diode2_pump_shot_filter, Diode2_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter = correlation_filter(Fluo2_pump_shot, Fluo2_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, quantile)
       Diode2_pump_shot_filter = Diode2_pump_shot_filter / Izero_pump_filter
       Diode2_unpump_shot_filter = Diode2_unpump_shot_filter / Izero_unpump_filter

       Pump_probe_2_shot = Diode2_pump_shot_filter - Diode2_unpump_shot_filter

       ######################################
       ### make dataframes Diode1
       ######################################

       df_pump_1 = pd.DataFrame(Diode1_pump_shot_filter)
       DataFluo1_pump.append(np.nanquantile(df_pump_1, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_unpump_1 = pd.DataFrame(Diode1_unpump_shot_filter)
       DataFluo1_unpump.append(np.nanquantile(df_unpump_1, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_pump_probe_1 = pd.DataFrame(Pump_probe_1_shot)
       Pump_probe1.append(np.nanquantile(df_pump_probe_1, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       ######################################
       ### make dataframes Diode2
       ######################################

       df_pump_2 = pd.DataFrame(Diode2_pump_shot_filter)
       DataFluo2_pump.append(np.nanquantile(df_pump_2, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_unpump_2 = pd.DataFrame(Diode2_unpump_shot_filter)
       DataFluo2_unpump.append(np.nanquantile(df_unpump_2, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_pump_probe_2 = pd.DataFrame(Pump_probe_2_shot)
       Pump_probe2.append(np.nanquantile(df_pump_probe_2, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))
       
       goodshots1.append(len(Pump_probe_1_shot))
       goodshots2.append(len(Pump_probe_2_shot))
       correlation1.append(pearsonr(IzeroFEL_pump_shot,Fluo1_pump_shot)[0])
       correlation2.append(pearsonr(IzeroFEL_pump_shot,Fluo2_pump_shot)[0])
       Izero_pump.append(np.mean(IzeroFEL_pump_shot))
       Izero_unpump.append(np.mean(IzeroFEL_unpump_shot))
	
       print ('Step {} of {}: Processed {}'.format(i+1, len(scan.files), filename))
       print ("correlation Diode1 (all shots) = {}".format(pearsonr(IzeroFEL_pump_shot,Fluo1_pump_shot)[0]))
       print ("correlation Diode2 (all shots) = {}".format(pearsonr(IzeroFEL_pump_shot,Fluo2_pump_shot)[0]))
    
    Adjustable = Adjustable[:np.shape(Pump_probe1)[0]]
    
    DataFluo1_pump = np.asarray(DataFluo1_pump)
    DataFluo1_unpump = np.asarray(DataFluo1_unpump)
    Pump_probe1 = np.asarray(Pump_probe1)

    DataFluo2_pump = np.asarray(DataFluo2_pump)
    DataFluo2_unpump = np.asarray(DataFluo2_unpump)
    Pump_probe2 = np.asarray(Pump_probe2)

    Izero_pump = np.asarray(Izero_pump)
    Izero_unpump = np.asarray(Izero_unpump)
    correlation1 = np.asarray(correlation1)
    correlation2 = np.asarray(correlation2)

    return (DataFluo1_pump, DataFluo1_unpump, Pump_probe1, DataFluo2_pump, DataFluo2_unpump, Pump_probe2, Izero_pump, Izero_unpump, correlation1, correlation2, Adjustable, goodshots1, goodshots2)

######################################

def XAS_delayscan_noTT(scan, diode, Izero, quantile):
    channels_pp = [channel_Events, diode, Izero]
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
    
    Izero_pump = []
    Izero_unpump = []

    DataFluo_pump = []
    DataFluo_unpump = []
    Pump_probe = []

    correlation = []
    goodshots = []

    for i, step in enumerate(scan):
       check_files_and_data(step)
       clear_output(wait=True)
       filename = scan.files[i][0].split('/')[-1].split('.')[0]
       print ('Processing: {}'.format(scan.fname.split('/')[-3]))
       print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
      
       resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)

       IzeroFEL_pump_shot = resultsPP[Izero].pump
       IzeroFEL_unpump_shot = resultsPP[Izero].unpump
       Fluo_pump_shot = resultsPP[diode].pump
       Fluo_unpump_shot = resultsPP[diode].unpump

       ######################################
       ### filter Diode1 data
       ######################################
    
       Diode_pump_shot_filter, Diode_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter = correlation_filter(Fluo_pump_shot, Fluo_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, quantile)
       Diode_pump_shot_filter = Diode_pump_shot_filter / Izero_pump_filter
       Diode_unpump_shot_filter = Diode_unpump_shot_filter / Izero_unpump_filter

       Pump_probe_shot = Diode_pump_shot_filter - Diode_unpump_shot_filter

       ######################################
       ### make dataframes Diode1
       ######################################

       df_pump = pd.DataFrame(Diode_pump_shot_filter)
       DataFluo_pump.append(np.nanquantile(df_pump, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_unpump = pd.DataFrame(Diode_unpump_shot_filter)
       DataFluo_unpump.append(np.nanquantile(df_unpump, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_pump_probe_APD1 = pd.DataFrame(Pump_probe_shot)
       Pump_probe.append(np.nanquantile(df_pump_probe_APD1, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       correlation.append(pearsonr(IzeroFEL_pump_shot,Fluo_pump_shot)[0])
       Izero_pump.append(np.mean(IzeroFEL_pump_shot))
       Izero_unpump.append(np.mean(IzeroFEL_unpump_shot))
	
       print ('Step {} of {}: Processed {}'.format(i+1, len(scan.files), filename))
       print ("correlation Diode (all shots) = {}".format(pearsonr(IzeroFEL_pump_shot,Fluo_pump_shot)[0]))

       goodshots.append(len(Pump_probe_shot))
    
    Delay_mm = Delay_mm[:np.shape(Pump_probe)[0]]
    Delay_fs = Delay_fs[:np.shape(Pump_probe)[0]]
    
    DataFluo_pump = np.asarray(DataFluo_pump)
    DataFluo_unpump = np.asarray(DataFluo_unpump)
    Pump_probe = np.asarray(Pump_probe)

    Izero_pump = np.asarray(Izero_pump)
    Izero_unpump = np.asarray(Izero_unpump)
    correlation = np.asarray(correlation)

    return (DataFluo_pump, DataFluo_unpump, Pump_probe, Izero_pump, Izero_unpump, correlation, Delay_mm, Delay_fs, goodshots)

######################################

def XAS_delayscan_noTT_2diodes(scan, diode1, diode2, Izero, quantile):
    channels_pp = [channel_Events, diode1, diode2, Izero]
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
    
    DataFluo1_pump = []
    DataFluo1_unpump = []
    Pump_probe1 = []

    DataFluo2_pump = []
    DataFluo2_unpump = []
    Pump_probe2 = []

    Izero_pump = []
    Izero_unpump = []

    correlation1 = []
    correlation2 = []

    goodshots1 = []
    goodshots2 = []

    for i, step in enumerate(scan):
       check_files_and_data(step)
       clear_output(wait=True)
       filename = scan.files[i][0].split('/')[-1].split('.')[0]
       print ('Processing: {}'.format(scan.fname.split('/')[-3]))
       print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
      
       resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)

       IzeroFEL_pump_shot = resultsPP[Izero].pump
       IzeroFEL_unpump_shot = resultsPP[Izero].unpump
       Fluo1_pump_shot = resultsPP[diode1].pump
       Fluo1_unpump_shot = resultsPP[diode1].unpump
       Fluo2_pump_shot = resultsPP[diode2].pump
       Fluo2_unpump_shot = resultsPP[diode2].unpump

       ######################################
       ### filter Diode1 data
       ######################################
    
       Diode1_pump_shot_filter, Diode1_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter = correlation_filter(Fluo1_pump_shot, Fluo1_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, quantile)
       Diode1_pump_shot_filter = Diode1_pump_shot_filter / Izero_pump_filter
       Diode1_unpump_shot_filter = Diode1_unpump_shot_filter / Izero_unpump_filter
       
       Pump_probe_1_shot = Diode1_pump_shot_filter - Diode1_unpump_shot_filter

       ######################################
       ### filter Diode2 data
       ######################################
    
       Diode2_pump_shot_filter, Diode2_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter = correlation_filter(Fluo2_pump_shot, Fluo2_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, quantile)
       Diode2_pump_shot_filter = Diode2_pump_shot_filter / Izero_pump_filter
       Diode2_unpump_shot_filter = Diode2_unpump_shot_filter / Izero_unpump_filter

       Pump_probe_2_shot = Diode2_pump_shot_filter - Diode2_unpump_shot_filter

       ######################################
       ### make dataframes Diode1
       ######################################

       df_pump_1 = pd.DataFrame(Diode1_pump_shot_filter)
       DataFluo1_pump.append(np.nanquantile(df_pump_1, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_unpump_1 = pd.DataFrame(Diode1_unpump_shot_filter)
       DataFluo1_unpump.append(np.nanquantile(df_unpump_1, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_pump_probe_1 = pd.DataFrame(Pump_probe_1_shot)
       Pump_probe1.append(np.nanquantile(df_pump_probe_1, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       ######################################
       ### make dataframes Diode2
       ######################################

       df_pump_2 = pd.DataFrame(Diode2_pump_shot_filter)
       DataFluo2_pump.append(np.nanquantile(df_pump_2, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_unpump_2 = pd.DataFrame(Diode2_unpump_shot_filter)
       DataFluo2_unpump.append(np.nanquantile(df_unpump_2, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       df_pump_probe_2 = pd.DataFrame(Pump_probe_2_shot)
       Pump_probe2.append(np.nanquantile(df_pump_probe_2, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))
       
       goodshots1.append(len(Pump_probe_1_shot))
       goodshots2.append(len(Pump_probe_2_shot))
       correlation1.append(pearsonr(IzeroFEL_pump_shot,Fluo1_pump_shot)[0])
       correlation2.append(pearsonr(IzeroFEL_pump_shot,Fluo2_pump_shot)[0])
       Izero_pump.append(np.mean(IzeroFEL_pump_shot))
       Izero_unpump.append(np.mean(IzeroFEL_unpump_shot))
	
       print ('Step {} of {}: Processed {}'.format(i+1, len(scan.files), filename))
       print ("correlation Diode1 (all shots) = {}".format(pearsonr(IzeroFEL_pump_shot,Fluo1_pump_shot)[0]))
       print ("correlation Diode2 (all shots) = {}".format(pearsonr(IzeroFEL_pump_shot,Fluo2_pump_shot)[0]))

    Delay_mm = Delay_mm[:np.shape(Pump_probe1)[0]]
    Delay_fs = Delay_fs[:np.shape(Pump_probe1)[0]]

    DataFluo1_pump = np.asarray(DataFluo1_pump)
    DataFluo1_unpump = np.asarray(DataFluo1_unpump)
    Pump_probe1 = np.asarray(Pump_probe1)

    DataFluo2_pump = np.asarray(DataFluo2_pump)
    DataFluo2_unpump = np.asarray(DataFluo2_unpump)
    Pump_probe2 = np.asarray(Pump_probe2)

    Izero_pump = np.asarray(Izero_pump)
    Izero_unpump = np.asarray(Izero_unpump)
    correlation1 = np.asarray(correlation1)
    correlation2 = np.asarray(correlation2)

    return (DataFluo1_pump, DataFluo1_unpump, Pump_probe1, DataFluo2_pump, DataFluo2_unpump, Pump_probe2, Izero_pump, Izero_unpump, correlation1, correlation2, Delay_mm, Delay_fs, goodshots1, goodshots2)

######################################

TT_PSEN119 = [channel_PSEN119_signal, channel_PSEN119_bkg] 
TT_PSEN124 = [channel_PSEN125_signal]
TT_PSEN126 = [channel_PSEN125_signal, channel_PSEN125_bkg, channel_PSEN125_arrTimes, channel_PSEN125_arrTimesAmp, channel_PSEN125_peaks, channel_PSEN125_edges]

def XAS_delayscan_PSEN(scan, TT, channel_delay_motor, diode, Izero, timezero_mm, quantile, target, calibration, filterTime=2000, filterAmp=0):
    channels_pp = [channel_Events, diode, Izero, channel_delay_motor] + TT
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

    Izero_pump = []
    Izero_unpump = []

    DataFluo_pump = []
    DataFluo_unpump = []
    Pump_probe = []

    Delay_fs_stage = []
    arrTimes_scan = []
    arrTimesAmp_scan = []
    Delays_fs_scan = []
    Pump_probe_scan = []

    correlation = []
    goodshots = []

    for i, step in enumerate(scan):
       check_files_and_data(step)
       clear_output(wait=True)
       filename = scan.files[i][0].split('/')[-1].split('.')[0]
       print ('Processing: {}'.format(scan.fname.split('/')[-3]))
       print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
      
       resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)

       IzeroFEL_pump_shot = resultsPP[Izero].pump
       IzeroFEL_unpump_shot = resultsPP[Izero].unpump
       Fluo_pump_shot = resultsPP[diode].pump
       Fluo_unpump_shot = resultsPP[diode].unpump

       delay_shot = resultsPP[channel_delay_motor].pump
       delay_shot_fs = mm2fs(delay_shot, timezero_mm)
       Delay_fs_stage.append(delay_shot_fs.mean())

       arrTimes, arrTimesAmp, sigtraces, peaktraces = get_arrTimes(resultsPP, step, TT, target, calibration)

       ######################################
       ### filter Diode1 data
       ######################################
    
       Diode_pump_shot_filter, Diode_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter, arrTimes_filter, delay_shot_fs_filter = correlation_filter_TT(Fluo_pump_shot, Fluo_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, arrTimes, delay_shot_fs, quantile)
       Diode_pump_shot_filter = Diode_pump_shot_filter / Izero_pump_filter
       Diode_unpump_shot_filter = Diode_unpump_shot_filter / Izero_unpump_filter

       Pump_probe_shot = Diode_pump_shot_filter - Diode_unpump_shot_filter
        
       Delays_fs_scan.append(delay_shot_fs_filter)
       arrTimes_scan.append(arrTimes_filter)
       Pump_probe_scan.append(Pump_probe_shot)

       df_pump = pd.DataFrame(Diode_pump_shot_filter)
       DataFluo_pump.append(np.nanquantile(df_pump, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))
    
       df_unpump = pd.DataFrame(Diode_unpump_shot_filter)
       DataFluo_unpump.append(np.nanquantile(df_unpump, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))       

       df_pump_probe = pd.DataFrame(Pump_probe_shot)
       Pump_probe.append(np.nanquantile(df_pump_probe, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       print ('Step {} of {}: Processed {}'.format(i+1, len(scan.files), filename))
       print ("correlation Diode (all shots) = {}".format(pearsonr(IzeroFEL_pump_shot,Fluo_pump_shot)[0]))

       goodshots.append(len(Pump_probe_shot))

    Delay_mm = Delay_mm[:np.shape(Pump_probe)[0]]
    Delay_fs = Delay_fs[:np.shape(Pump_probe)[0]]

    Delays_fs_scan = np.asarray(list(itertools.chain.from_iterable(Delays_fs_scan)))
    arrTimes_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_scan)))
    Pump_probe_scan = np.asarray(list(itertools.chain.from_iterable(Pump_probe_scan)))

    Delays_fs_scan = np.asarray(Delays_fs_scan)
    arrTimes_scan = np.asarray(arrTimes_scan)
    Pump_probe_scan = np.asarray(Pump_probe_scan)

    DataFluo_pump = np.asarray(DataFluo_pump)
    DataFluo_unpump = np.asarray(DataFluo_unpump)
    Pump_probe = np.asarray(Pump_probe)
    Izero_pump = np.asarray(Izero_pump)
    Izero_unpump = np.asarray(Izero_unpump)
    correlation = np.asarray(correlation)

    Delays_corr_scan = Delays_fs_scan + arrTimes_scan
    
    return (Delays_fs_scan, Delays_corr_scan, DataFluo_pump, DataFluo_unpump, Pump_probe, Pump_probe_scan, Izero_pump, Izero_unpump, correlation, Delay_mm, Delay_fs, goodshots)
    
######################################

def XAS_delayscan_PSEN_bs(scan, TT, channel_delay_motor, diode, Izero, timezero_mm, quantile, filterTime=2000, filterAmp=0):
    channels_pp = [channel_Events, diode, Izero, channel_delay_motor] + TT
    channels_all = channels_pp

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

    Izero_pump = []
    Izero_unpump = []

    DataFluo_pump = []
    DataFluo_unpump = []
    Pump_probe = []

    Delay_fs_stage = []
    arrTimes_scan = []
    arrTimesAmp_scan = []
    Delays_fs_scan = []
    Pump_probe_scan = []

    correlation = []
    goodshots = []

    for i, step in enumerate(scan):
       check_files_and_data(step)
       clear_output(wait=True)
       filename = scan.files[i][0].split('/')[-1].split('.')[0]
       print ('Processing: {}'.format(scan.fname.split('/')[-3]))
       print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
      
       resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)

       IzeroFEL_pump_shot = resultsPP[Izero].pump
       IzeroFEL_unpump_shot = resultsPP[Izero].unpump
       Fluo_pump_shot = resultsPP[diode].pump
       Fluo_unpump_shot = resultsPP[diode].unpump

       delay_shot = resultsPP[channel_delay_motor].pump
       delay_shot_fs = mm2fs(delay_shot, timezero_mm)
       Delay_fs_stage.append(delay_shot_fs.mean())

       arrTimes = resultsPP[channel_PSEN125_arrTimes].pump
       arrTimesAmp = resultsPP[channel_PSEN125_arrTimesAmp].pump
       sigtraces = resultsPP[channel_PSEN125_edges].pump
       peaktraces = resultsPP[channel_PSEN125_peaks].pump

       #arrTimes, arrTimesAmp, sigtraces, peaktraces = get_arrTimes(resultsPP, step, TT, target, calibration)

       ######################################
       ### filter Diode1 data
       ######################################
    
       Diode_pump_shot_filter, Diode_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter, arrTimes_filter, delay_shot_fs_filter = correlation_filter_TT(Fluo_pump_shot, Fluo_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, arrTimes, delay_shot_fs, quantile)
       Diode_pump_shot_filter = Diode_pump_shot_filter / Izero_pump_filter
       Diode_unpump_shot_filter = Diode_unpump_shot_filter / Izero_unpump_filter

       Pump_probe_shot = Diode_pump_shot_filter - Diode_unpump_shot_filter
        
       Delays_fs_scan.append(delay_shot_fs_filter)
       arrTimes_scan.append(arrTimes_filter)
       Pump_probe_scan.append(Pump_probe_shot)

       df_pump = pd.DataFrame(Diode_pump_shot_filter)
       DataFluo_pump.append(np.nanquantile(df_pump, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))
    
       df_unpump = pd.DataFrame(Diode_unpump_shot_filter)
       DataFluo_unpump.append(np.nanquantile(df_unpump, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))       

       df_pump_probe = pd.DataFrame(Pump_probe_shot)
       Pump_probe.append(np.nanquantile(df_pump_probe, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

       print ('Step {} of {}: Processed {}'.format(i+1, len(scan.files), filename))
       print ("correlation Diode (all shots) = {}".format(pearsonr(IzeroFEL_pump_shot,Fluo_pump_shot)[0]))

       goodshots.append(len(Pump_probe_shot))

    Delay_mm = Delay_mm[:np.shape(Pump_probe)[0]]
    Delay_fs = Delay_fs[:np.shape(Pump_probe)[0]]

    Delays_fs_scan = np.asarray(list(itertools.chain.from_iterable(Delays_fs_scan)))
    arrTimes_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_scan)))
    Pump_probe_scan = np.asarray(list(itertools.chain.from_iterable(Pump_probe_scan)))

    Delays_fs_scan = np.asarray(Delays_fs_scan)
    arrTimes_scan = np.asarray(arrTimes_scan)
    Pump_probe_scan = np.asarray(Pump_probe_scan)

    DataFluo_pump = np.asarray(DataFluo_pump)
    DataFluo_unpump = np.asarray(DataFluo_unpump)
    Pump_probe = np.asarray(Pump_probe)
    Izero_pump = np.asarray(Izero_pump)
    Izero_unpump = np.asarray(Izero_unpump)
    correlation = np.asarray(correlation)

    Delays_corr_scan = Delays_fs_scan + arrTimes_scan
    
    return (Delays_fs_scan, Delays_corr_scan, DataFluo_pump, DataFluo_unpump, Pump_probe, Pump_probe_scan, Izero_pump, Izero_unpump, correlation, Delay_mm, Delay_fs, goodshots)

######################################

def save_run_array_XANES(reducedir, run_name, En, D1p, D1u, PP1, gs1):
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "Energy_eV" : En}

    np.save(reducedir+run_name+'/run_array', run_array)

######################################

def save_data_XANES(reducedir, run_name, En, D1p, D1u, PP1, gs1):
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "Energy_eV" : En}
    
    np.save(reducedir+run_name+'/XANES_energy_eV.npy', En)
    np.save(reducedir+run_name+'/XANES_goodshots1.npy', gs1)

    np.save(reducedir+run_name+'/XANES_DataDiode1_pump', D1p)
    np.save(reducedir+run_name+'/XANES_DataDiode1_unpump', D1u)
    np.save(reducedir+run_name+'/XANES_Pump_probe_Diode1', PP1)

    np.save(reducedir+run_name+'/run_array', run_array)

######################################

def save_run_array_timescans(reducedir, run_name, delaymm, delayfs, D1p, D1u, PP1, gs1):
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "Delay_mm" : delaymm,
                                    "Delay_fs" : delayfs}

    np.save(reducedir+run_name+'/run_array', run_array)

######################################

def save_data_timescans(reducedir, run_name, delaymm, delayfs, D1p, D1u, PP1, gs1):
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "Delay_mm" : delaymm,
                                    "Delay_fs" : delayfs}
    
    np.save(reducedir+run_name+'/timescan_Delay_mm.npy', delaymm)
    np.save(reducedir+run_name+'/timescan_Delay_fs.npy', delayfs)
    np.save(reducedir+run_name+'/timescan_goodshots1.npy', gs1)

    np.save(reducedir+run_name+'/timescan_DataDiode1_pump', D1p)
    np.save(reducedir+run_name+'/timescan_DataDiode1_unpump', D1u)
    np.save(reducedir+run_name+'/timescan_Pump_probe_Diode1', PP1)

    np.save(reducedir+run_name+'/run_array', run_array)

######################################

def save_run_array_XANES_2diodes(reducedir, run_name, En, D1p, D1u, PP1, gs1, D2p, D2u, PP2, gs2):
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "DataDiode2_pump": D2p, 
                                    "DataDiode2_unpump" : D2u, 
                                    "Pump_probe_Diode2" : PP2, 
                                    "goodshots2" : gs2,
                                    "Energy_eV" : En}
   
    np.save(reducedir+run_name+'/run_array', run_array)

######################################

def save_data_XANES_2diodes(reducedir, run_name, En, D1p, D1u, PP1, gs1, D2p, D2u, PP2, gs2):
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "DataDiode2_pump": D2p, 
                                    "DataDiode2_unpump" : D2u, 
                                    "Pump_probe_Diode2" : PP2, 
                                    "goodshots2" : gs2,
                                    "Energy_eV" : En}
    
    np.save(reducedir+run_name+'/XANES_energy_eV.npy', En)
    np.save(reducedir+run_name+'/XANES_goodshots1.npy', gs1)
    np.save(reducedir+run_name+'/XANES_goodshots2.npy', gs2)

    np.save(reducedir+run_name+'/XANES_DataDiode1_pump', D1p)
    np.save(reducedir+run_name+'/XANES_DataDiode1_unpump', D1u)
    np.save(reducedir+run_name+'/XANES_Pump_probe_Diode1', PP1)

    np.save(reducedir+run_name+'/XANES_DataDiode2_pump', D2p)
    np.save(reducedir+run_name+'/XANES_DataDiode2_unpump', D2u)
    np.save(reducedir+run_name+'/XANES_Pump_probe_Diode2', PP2)

    np.save(reducedir+run_name+'/run_array', run_array)

######################################

def save_run_array_timescans_2diodes(reducedir, run_name, delaymm, delayfs, D1p, D1u, PP1, gs1):
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "DataDiode2_pump": D2p, 
                                    "DataDiode2_unpump" : D2u, 
                                    "Pump_probe_Diode2" : PP2, 
                                    "goodshots2" : gs2,
                                    "Delay_mm" : delaymm,
                                    "Delay_fs" : delayfs}
   
    np.save(reducedir+run_name+'/run_array', run_array)

######################################


def save_data_timescans_2diodes(reducedir, run_name, delaymm, delayfs, D1p, D1u, PP1, gs1, D2p, D2u, PP2, gs2):
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "DataDiode2_pump": D2p, 
                                    "DataDiode2_unpump" : D2u, 
                                    "Pump_probe_Diode2" : PP2, 
                                    "goodshots2" : gs2,
                                    "Delay_mm" : delaymm,
                                    "Delay_fs" : delayfs}
    
    np.save(reducedir+run_name+'/timescan_Delay_mm.npy', delaymm)
    np.save(reducedir+run_name+'/timescan_Delay_fs.npy', delayfs)
    np.save(reducedir+run_name+'/timescan_goodshots1.npy', gs1)
    np.save(reducedir+run_name+'/timescan_goodshots2.npy', gs2)

    np.save(reducedir+run_name+'/timescan_DataDiode1_pump', D1p)
    np.save(reducedir+run_name+'/timescan_DataDiode1_unpump', D1u)
    np.save(reducedir+run_name+'/timescan_Pump_probe_Diode1', PP1)

    np.save(reducedir+run_name+'/timescan_DataDiode2_pump', D2p)
    np.save(reducedir+run_name+'/timescan_DataDiode2_unpump', D2u)
    np.save(reducedir+run_name+'/timescan_Pump_probe_Diode2', PP2)

    np.save(reducedir+run_name+'/run_array', run_array)




