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

    #print ('{} shots out of {} survived'.format(np.shape(FluoData_filter), np.shape(FluoData)))

    return (FluoData_filter, Izero_filter)

######################################

def make_correlation_filter(arrayX, arrayY, quantile):

    m,b = np.polyfit(arrayX, arrayY, 1)

    line = m*arrayX+b
    line2 = (arrayY-b)/m

    projection_X = arrayY - line
    projection_Y = arrayX - line2

    qnt_low_Y  = np.nanquantile(projection_Y, 0.5 - quantile/2)
    qnt_high_Y = np.nanquantile(projection_Y, 0.5 + quantile/2)
    qnt_low_X  = np.nanquantile(projection_X, 0.5 - quantile/2)
    qnt_high_X = np.nanquantile(projection_X, 0.5 + quantile/2)
    
    condition_low_Y  = projection_Y > qnt_low_Y
    condition_high_Y = projection_Y < qnt_high_Y
    condition_low_X  = projection_X > qnt_low_X
    condition_high_X = projection_X < qnt_high_X

    correlation_filter = condition_low_Y & condition_high_Y & condition_low_X & condition_high_X

    return (correlation_filter)

######################################

def make_correlation_filter2(arrayX, arrayY, quantile):

    m,b = np.polyfit(arrayX, arrayY, 1)

    line = m*arrayX+b

    #projection_X = arrayY - line
    projection_X = (arrayY-line)/np.sqrt(1+m**2)

    qnt_low_X  = np.nanquantile(projection_X, 0.5 - quantile/2)
    qnt_high_X = np.nanquantile(projection_X, 0.5 + quantile/2)
    
    condition_low_X  = projection_X > qnt_low_X
    condition_high_X = projection_X < qnt_high_X

    correlation_filter = condition_low_X & condition_high_X

    return (correlation_filter, line)

######################################

def apply_filter(c_filter, arrays, printflag=False, suffix="filter", globals_dict=None):
    if globals_dict is None:
        print("please pass globals() as globals_dict")
        return
    new_ones = {}
    for k, v in globals_dict.items():
        if not k in arrays:
            continue
        n = k + "_" + suffix
        new_ones[n] = v[c_filter]
    if printflag:
        print("creating the following variables:")
        for n in sorted(new_ones):
            print("-", n)
    globals_dict.update(new_ones)
    return new_ones

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

    #print ('{} shots out of {} survived'.format(np.shape(FluoData_pump_filter), np.shape(FluoData_pump)))

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

    #print ('{} shots out of {} survived'.format(np.shape(FluoData_pump_filter), np.shape(FluoData_pump)))

    return (FluoData_pump_filter, FluoData_unpump_filter, Izero_pump_filter, Izero_unpump_filter, arrTimes_filter, delay_fs_filter)

######################################

def correlation_filter2_pp(arrayXp, arrayYp, arrayXu, arrayYu, arrTimes, delay_fs, quantile):
    
    mp,bp = np.polyfit(arrayXp, arrayYp, 1)
    mu,bu = np.polyfit(arrayXu, arrayYu, 1)
    
    linep = mp*arrayXp+bp
    lineu = mu*arrayXu+bu

    projection_Xp = (arrayYp-linep)/np.sqrt(1+mp**2)
    projection_Xu = (arrayYu-lineu)/np.sqrt(1+mu**2)

    qnt_low_Xp  = np.nanquantile(projection_Xp, 0.5 - quantile/2)
    qnt_high_Xp = np.nanquantile(projection_Xp, 0.5 + quantile/2)
    qnt_low_Xu  = np.nanquantile(projection_Xu, 0.5 - quantile/2)
    qnt_high_Xu = np.nanquantile(projection_Xu, 0.5 + quantile/2)
    
    condition_low_Xp  = projection_Xp > qnt_low_Xp
    condition_high_Xp = projection_Xp < qnt_high_Xp
    condition_low_Xu  = projection_Xu > qnt_low_Xu
    condition_high_Xu = projection_Xu < qnt_high_Xu
    
    correlation_filter = condition_low_Xu & condition_high_Xu & condition_low_Xp & condition_high_Xp 
    
    arrayXp_filter = arrayXp[correlation_filter]
    arrayYp_filter = arrayYp[correlation_filter]
    arrayXu_filter = arrayXu[correlation_filter]
    arrayYu_filter = arrayYu[correlation_filter]
    arrTimes_filter = arrTimes[correlation_filter]
    delay_fs_filter = delay_fs[correlation_filter]
    
    return (arrayXp_filter, arrayYp_filter, arrayXu_filter, arrayYu_filter, arrTimes_filter, delay_fs_filter)

######################################

def Get_correlation_from_scan_static(scan, index, diode, Izero, quantile):
    channels = [channel_Events, diode, Izero]

    data = scan[index]
    results,_ = load_data_compact(channels, data)
    data.close()
    
    clear_output(wait=True)
    
    Fluo = results[diode]
    IzeroFEL = results[Izero]
    
    # Fluo_filter, Izero_filter = correlation_filter_static(Fluo, IzeroFEL, quantile)
    correlation_filter = make_correlation_filter(Fluo, IzeroFEL, quantile)
    Fluo_filter  = Fluo[correlation_filter]
    Izero_filter = IzeroFEL[correlation_filter]
    
    return (Fluo, IzeroFEL, Fluo_filter, Izero_filter)
    
######################################

def Get_correlation_from_scan2(scan, index, diode, Izero, quantile):
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

    correlation_pump   = make_correlation_filter2(Izero_pump, Fluo_pump, quantile)
    correlation_umpump = make_correlation_filter2(Izero_unpump, Fluo_unpump, quantile)

    correlation_filter = correlation_pump & correlation_umpump

    return (Fluo_pump, Fluo_unpump, Izero_pump, Izero_unpump, correlation_filter)

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

    scanvar = scan.readbacks

    DataFluo = []
    IzeroFEL = []
 
    correlation = []

    for i, step in enumerate(scan):
        check_files_and_data(step)
        check = get_filesize_diff(step)     
        if check:
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
            
            correlation_filter = make_correlation_filter(Fluo_shot, IzeroFEL_shot, quantile)
            Diode_shot_filter  = Fluo_shot[correlation_filter]
            Izero_filter        = IzeroFEL_shot[correlation_filter]
            #Diode1_shot_filter, Izero_filter = correlation_filter_static(Fluo_shot1, IzeroFEL_shot, quantile)
            Diode_shot_filter  = Diode_shot_filter / Izero_filter
            
            print ('{} shots out of {} survived'.format(len(Diode_shot_filter), len(Fluo_shot)))

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
    
    scanvar = scanvar[:np.shape(DataFluo)[0]]
    
    DataFluo = np.asarray(DataFluo)
    IzeroFEL = np.asarray(Izero)
    correlation = np.asarray(correlation)
    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(scanvar), len(scan)))

    return (DataFluo, IzeroFEL, correlation, scanvar)

######################################

def XAS_scan_2diodes_static(scan, diode1, diode2, Izero, quantile):
    channels = [channel_Events, diode1, diode2, Izero]

    scanvar = scan.readbacks

    DataFluo1 = []
    DataFluo2 = []
    IzeroFEL = []
 
    correlation1 = []
    correlation2 = []

    for i, step in enumerate(scan):
        check_files_and_data(step)
        check = get_filesize_diff(step)     
        if check:
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
            
            # correlation_filter1 = make_correlation_filter(Fluo_shot1, IzeroFEL_shot, quantile)
            # Diode1_shot_filter  = Fluo_shot1[correlation_filter1]
            # Izero_filter        = IzeroFEL_shot[correlation_filter1]
            Diode1_shot_filter, Izero_filter = correlation_filter_static(Fluo_shot1, IzeroFEL_shot, quantile)
            Diode1_shot_filter  = Diode1_shot_filter / Izero_filter
            
            print ('{} shots out of {} survived'.format(len(Diode1_shot_filter), len(Fluo_shot1)))
            
            ######################################
            ### filter Diode2 data
            ######################################
            
            # correlation_filter2 = make_correlation_filter(Fluo_shot2, IzeroFEL_shot, quantile)
            # Diode2_shot_filter  = Fluo_shot2[correlation_filter2]
            # Izero_filter        = IzeroFEL_shot[correlation_filter2]
            Diode2_shot_filter, Izero_filter = correlation_filter_static(Fluo_shot2, IzeroFEL_shot, quantile)
            Diode2_shot_filter  = Diode2_shot_filter / Izero_filter
            
            print ('{} shots out of {} survived'.format(len(Diode2_shot_filter), len(Fluo_shot2)))

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
    
    scanvar = scanvar[:np.shape(DataFluo1)[0]]
    
    DataFluo1 = np.asarray(DataFluo1)
    DataFluo2 = np.asarray(DataFluo2)
    IzeroFEL = np.asarray(Izero)
    correlation1 = np.asarray(correlation1)
    correlation2 = np.asarray(correlation2)
    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(scanvar), len(scan)))

    return (DataFluo1, DataFluo2, IzeroFEL, correlation1, correlation2, scanvar)

######################################

def XAS_scanPP_1diode_noTT(scan, diode, Izero, quantile):
    channels_pp = [channel_Events, diode, Izero]
    channels_all = channels_pp

    scanvar = scan.readbacks
   
    Izero_pump = []
    Izero_unpump = []

    DataFluo_pump = []
    DataFluo_unpump = []
    Pump_probe = []

    correlation = []
    goodshots = []

    for i, step in enumerate(scan):
        check_files_and_data(step)
        check = get_filesize_diff(step)     
        if check:
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
            
            # correlation_filter_pump1   = make_correlation_filter(Fluo_pump_shot, IzeroFEL_pump_shot, quantile)
            # correlation_filter_unpump1 = make_correlation_filter(Fluo_unpump_shot, IzeroFEL_unpump_shot, quantile)
            # correlation_filter1 = correlation_filter_pump1 & correlation_filter_unpump1
            # Diode_pump_shot_filter   = Fluo_pump_shot[correlation_filter1]
            # Diode_unpump_shot_filter = Fluo_unpump_shot[correlation_filter1]
            # Izero_pump_filter        = IzeroFEL_pump_shot[correlation_filter1]
            # Izero_unpump_filter      = IzeroFEL_unpump_shot[correlation_filter1]
            
            Diode_pump_shot_filter, Diode_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter = correlation_filter(Fluo_pump_shot, Fluo_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, quantile)

            Diode_pump_shot_filter = Diode_pump_shot_filter / Izero_pump_filter
            Diode_unpump_shot_filter = Diode_unpump_shot_filter / Izero_unpump_filter            
            
            print ('{} shots out of {} survived'.format(len(Diode_pump_shot_filter), len(Fluo_pump_shot)))
            
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
    
    scanvar = scanvar[:np.shape(Pump_probe)[0]]

    DataFluo_pump = np.asarray(DataFluo_pump)
    DataFluo_unpump = np.asarray(DataFluo_unpump)
    Pump_probe = np.asarray(Pump_probe)

    Izero_pump = np.asarray(Izero_pump)
    Izero_unpump = np.asarray(Izero_unpump)
    correlation = np.asarray(correlation)
    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(scanvar), len(scan)))

    return (DataFluo_pump, DataFluo_unpump, Pump_probe, Izero_pump, Izero_unpump, correlation, scanvar, goodshots)

######################################

def XAS_scanPP_2diodes_noTT(scan, diode1, diode2, Izero, quantile):
    channels_pp = [channel_Events, diode1, diode2, Izero]
    channels_all = channels_pp

    scanvar = scan.readbacks
    
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
        check = get_filesize_diff(step)
        if check:
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
            
            # correlation_filter_pump1   = make_correlation_filter(Fluo1_pump_shot, IzeroFEL_pump_shot, quantile)
            # correlation_filter_unpump1 = make_correlation_filter(Fluo1_unpump_shot, IzeroFEL_unpump_shot, quantile)
            # correlation_filter1 = correlation_filter_pump1 & correlation_filter_unpump1
            # Diode1_pump_shot_filter   = Fluo1_pump_shot[correlation_filter1]
            # Diode1_unpump_shot_filter = Fluo1_unpump_shot[correlation_filter1]
            # Izero_pump_filter         = IzeroFEL_pump_shot[correlation_filter1]
            # Izero_unpump_filter       = IzeroFEL_unpump_shot[correlation_filter1]
            
            Diode1_pump_shot_filter, Diode1_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter = correlation_filter(Fluo1_pump_shot, Fluo1_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, quantile)
    
            Diode1_pump_shot_filter = Diode1_pump_shot_filter / Izero_pump_filter
            Diode1_unpump_shot_filter = Diode1_unpump_shot_filter / Izero_unpump_filter            
            
            print ('{} shots out of {} survived'.format(len(Diode1_pump_shot_filter), len(Fluo1_pump_shot)))

            Pump_probe_1_shot = Diode1_pump_shot_filter - Diode1_unpump_shot_filter

            ######################################
            ### filter Diode2 data
            ######################################
            
            # correlation_filter_pump2   = make_correlation_filter(Fluo2_pump_shot, IzeroFEL_pump_shot, quantile)
            # correlation_filter_unpump2 = make_correlation_filter(Fluo2_unpump_shot, IzeroFEL_unpump_shot, quantile)
            # correlation_filter2 = correlation_filter_pump2 & correlation_filter_unpump2
            # Diode2_pump_shot_filter   = Fluo1_pump_shot[correlation_filter2]
            # Diode2_unpump_shot_filter = Fluo1_unpump_shot[correlation_filter2]
            # Izero_pump_filter         = IzeroFEL_pump_shot[correlation_filter2]
            # Izero_unpump_filter       = IzeroFEL_unpump_shot[correlation_filter2]
            
            Diode2_pump_shot_filter, Diode2_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter = correlation_filter(Fluo2_pump_shot, Fluo2_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, quantile)

            Diode2_pump_shot_filter = Diode2_pump_shot_filter / Izero_pump_filter
            Diode2_unpump_shot_filter = Diode2_unpump_shot_filter / Izero_unpump_filter            
            
            print ('{} shots out of {} survived'.format(len(Diode2_pump_shot_filter), len(Fluo2_pump_shot)))

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

    scanvar = scanvar[:np.shape(Pump_probe1)[0]]

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
    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(scanvar), len(scan)))

    return (DataFluo1_pump, DataFluo1_unpump, Pump_probe1, DataFluo2_pump, DataFluo2_unpump, Pump_probe2, Izero_pump, Izero_unpump, correlation1, correlation2, scanvar, goodshots1, goodshots2)

######################################

TT_PSEN119 = [channel_PSEN119_signal, channel_PSEN119_bkg] 
TT_PSEN124 = [channel_PSEN124_signal, channel_PSEN124_bkg, channel_PSEN124_arrTimes, channel_PSEN124_arrTimesAmp, channel_PSEN124_peaks, channel_PSEN124_edges]
TT_PSEN126 = [channel_PSEN126_signal, channel_PSEN126_bkg, channel_PSEN126_arrTimes, channel_PSEN126_arrTimesAmp, channel_PSEN126_peaks, channel_PSEN126_edges]

def XAS_scanPP_PSEN(scan, TT, channel_delay_motor, diode, Izero, timezero_mm, quantile, target, calibration, filterTime=2000, filterAmp=0):
    channels_pp = [channel_Events, diode, Izero, channel_delay_motor] + TT
    channels_all = channels_pp

    scanvar = scan.readbacks

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
        check = get_filesize_diff(step)     
        if check:
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

    scanvar = scanvar[:np.shape(Pump_probe)[0]]

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

    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(scanvar), len(scan)))
    
    return (Delays_fs_scan, Delays_corr_scan, DataFluo_pump, DataFluo_unpump, Pump_probe, Pump_probe_scan, Izero_pump, Izero_unpump, correlation, scanvar, goodshots)
    
######################################

def XAS_scanPP_PSEN_bs(scan, TT, channel_delay_motor, diode, Izero, quantile, timezero_offset=None, filterTime=2000, filterAmp=0):
    channels_pp = [channel_Events, diode, Izero, channel_delay_motor] + TT
    channels_all = channels_pp

    if TT == TT_PSEN124:
        channel_arrTimes = channel_PSEN124_arrTimes
        channel_arrTimesAmp = channel_PSEN124_arrTimesAmp
        channel_edges = channel_PSEN124_edges
        channel_peaks = channel_PSEN124_peaks
    elif TT == TT_PSEN126:
        channel_arrTimes = channel_PSEN126_arrTimes
        channel_arrTimesAmp = channel_PSEN126_arrTimesAmp
        channel_edges = channel_PSEN126_edges
        channel_peaks = channel_PSEN126_peaks
    
    timezero_mm = get_timezero_NBS(scan.fname)
    if timezero_offset is not None:
        timezero_mm = timezero_mm + fs2mm(timezero_offset, 0)
    
    scanvar = scan.readbacks

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
        check = get_filesize_diff(step)     
        if check:
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

            arrTimes = resultsPP[channel_arrTimes].pump
            arrTimesAmp = resultsPP[channel_arrTimesAmp].pump
            sigtraces = resultsPP[channel_edges].pump
            peaktraces = resultsPP[channel_peaks].pump

            #arrTimes, arrTimesAmp, sigtraces, peaktraces = get_arrTimes(resultsPP, step, TT, target, calibration)

            ######################################
            ### filter Diode1 data
            ######################################
            
            # correlation_filter_pump   = make_correlation_filter2(IzeroFEL_pump_shot, Fluo_pump_shot, quantile)
            # correlation_filter_unpump = make_correlation_filter2(IzeroFEL_unpump_shot, Fluo_unpump_shot, quantile)
            # correlation_filter = correlation_filter_pump & correlation_filter_unpump
            # correlation_filter = make_correlation_filter2_pp(IzeroFEL_pump_shot, Fluo_pump_shot, IzeroFEL_unpump_shot, Fluo_unpump_shot, quantile)
            # Diode_pump_shot_filter   = Fluo_pump_shot[correlation_filter]
            # Diode_unpump_shot_filter = Fluo_unpump_shot[correlation_filter]
            # Izero_pump_filter        = IzeroFEL_pump_shot[correlation_filter]
            # Izero_unpump_filter      = IzeroFEL_unpump_shot[correlation_filter]
            # delay_shot_fs_filter     = delay_shot_fs[correlation_filter]
            # arrTimes_filter          = arrTimes[correlation_filter]
        
            # Izero_pump_filter, Diode_pump_shot_filter, Izero_unpump_filter, Diode_unpump_shot_filter, arrTimes_filter, delay_shot_fs_filter = make_correlation_filter2_pp(IzeroFEL_pump_shot, Fluo_pump_shot, IzeroFEL_unpump_shot, Fluo_unpump_shot, arrTimes, delay_shot_fs, quantile)
            
            Diode_pump_shot_filter, Diode_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter, arrTimes_filter, delay_shot_fs_filter = correlation_filter_TT(Fluo_pump_shot, Fluo_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, arrTimes, delay_shot_fs, quantile)
            
            Diode_pump_shot_filter = Diode_pump_shot_filter / Izero_pump_filter
            Diode_unpump_shot_filter = Diode_unpump_shot_filter / Izero_unpump_filter

            Pump_probe_shot = Diode_pump_shot_filter - Diode_unpump_shot_filter
            
            print ('{} shots out of {} survived'.format(len(Diode_pump_shot_filter), len(Fluo_pump_shot)))

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

    scanvar = scanvar[:np.shape(Pump_probe)[0]]

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

    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(scanvar), len(scan)))
    
    return (Delays_fs_scan, Delays_corr_scan, DataFluo_pump, DataFluo_unpump, Pump_probe, Pump_probe_scan, Izero_pump, Izero_unpump, correlation, scanvar, goodshots)

######################################

def XAS_scanPP_2diodes_PSEN_bs(scan, TT, channel_delay_motor, diode1, diode2, Izero, quantile, timezero_offset=None, filterTime=2000, filterAmp=0):
    channels_pp = [channel_Events, diode1, diode2, Izero, channel_delay_motor] + TT
    channels_all = channels_pp

    if TT == TT_PSEN124:
        channel_arrTimes = channel_PSEN124_arrTimes
        channel_arrTimesAmp = channel_PSEN124_arrTimesAmp
        channel_edges = channel_PSEN124_edges
        channel_peaks = channel_PSEN124_peaks
    elif TT == TT_PSEN126:
        channel_arrTimes = channel_PSEN126_arrTimes
        channel_arrTimesAmp = channel_PSEN126_arrTimesAmp
        channel_edges = channel_PSEN126_edges
        channel_peaks = channel_PSEN126_peaks

    timezero_mm = get_timezero_NBS(scan.fname)
    if timezero_offset is not None:
        timezero_mm = timezero_mm + fs2mm(timezero_offset, 0)
	
    scanvar = scan.readbacks

    Izero_pump = []
    Izero_unpump = []

    DataFluo_pump = []
    DataFluo_unpump = []
    Pump_probe = []

    DataFluo2_pump = []
    DataFluo2_unpump = []
    Pump_probe2 = []

    Delay_fs_stage = []
    arrTimes_scan = []
    arrTimesAmp_scan = []
    Delays_fs_scan = []
    Pump_probe_scan = []

    arrTimes_scan2 = []
    arrTimesAmp_scan2 = []
    Delays_fs_scan2 = []
    Pump_probe_scan2 = []

    correlation = []
    correlation2 = []
    goodshots = []
    goodshots2 = []

    for i, step in enumerate(scan):
        check_files_and_data(step)
        check = get_filesize_diff(step)     
        if check:
            clear_output(wait=True)
            filename = scan.files[i][0].split('/')[-1].split('.')[0]
            print ('Processing: {}'.format(scan.fname.split('/')[-3]))
            print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))

            resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)

            IzeroFEL_pump_shot = resultsPP[Izero].pump
            IzeroFEL_unpump_shot = resultsPP[Izero].unpump
            Fluo_pump_shot = resultsPP[diode1].pump
            Fluo_unpump_shot = resultsPP[diode1].unpump
            Fluo2_pump_shot = resultsPP[diode2].pump
            Fluo2_unpump_shot = resultsPP[diode2].unpump

            delay_shot = resultsPP[channel_delay_motor].pump
            delay_shot_fs = mm2fs(delay_shot, timezero_mm)
            Delay_fs_stage.append(delay_shot_fs.mean())

            arrTimes = resultsPP[channel_arrTimes].pump
            arrTimesAmp = resultsPP[channel_arrTimesAmp].pump
            sigtraces = resultsPP[channel_edges].pump
            peaktraces = resultsPP[channel_peaks].pump

            #arrTimes, arrTimesAmp, sigtraces, peaktraces = get_arrTimes(resultsPP, step, TT, target, calibration)

            ######################################
            ### filter Diode1 data
            ######################################
            
            # correlation_filter_pump   = make_correlation_filter2(IzeroFEL_pump_shot, Fluo_pump_shot, quantile)
            # correlation_filter_unpump = make_correlation_filter2(IzeroFEL_unpump_shot, Fluo_unpump_shot, quantile)
            # correlation_filter = correlation_filter_pump & correlation_filter_unpump
            # correlation_filter = make_correlation_filter2_pp(IzeroFEL_pump_shot, Fluo_pump_shot, IzeroFEL_unpump_shot, Fluo_unpump_shot, quantile)
            # Diode_pump_shot_filter   = Fluo_pump_shot[correlation_filter]
            # Diode_unpump_shot_filter = Fluo_unpump_shot[correlation_filter]
            # Izero_pump_filter        = IzeroFEL_pump_shot[correlation_filter]
            # Izero_unpump_filter      = IzeroFEL_unpump_shot[correlation_filter]
            # delay_shot_fs_filter     = delay_shot_fs[correlation_filter]
            # arrTimes_filter          = arrTimes[correlation_filter]
        
            # Izero_pump_filter, Diode_pump_shot_filter, Izero_unpump_filter, Diode_unpump_shot_filter, arrTimes_filter, delay_shot_fs_filter = make_correlation_filter2_pp(IzeroFEL_pump_shot, Fluo_pump_shot, IzeroFEL_unpump_shot, Fluo_unpump_shot, arrTimes, delay_shot_fs, quantile)
            
            Diode_pump_shot_filter, Diode_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter, arrTimes_filter, delay_shot_fs_filter = correlation_filter_TT(Fluo_pump_shot, Fluo_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, arrTimes, delay_shot_fs, quantile)
            
            Diode_pump_shot_filter = Diode_pump_shot_filter / Izero_pump_filter
            Diode_unpump_shot_filter = Diode_unpump_shot_filter / Izero_unpump_filter

            Pump_probe_shot = Diode_pump_shot_filter - Diode_unpump_shot_filter
            
            print ('{} shots out of {} survived'.format(len(Diode_pump_shot_filter), len(Fluo_pump_shot)))

            Delays_fs_scan.append(delay_shot_fs_filter)
            arrTimes_scan.append(arrTimes_filter)
            Pump_probe_scan.append(Pump_probe_shot)

            ######################################
            ### filter Diode2 data
            ######################################
            
            # correlation_filter_pump   = make_correlation_filter2(IzeroFEL_pump_shot, Fluo_pump_shot, quantile)
            # correlation_filter_unpump = make_correlation_filter2(IzeroFEL_unpump_shot, Fluo_unpump_shot, quantile)
            # correlation_filter = correlation_filter_pump & correlation_filter_unpump
            # correlation_filter = make_correlation_filter2_pp(IzeroFEL_pump_shot, Fluo_pump_shot, IzeroFEL_unpump_shot, Fluo_unpump_shot, quantile)
            # Diode_pump_shot_filter   = Fluo_pump_shot[correlation_filter]
            # Diode_unpump_shot_filter = Fluo_unpump_shot[correlation_filter]
            # Izero_pump_filter        = IzeroFEL_pump_shot[correlation_filter]
            # Izero_unpump_filter      = IzeroFEL_unpump_shot[correlation_filter]
            # delay_shot_fs_filter     = delay_shot_fs[correlation_filter]
            # arrTimes_filter          = arrTimes[correlation_filter]
        
            # Izero_pump_filter, Diode_pump_shot_filter, Izero_unpump_filter, Diode_unpump_shot_filter, arrTimes_filter, delay_shot_fs_filter = make_correlation_filter2_pp(IzeroFEL_pump_shot, Fluo_pump_shot, IzeroFEL_unpump_shot, Fluo_unpump_shot, arrTimes, delay_shot_fs, quantile)
            
            Diode2_pump_shot_filter, Diode2_unpump_shot_filter, Izero_pump_filter, Izero_unpump_filter, arrTimes_filter2, delay_shot_fs_filter2 = correlation_filter_TT(Fluo2_pump_shot, Fluo2_unpump_shot, IzeroFEL_pump_shot, IzeroFEL_unpump_shot, arrTimes, delay_shot_fs, quantile)
            
            Diode2_pump_shot_filter = Diode2_pump_shot_filter / Izero_pump_filter
            Diode2_unpump_shot_filter = Diode2_unpump_shot_filter / Izero_unpump_filter

            Pump_probe_2_shot = Diode2_pump_shot_filter - Diode2_unpump_shot_filter
            
            print ('{} shots out of {} survived'.format(len(Diode2_pump_shot_filter), len(Fluo2_pump_shot)))

            Delays_fs_scan2.append(delay_shot_fs_filter2)
            arrTimes_scan2.append(arrTimes_filter2)
            Pump_probe_scan2.append(Pump_probe_2_shot)

            ######################################
            ### make dataframes Diode1
            ######################################

            df_pump = pd.DataFrame(Diode_pump_shot_filter)
            DataFluo_pump.append(np.nanquantile(df_pump, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

            df_unpump = pd.DataFrame(Diode_unpump_shot_filter)
            DataFluo_unpump.append(np.nanquantile(df_unpump, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))       

            df_pump_probe = pd.DataFrame(Pump_probe_shot)
            Pump_probe.append(np.nanquantile(df_pump_probe, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

            ######################################
            ### make dataframes Diode2
            ######################################

            df_pump_2 = pd.DataFrame(Diode2_pump_shot_filter)
            DataFluo2_pump.append(np.nanquantile(df_pump_2, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

            df_unpump_2 = pd.DataFrame(Diode2_unpump_shot_filter)
            DataFluo2_unpump.append(np.nanquantile(df_unpump_2, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))       

            df_pump_probe_2 = pd.DataFrame(Pump_probe_2_shot)
            Pump_probe2.append(np.nanquantile(df_pump_probe_2, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))

            goodshots.append(len(Pump_probe_shot))
            goodshots2.append(len(Pump_probe_2_shot))
            correlation.append(pearsonr(IzeroFEL_pump_shot,Fluo_pump_shot)[0])
            correlation2.append(pearsonr(IzeroFEL_pump_shot,Fluo2_pump_shot)[0])

            print ('Step {} of {}: Processed {}'.format(i+1, len(scan.files), filename))
            print ("correlation Diode1 (all shots) = {}".format(pearsonr(IzeroFEL_pump_shot,Fluo_pump_shot)[0]))
            print ("correlation Diode2 (all shots) = {}".format(pearsonr(IzeroFEL_pump_shot,Fluo2_pump_shot)[0]))

    scanvar = scanvar[:np.shape(Pump_probe)[0]]

    Delays_fs_scan = np.asarray(list(itertools.chain.from_iterable(Delays_fs_scan)))
    arrTimes_scan = np.asarray(list(itertools.chain.from_iterable(arrTimes_scan)))
    Pump_probe_scan = np.asarray(list(itertools.chain.from_iterable(Pump_probe_scan)))

    Delays_fs_scan = np.asarray(Delays_fs_scan)
    arrTimes_scan = np.asarray(arrTimes_scan)
    Pump_probe_scan = np.asarray(Pump_probe_scan)

    Delays_fs_scan2 = np.asarray(list(itertools.chain.from_iterable(Delays_fs_scan2)))
    arrTimes_scan2 = np.asarray(list(itertools.chain.from_iterable(arrTimes_scan2)))
    Pump_probe_scan2 = np.asarray(list(itertools.chain.from_iterable(Pump_probe_scan2)))

    Delays_fs_scan2 = np.asarray(Delays_fs_scan2)
    arrTimes_scan2 = np.asarray(arrTimes_scan2)
    Pump_probe_scan2 = np.asarray(Pump_probe_scan2)

    DataFluo_pump = np.asarray(DataFluo_pump)
    DataFluo_unpump = np.asarray(DataFluo_unpump)
    Pump_probe = np.asarray(Pump_probe)
    DataFluo2_pump = np.asarray(DataFluo2_pump)
    DataFluo2_unpump = np.asarray(DataFluo2_unpump)
    Pump_probe2 = np.asarray(Pump_probe2)
    
    Izero_pump = np.asarray(Izero_pump)
    Izero_unpump = np.asarray(Izero_unpump)
    correlation = np.asarray(correlation)
    correlation2 = np.asarray(correlation2)

    Delays_corr_scan  = Delays_fs_scan  + arrTimes_scan
    Delays_corr_scan2 = Delays_fs_scan2 + arrTimes_scan2

    print ('------------------------------')
    print ('Processed {} out of {} files'.format(len(scanvar), len(scan)))
    
    return (Delays_fs_scan, Delays_corr_scan, DataFluo_pump, DataFluo_unpump, Pump_probe, Pump_probe_scan, Delays_fs_scan2, Delays_corr_scan2, DataFluo2_pump, DataFluo2_unpump, Pump_probe2, Pump_probe_scan2, Izero_pump, Izero_unpump, correlation, correlation2, scanvar, goodshots, goodshots2)


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

def save_data_timescans_TT(reducedir, run_name, delaymm, delaystage, delayfs, delaycorr, D1p, D1u, PP1, PPscan, gs1):
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "Pump_probe_scan" : PPscan,
                                    "goodshots1" : gs1,
                                    "Delays_fs_scan" : delayfs,
                                    "Delays_corr_scan" : delaycorr,
                                    "Delay_mm" : delaymm,
                                    "Delay_fs" : delaystage}
    
    np.save(reducedir+run_name+'/timescan_Delay_mm.npy', delaymm)
    np.save(reducedir+run_name+'/timescan_Delay_fs.npy', delaystage)
    np.save(reducedir+run_name+'/timescan_Delay_corr.npy', delaycorr)
    np.save(reducedir+run_name+'/timescan_Delay_fs_scan.npy', delayfs)
    np.save(reducedir+run_name+'/timescan_goodshots1.npy', gs1)

    np.save(reducedir+run_name+'/timescan_DataDiode1_pump', D1p)
    np.save(reducedir+run_name+'/timescan_DataDiode1_unpump', D1u)
    np.save(reducedir+run_name+'/timescan_Pump_probe_Diode1', PP1)
    np.save(reducedir+run_name+'/timescan_Pump_probe_scan', PPscan)

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

def save_run_array_timescans_2diodes(reducedir, run_name, delaymm, delayfs, D1p, D1u, PP1, gs1, D2p, D2u, PP2, gs2):
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
    
################################################

def save_data_timescans_TT_2diodes(reducedir, run_name, delaymm, delaystage, delayfs, delaycorr, D1p, D1u, PP1, gs1, D2p, D2u, PP2, gs2):
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
                                    "Delays_fs_scan" : delayfs,
                                    "Delays_corr_scan" : delaycorr,
                                    "Delay_mm" : delaymm,
                                    "Delay_fs" : delaystage}
    
    np.save(reducedir+run_name+'/timescan_Delay_mm.npy', delaymm)
    np.save(reducedir+run_name+'/timescan_Delay_fs.npy', delayfs)
    np.save(reducedir+run_name+'/timescan_Delays_corr_scan.npy', delaycorr)
    np.save(reducedir+run_name+'/timescan_Delays_fs_scan.npy', delaystage)
    np.save(reducedir+run_name+'/timescan_goodshots1.npy', gs1)
    np.save(reducedir+run_name+'/timescan_goodshots2.npy', gs2)

    np.save(reducedir+run_name+'/timescan_DataDiode1_pump', D1p)
    np.save(reducedir+run_name+'/timescan_DataDiode1_unpump', D1u)
    np.save(reducedir+run_name+'/timescan_Pump_probe_Diode1', PP1)

    np.save(reducedir+run_name+'/timescan_DataDiode2_pump', D2p)
    np.save(reducedir+run_name+'/timescan_DataDiode2_unpump', D2u)
    np.save(reducedir+run_name+'/timescan_Pump_probe_Diode2', PP2)

    np.save(reducedir+run_name+'/run_array', run_array)

################################################
################################################

def save_reduced_data_1diode(reducedir, run_name, scan, D1p, D1u, PP1, gs1, corr1, t0_offset=None):
    t0 = get_timezero_NBS(scan.fname)
    if t0_offset is not None:
        t0 = t0 + fs2mm(t0_offset, 0)
    rdb = scan.readbacks
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "scan_params": scan.parameters,
                                    "timezero_mm": t0,
                                    "readbacks": rdb,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "correlation1": corr1}

    np.save(reducedir+run_name+'/run_array', run_array)

################################################

def save_reduced_data_1diode_TT(reducedir, run_name, scan, D1p, D1u, PP1, gs1, corr1, PPscan, delayfs, delaycorr, t0_offset=None):
    t0 = get_timezero_NBS(scan.fname)
    if t0_offset is not None:
        t0 = t0 + fs2mm(t0_offset, 0)
    rdb = scan.readbacks
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "scan_params": scan.parameters,
                                    "timezero_mm": t0,
                                    "readbacks": rdb,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "correlation1" : corr1,
                                    "Pump_probe_scan" : PPscan,
                                    "Delays_fs_scan" : delayfs,
                                    "Delays_corr_scan" : delaycorr}

    np.save(reducedir+run_name+'/run_array', run_array)

################################################

def save_reduced_data_2diodes(reducedir, run_name, scan, D1p, D1u, PP1, gs1, corr1, D2p, D2u, PP2, gs2, corr2, t0_offset=None):
    t0 = get_timezero_NBS(scan.fname)
    if t0_offset is not None:
        t0 = t0 + fs2mm(t0_offset, 0)
    rdb = scan.readbacks
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "scan_params": scan.parameters,
                                    "timezero_mm": t0,
                                    "readbacks": rdb,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "correlation1" : corr1,
                                    "DataDiode2_pump": D2p, 
                                    "DataDiode2_unpump" : D2u, 
                                    "Pump_probe_Diode2" : PP2, 
                                    "goodshots2" : gs2,
                                    "correlation2": corr2}

    np.save(reducedir+run_name+'/run_array', run_array)

################################################

def save_reduced_data_2diodes_TT(reducedir, run_name, scan, D1p, D1u, PP1, gs1, corr1, PPscan, D2p, D2u, PP2, gs2, corr2, PPscan2, delayfs1, delaycorr1, delayfs2, delaycorr2, t0_offset=None):
    t0 = get_timezero_NBS(scan.fname)
    if t0_offset is not None:
        t0 = t0 + fs2mm(t0_offset, 0)
    rdb = scan.readbacks
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                    "scan_params": scan.parameters,
                                    "timezero_mm": t0,
                                    "readbacks": rdb,
                                    "DataDiode1_pump": D1p, 
                                    "DataDiode1_unpump" : D1u, 
                                    "Pump_probe_Diode1" : PP1, 
                                    "goodshots1" : gs1,
                                    "correlation1" : corr1,
                                    "Pump_probe_scan" : PPscan,
                                    "DataDiode2_pump": D2p, 
                                    "DataDiode2_unpump" : D2u, 
                                    "Pump_probe_Diode2" : PP2, 
                                    "goodshots2" : gs2,
                                    "correlation2" : corr2,
                                    "Pump_probe_scan2" : PPscan2,
                                    "Delays_fs_scan" : delayfs1,
                                    "Delays_corr_scan" : delaycorr1,
                                    "Delays_fs_scan2" : delayfs2,
                                    "Delays_corr_scan2" : delaycorr2}

    np.save(reducedir+run_name+'/run_array', run_array)

################################################

def load_reduced_data(pgroup, loaddir, runlist):
    from collections import defaultdict
    titlestring = pgroup + ' --- ' +str(runlist)

    d = defaultdict(list)
    for run in runlist:
        #data = {}
        file = glob.glob(loaddir + '/*{:04d}*/*run_array*'.format(run))
        run_array = np.load(file[0], allow_pickle=True).item()
        for k, v in run_array.items():
            for key, value in v.items():
                #data[key] = value
                if key == "timezero_mm" or key=="name":
                    d[key].append(value)
                else:
                    d[key].extend(value)
    return d, titlestring

################################################

def LoadTimescansXANES(with_TT, Two_diodes, scan, TT, channel_delay_motor, detector_XAS_1, detector_XAS_2, detector_Izero, quantile_corr, saveflag, reducedir, runname, timezero_offset=None):
    if with_TT:
        if Two_diodes:
            (Delays_fs_scan, Delays_corr_scan, DataDiode_pump, DataDiode_unpump, Pump_probe_Diode, Pump_probe_scan,
             Delays_fs_scan2, Delays_corr_scan2, DataDiode2_pump, DataDiode2_unpump, Pump_probe_Diode2, Pump_probe_scan2,
             Izero_pump, Izero_unpump, correlation, correlation2, readbacks, goodshots, goodshots2) = \
             XAS_scanPP_2diodes_PSEN_bs(scan, TT, channel_delay_motor, detector_XAS_1, detector_XAS_2, detector_Izero, quantile_corr, timezero_offset)
            if saveflag:
                os.makedirs(reducedir+runname, exist_ok=True)
                save_reduced_data_2diodes_TT(reducedir, runname, scan, 
                                             DataDiode_pump, DataDiode_unpump, Pump_probe_Diode, goodshots, correlation, Pump_probe_scan,
                                             DataDiode2_pump, DataDiode2_unpump, Pump_probe_Diode2, goodshots2, correlation2, Pump_probe_scan2,
                                             Delays_fs_scan, Delays_corr_scan, Delays_fs_scan2, Delays_corr_scan2, timezero_offset)
        else:
            (Delays_fs_scan, Delays_corr_scan, DataDiode_pump, DataDiode_unpump, Pump_probe_Diode, Pump_probe_scan, 
             Izero_pump_scan, Izero_unpump_scan, correlation, readbacks, goodshots) = \
             XAS_scanPP_PSEN_bs(scan, TT, channel_delay_motor, detector_XAS_1, detector_Izero, quantile_corr, timezero_offset)
            if saveflag:
                os.makedirs(reducedir+runname, exist_ok=True)
                save_reduced_data_1diode_TT(reducedir, runname, scan, 
                                            DataDiode_pump, DataDiode_unpump, Pump_probe_Diode, goodshots, correlation, Pump_probe_scan, Delays_fs_scan, Delays_corr_scan, timezero_offset)                
    else:
        if Two_diodes:
            (DataDiode1_pump, DataDiode1_unpump, Pump_probe_Diode1, 
             DataDiode2_pump, DataDiode2_unpump, Pump_probe_Diode2, 
             Izero_pump, Izero_unpump, correlation1, correlation2, readbacks, goodshots1, goodshots2) = \
             XAS_scanPP_2diodes_noTT(scan, detector_XAS_1, detector_XAS_2, detector_Izero, quantile_corr)
            if saveflag:
                os.makedirs(reducedir+runname, exist_ok=True)
                save_reduced_data_2diodes(reducedir, runname, scan, 
                                          DataDiode1_pump, DataDiode1_unpump, Pump_probe_Diode1, goodshots1, correlation1,
                                          DataDiode2_pump, DataDiode2_unpump, Pump_probe_Diode2, goodshots2, correlation2, timezero_offset)
                
        else:
            (DataDiode_pump, DataDiode_unpump, Pump_probe_Diode, 
             Izero_pump_scan, Izero_unpump_scan, correlation, readbacks, goodshots) = \
             XAS_scanPP_1diode_noTT(scan, detector_XAS_1, detector_Izero, quantile_corr)
            if saveflag:
                os.makedirs(reducedir+runname, exist_ok=True)
                save_reduced_data_1diode(reducedir, runname, scan, 
                                         DataDiode_pump, DataDiode_unpump, Pump_probe_Diode, goodshots, correlation, timezero_offset)

################################################

def LoadXANES(Two_diodes, scan, detector_XAS_1, detector_XAS_2, detector_Izero, quantile_corr, saveflag, reducedir, runname):
    if Two_diodes: 
        (DataDiode1_pump, DataDiode1_unpump, Pump_probe_Diode1,
         DataDiode2_pump, DataDiode2_unpump, Pump_probe_Diode2, 
         Izero_pump, Izero_unpump, correlation1, correlation2, Energy_eV, goodshots1, goodshots2) = \
        XAS_scanPP_2diodes_noTT(scan, detector_XAS_1, detector_XAS_2, detector_Izero, quantile_corr)
        if saveflag:
            os.makedirs(reducedir+runname, exist_ok=True)
            save_reduced_data_2diodes(reducedir, runname, scan, 
                                      DataDiode1_pump, DataDiode1_unpump, Pump_probe_Diode1, goodshots1, correlation1,
                                      DataDiode2_pump, DataDiode2_unpump, Pump_probe_Diode2, goodshots2, correlation2)
    else:
        (DataDiode1_pump, DataDiode1_unpump, Pump_probe_Diode1,
         Izero_pump_scan, Izero_unpump_scan, correlation1, Energy_eV, goodshots1) = \
        XAS_scanPP_1diode_noTT(scan, detector_XAS_1, detector_Izero, quantile_corr)
        if saveflag:
                os.makedirs(reducedir+runname, exist_ok=True)
                save_reduced_data_1diode(reducedir, runname, scan, 
                                         DataDiode1_pump, DataDiode1_unpump, Pump_probe_Diode1, goodshots1, correlation1)


