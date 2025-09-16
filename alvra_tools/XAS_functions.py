import numpy as np
import json, glob
import os, math
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from IPython.display import clear_output, display
from datetime import datetime
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import itertools

from alvra_tools.load_data import *
from alvra_tools.channels import *
from alvra_tools.utils import *
from alvra_tools.timing_tool import *
from alvra_tools.XAS_functions import *


def Reduce_scan_PP(reducedir, saveflag, jsonlist, TT, motor, diode1, diode2, Izero, shots2average=None):
    
    if TT == TT_PSEN124:
        TT = [channel_PSEN124_arrTimes, channel_PSEN124_arrTimesAmp]
        channel_arrTimes = channel_PSEN124_arrTimes
        channel_arrTimesAmp = channel_PSEN124_arrTimesAmp
    elif TT == TT_PSEN126:
        TT = [channel_PSEN126_arrTimes, channel_PSEN126_arrTimesAmp]
        channel_arrTimes = channel_PSEN126_arrTimes
        channel_arrTimesAmp = channel_PSEN126_arrTimesAmp
    elif TT == None:
        TT = [motor, motor]
        channel_arrTimes = motor
        channel_arrTimesAmp = motor

    channels_pp = [channel_Events, diode1, diode2, Izero, motor, channel_monoEnergy] + TT
    channels_all = channels_pp
    
    from sfdata import SFScanInfo
    
    pump_1, unpump_1, pump_2, unpump_2, pump_1_raw, unpump_1_raw, pump_2_raw, unpump_2_raw, Izero_pump, Izero_unpump, Delays_stage, arrTimes, Delays_corr, energy, energypad, readbacks, corr1, corr2 = ([] for i in range(18))
    
    for jsonfile in jsonlist:
        runname = jsonfile.split('/')[-3]
        scan = SFScanInfo(jsonfile)
        #rbk = np.ravel(scan.readbacks)
        rbk = np.ravel(scan.values)

        unique = np.roll(np.diff(rbk, prepend=1)>0.05, -1)
        unique[-1] = True
        if scan.parameters['Id'] == ['dummy']:
            unique = np.full(len(rbk), True)
        rbk = rbk[unique]

        p1_raw, u1_raw, p2_raw, u2_raw, p1, u1, p2, u2, Ip, Iu, ds, aT, dc, en, en2, c1, c2 = ([] for i in range(17))

        for i, step in enumerate(scan):
            check_files_and_data(step)
            check = get_filesize_diff(step)  
            go = unique[i]

            if check & go:
                clear_output(wait=True)
                filename = scan.files[i][0].split('/')[-1].split('.')[0]
                print (jsonfile)
                print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
    
                resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)

                p1_raw.extend(resultsPP[diode1].pump)
                u1_raw.extend(resultsPP[diode1].unpump)
                p2_raw.extend(resultsPP[diode2].pump)
                u2_raw.extend(resultsPP[diode2].unpump)
                Ip.extend(resultsPP[Izero].pump)
                Iu.extend(resultsPP[Izero].unpump)
                ds.extend(resultsPP[motor].pump)
                aT.extend(resultsPP[channel_arrTimes].pump)
                dc.extend(resultsPP[motor].pump + resultsPP[channel_arrTimes].pump)

                enshot = resultsPP[channel_monoEnergy].pump
                en.extend(enshot)
                #en2 = np.pad(en2, (0,len(enshot)), constant_values=(np.random.normal(rbk[i],0.01,1)))
                en2 = np.pad(en2, (0,len(enshot)), constant_values=(np.nanmean(enshot)))

                pearsonr1 = pearsonr(resultsPP[diode1].unpump,resultsPP[Izero].unpump)[0]
                pearsonr2 = pearsonr(resultsPP[diode2].unpump,resultsPP[Izero].unpump)[0]

                c1.append(pearsonr1)
                c2.append(pearsonr2)

                print ("correlation Diode1 (dark shots) = {}".format(pearsonr1))
                print ("correlation Diode2 (dark shots) = {}".format(pearsonr2))

        u1 = u1_raw/np.nanmean(np.array(u1_raw)[:shots2average])
        p1 = p1_raw/np.nanmean(np.array(u1_raw)[:shots2average])
        u2 = u2_raw/np.nanmean(np.array(u2_raw)[:shots2average])        
        p2 = p2_raw/np.nanmean(np.array(u2_raw)[:shots2average])        

        if saveflag:
            os.makedirs(reducedir+runname, exist_ok=True)
            os.chmod(reducedir+runname, 0o775)
            save_reduced_data_scanPP(reducedir, runname, scan, p1, u1, p2, u2, p1_raw, u1_raw, p2_raw, u2_raw, Ip, Iu, ds, aT, dc, en, en2, rbk, c1, c2)
                
        pump_1.extend(p1)
        unpump_1.extend(u1)
        pump_2.extend(p2)
        unpump_2.extend(u2)
        pump_1_raw.extend(p1_raw)
        unpump_1_raw.extend(u1_raw)
        pump_2_raw.extend(p2_raw)
        unpump_2_raw.extend(u2_raw)
        Izero_pump.extend(Ip)
        Izero_unpump.extend(Iu)
        Delays_stage.extend(ds)
        arrTimes.extend(aT)
        Delays_corr.extend(dc)
        energy.extend(en)
        energypad.extend(en2)
        #readbacks.append(rbk)
        corr1.append(c1)
        corr2.append(c2)

    print ('----------------------------')
    print ('Loaded {} total on/off pairs'.format(len(Delays_corr)))

    return (pump_1, unpump_1, pump_2, unpump_2, pump_1_raw, unpump_1_raw, pump_2_raw, unpump_2_raw, Izero_pump, Izero_unpump, Delays_stage, arrTimes, Delays_corr, energy, energypad, rbk, corr1, corr2)

##################################################################

def Reduce_scan_PP_loop(reducedir, saveflag, jsonlist, TT, motor, diode1, diode2, Izero, shots2average=None):
    
    if TT == TT_PSEN124:
        TT = [channel_PSEN124_arrTimes, channel_PSEN124_arrTimesAmp]
        channel_arrTimes = channel_PSEN124_arrTimes
        channel_arrTimesAmp = channel_PSEN124_arrTimesAmp
    elif TT == TT_PSEN126:
        TT = [channel_PSEN126_arrTimes, channel_PSEN126_arrTimesAmp]
        channel_arrTimes = channel_PSEN126_arrTimes
        channel_arrTimesAmp = channel_PSEN126_arrTimesAmp

    channels_pp = [channel_Events, diode1, diode2, Izero, motor, channel_monoEnergy] + TT
    channels_all = channels_pp
    
    from sfdata import SFScanInfo
    
    pump_1, unpump_1, pump_2, unpump_2, pump_1_raw, unpump_1_raw, pump_2_raw, unpump_2_raw, Izero_pump, Izero_unpump, Delays_stage, arrTimes, Delays_corr, energy, energypad, readbacks, corr1, corr2 = ([] for i in range(18))
    
    for jsonfile in jsonlist:
        runname = jsonfile.split('/')[-3]
        scan = SFScanInfo(jsonfile)
        #rbk = np.ravel(scan.readbacks)
        rbk = np.ravel(scan.values)

        unique = np.roll(np.diff(rbk, prepend=1)>0.05, -1)
        unique[-1] = True
        if scan.parameters['Id'] == ['dummy']:
            unique = np.full(len(rbk), True)
        rbk = rbk[unique]

        p1_raw, u1_raw, p2_raw, u2_raw, p1, u1, p2, u2, Ip, Iu, ds, aT, dc, en, en2, c1, c2 = ([] for i in range(17))

        for i, step in enumerate(scan):
            check_files_and_data(step)
            check = get_filesize_diff(step)  
            go = unique[i]

            if check & go:
                clear_output(wait=True)
                filename = scan.files[i][0].split('/')[-1].split('.')[0]
                print (jsonfile)
                print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
    
                resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_all, step)
                #try:
                p1_raw_acq = resultsPP[diode1].pump
                u1_raw_acq = resultsPP[diode1].unpump
                p2_raw_acq = resultsPP[diode2].pump
                u2_raw_acq = resultsPP[diode2].unpump
                Ip_acq     = resultsPP[Izero].pump
                Iu_acq     = resultsPP[Izero].unpump
                ds_acq     = resultsPP[motor].pump
                aT_acq     = resultsPP[channel_arrTimes].pump
                dc_acq     = resultsPP[motor].pump + resultsPP[channel_arrTimes].pump   

                enshot = resultsPP[channel_monoEnergy].pump                 
                en_acq = enshot                         
                en2 = np.pad(en2, (0,len(enshot)), constant_values=(np.nanmean(enshot)))  

                pearsonr1 = pearsonr(resultsPP[diode1].unpump,resultsPP[Izero].unpump)[0]
                pearsonr2 = pearsonr(resultsPP[diode2].unpump,resultsPP[Izero].unpump)[0]

                c1_acq = [pearsonr1]
                c2_acq = [pearsonr2]

                p1_raw.extend(p1_raw_acq)
                u1_raw.extend(u1_raw_acq)
                p2_raw.extend(p2_raw_acq)
                u2_raw.extend(u2_raw_acq)
                Ip.extend(Ip_acq)
                Iu.extend(Iu_acq)
                ds.extend(ds_acq)
                aT.extend(aT_acq)
                dc.extend(dc_acq)
                
                en.extend(enshot)
                c1.append(pearsonr1)
                c2.append(pearsonr2)

                print ("correlation Diode1 (dark shots) = {}".format(pearsonr1))
                print ("correlation Diode2 (dark shots) = {}".format(pearsonr2))

                u1_acq = u1_raw_acq/np.nanmean(np.array(u1_raw_acq)[:shots2average])
                p1_acq = p1_raw_acq/np.nanmean(np.array(u1_raw_acq)[:shots2average])
                u2_acq = u2_raw_acq/np.nanmean(np.array(u2_raw_acq)[:shots2average])        
                p2_acq = p2_raw_acq/np.nanmean(np.array(u2_raw_acq)[:shots2average])        

                if saveflag:
                    os.makedirs(reducedir+runname+'/'+filename, exist_ok=True)
                    os.chmod(reducedir+runname+'/'+filename, 0o775)
                    save_reduced_data_scanPP(reducedir, runname+'/'+filename, scan, p1_acq, u1_acq, p2_acq, u2_acq, p1_raw_acq, u1_raw_acq, p2_raw_acq, u2_raw_acq, Ip_acq, Iu_acq, ds_acq, aT_acq, dc_acq, en_acq, en2, rbk, c1_acq, c2_acq)
                print ('Saved in: {}'.format(reducedir+runname+'/'+filename+'/'))
                #except:
                #    print ('Error in loading this acquisition, skipped!')
        
        u1 = u1_raw/np.nanmean(np.array(u1_raw)[:shots2average])
        p1 = p1_raw/np.nanmean(np.array(u1_raw)[:shots2average])
        u2 = u2_raw/np.nanmean(np.array(u2_raw)[:shots2average])        
        p2 = p2_raw/np.nanmean(np.array(u2_raw)[:shots2average])   

        if saveflag:
            os.makedirs(reducedir+runname, exist_ok=True)
            os.chmod(reducedir+runname, 0o775)
            save_reduced_data_scanPP(reducedir, runname, scan, p1, u1, p2, u2, p1_raw, u1_raw, p2_raw, u2_raw, Ip, Iu, ds, aT, dc, en, en2, rbk, c1, c2)
            print ('Saved in: {}'.format(reducedir+runname+'/'))

        pump_1.extend(p1)
        unpump_1.extend(u1)
        pump_2.extend(p2)
        unpump_2.extend(u2)
        pump_1_raw.extend(p1_raw)
        unpump_1_raw.extend(u1_raw)
        pump_2_raw.extend(p2_raw)
        unpump_2_raw.extend(u2_raw)
        Izero_pump.extend(Ip)
        Izero_unpump.extend(Iu)
        Delays_stage.extend(ds)
        arrTimes.extend(aT)
        Delays_corr.extend(dc)
        energy.extend(en)
        energypad.extend(en2)
        #readbacks.append(rbk)
        corr1.append(c1)
        corr2.append(c2)

    print ('----------------------------')
    print ('Loaded {} total on/off pairs'.format(len(Delays_corr)))

    return (pump_1, unpump_1, pump_2, unpump_2, pump_1_raw, unpump_1_raw, pump_2_raw, unpump_2_raw, Izero_pump, Izero_unpump, Delays_stage, arrTimes, Delays_corr, energy, energypad, rbk, corr1, corr2)

##################################################################

def Reduce_scan_PP_noPair(reducedir, saveflag, jsonlist, TT, motor, diode1, diode2, det_Izero, shots2average=None):
    
    if TT == TT_PSEN124:
        TT = [channel_PSEN124_arrTimes, channel_PSEN124_arrTimesAmp]
        channel_arrTimes = channel_PSEN124_arrTimes
        channel_arrTimesAmp = channel_PSEN124_arrTimesAmp
    elif TT == TT_PSEN126:
        TT = [channel_PSEN126_arrTimes, channel_PSEN126_arrTimesAmp]
        channel_arrTimes = channel_PSEN126_arrTimes
        channel_arrTimesAmp = channel_PSEN126_arrTimesAmp

    channels_pp = [channel_Events, diode1, diode2, det_Izero, motor, channel_monoEnergy] + TT
    channels_all = channels_pp
    
    from sfdata import SFScanInfo
    
    pump_1, pump_2, pump_1_raw, pump_2_raw, Izero_pump, Delays_stage, arrTimes, Delays_corr, energy, energypad, readbacks, corr1, corr2, lights, darks = ([] for i in range(15))
    
    for jsonfile in jsonlist:
        runname = jsonfile.split('/')[-3]
        scan = SFScanInfo(jsonfile)
        #rbk = np.ravel(scan.readbacks)
        rbk = np.ravel(scan.values)

        unique = np.roll(np.diff(rbk, prepend=1)>0.001, -1)
        unique[-1] = True
        if scan.parameters['Id'] == ['dummy']:
            unique = np.full(len(rbk), True)
        rbk = rbk[unique]

        p1_raw, p2_raw, p1, p2, Ip, ds, aT, dc, en, en2, c1, c2, light, dark = ([] for i in range(14))

        for i, step in enumerate(scan):
            check_files_and_data(step)
            check = get_filesize_diff(step)  
            go = unique[i]

            if check & go:
                clear_output(wait=True)
                filename = scan.files[i][0].split('/')[-1].split('.')[0]
                print (jsonfile)
                print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
    
                resultsPP, results, index_light, index_dark = load_data_compact_pump_probe(channels_pp, channels_all, step)

                light.extend(index_light)
                dark.extend(index_dark)

                p1_raw.extend(results[diode1])
                p2_raw.extend(results[diode2])

                Ip.extend(results[det_Izero])

                ds.extend(results[motor])
                aT.extend(results[channel_arrTimes])
                dc.extend(results[motor] + results[channel_arrTimes])

                enshot = results[channel_monoEnergy]
                en.extend(enshot)
                #en2 = np.pad(en2, (0,len(enshot)), constant_values=(np.random.normal(rbk[i],0.01,1)))
                en2 = np.pad(en2, (0,len(enshot)), constant_values=(np.nanmean(enshot)))

                pearsonr1 = pearsonr(results[diode1],results[det_Izero])[0]
                pearsonr2 = pearsonr(results[diode2],results[det_Izero])[0]

                c1.append(pearsonr1)
                c2.append(pearsonr2)

                print ("correlation Diode1 (ALL shots) = {}".format(pearsonr1))
                print ("correlation Diode2 (ALL shots) = {}".format(pearsonr2))

        p1 = p1_raw/np.nanmean(np.array(p1_raw)[:shots2average])
        p2 = p2_raw/np.nanmean(np.array(p2_raw)[:shots2average])

        if saveflag:
            os.makedirs(reducedir+runname, exist_ok=True)
            save_reduced_data_scanPP_noPair(reducedir, runname, scan, p1, p2, p1_raw, p2_raw, Ip, ds, aT, dc, en, en2, rbk, c1, c2, light, dark)
                
        pump_1.extend(p1)
        pump_2.extend(p2)
        pump_1_raw.extend(p1_raw)
        pump_2_raw.extend(p2_raw)
        Izero_pump.extend(Ip)

        Delays_stage.extend(ds)
        arrTimes.extend(aT)
        Delays_corr.extend(dc)
        energy.extend(en)
        energypad.extend(en2)
        #readbacks.append(rbk)
        corr1.append(c1)
        corr2.append(c2)
        lights.extend(light)
        darks.extend(dark)

    print ('----------------------------')
    print ('Loaded {} total on/off pairs'.format(len(Delays_corr)))

    return (pump_1, pump_2, pump_1_raw, pump_2_raw, Izero_pump, Delays_stage, arrTimes, Delays_corr, energy, energypad, rbk, corr1, corr2, lights, darks)

##################################################################

def Reduce_scan_static(reducedir, saveflag, jsonlist, diode1, diode2, Izero, shots2average=None):
    
    channels = [channel_Events, diode1, diode2, Izero, channel_monoEnergy]

    from sfdata import SFScanInfo
    
    unpump_1, unpump_2, unpump_1_raw, unpump_2_raw, Izero_unpump, energy, energypad, readbacks, corr1, corr2 = ([] for i in range(10))
    
    for jsonfile in jsonlist:
        runname = jsonfile.split('/')[-3]
        scan = SFScanInfo(jsonfile)
        rbk = scan.readbacks

        u1, u2, u1_raw, u2_raw, Iu, en, en2, c1, c2 = ([] for i in range(9))

        for i, step in enumerate(scan):
            check_files_and_data(step)
            check = get_filesize_diff(step)  
            
            if check:
                clear_output(wait=True)
                filename = scan.files[i][0].split('/')[-1].split('.')[0]
                print (jsonfile)
                print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))

                results, _ = load_data_compact(channels, step)

                u1_raw.extend(results[diode1])
                u2_raw.extend(results[diode2])
                Iu.extend(results[Izero])

                enshot = results[channel_monoEnergy]
                en.extend(enshot)
                en2 = np.pad(en2, (0,len(enshot)), constant_values=(np.random.normal(rbk[i],0.01,1)))
                
                pearsonr1 = pearsonr(results[diode1],results[Izero])[0]
                pearsonr2 = pearsonr(results[diode2],results[Izero])[0]

                c1.append(pearsonr1)
                c2.append(pearsonr2)

                print ("correlation Diode1 (dark shots) = {}".format(pearsonr1))
                print ("correlation Diode2 (dark shots) = {}".format(pearsonr2))

        u1 = u1_raw/np.nanmean(np.array(u1_raw)[:shots2average])
        u2 = u2_raw/np.nanmean(np.array(u2_raw)[:shots2average])

        if saveflag:
            os.makedirs(reducedir+runname, exist_ok=True)
            save_reduced_data_scan_static(reducedir, runname, scan, u1, u2, u1_raw, u2_raw, Iu, en, en2, rbk, c1, c2)
                
        unpump_1.extend(u1_raw)
        unpump_2.extend(u2_raw)
        Izero_unpump.extend(Iu)
        energy.extend(en)
        energypad.extend(en2)
        readbacks.append(rbk)
        corr1.append(c1)
        corr2.append(c2)

    print ('----------------------------')
    print ('Loaded {} total shots'.format(len(unpump_1)))

    return (unpump_1, unpump_2, Izero_unpump, energy, energypad, readbacks, corr1, corr2)

######################################

def Rebin_energyscans_static(unpump, Iunpump, energy, readbacks, threshold=0):

    unpump = np.asarray(unpump)
    Iunpump = np.asarray(Iunpump)
    energy = np.asarray(energy)
    readbacks = np.asarray(readbacks)

    ordered = np.argsort(np.asarray(energy))
    peaks, what = find_peaks(np.diff(energy[ordered]))

    unpump = unpump[ordered]
    Iunpump = Iunpump[ordered]

    starts = np.append(0, peaks)
    ends = np.append(peaks, None)

    GS, err_GS, err_GS2 = ([] for i in range(3))
    
    for s, e in zip(starts, ends):
        u  = unpump[s:e]
        I_u = Iunpump[s:e]
        
        thresh   = (I_u > threshold)

        unpump_ebin = u[thresh] / I_u[thresh]

        GS_bin = np.mean(unpump_ebin)
        errGS_bin = np.nanstd(unpump_ebin)/np.sqrt(len(unpump_ebin))
        GS.append(GS_bin)
        err_GS.append(errGS_bin)
        
    print (len(peaks), len(readbacks), np.shape(GS))

    GS = np.reshape(np.array(GS), (len(readbacks),-1)) 
    err_GS = np.reshape(np.array(err_GS), (len(readbacks), -1))

    nscans = np.shape(GS)[1]
    
    err_GS2 = np.nanstd(GS, axis=1)
    GS = np.nanmean(GS, axis=1)
    err_GS = np.sqrt(np.sum(err_GS**2, axis=1))

    return GS, err_GS, err_GS2

######################################

def Rebin_and_filter_energyscans_static(data, quantile, readbacks, threshold=0, n_sigma=1, raw=True):

    for k,v in data.items():
        data[k] = v
    
    unpump_1 = np.asarray(data['unpump_1'])
    if raw: 
        unpump_1 = np.asarray(data['unpump_1_raw'])
    Izero_unpump = np.asarray(data['Izero_unpump'])
    energy = np.asarray(data['energypad'])

    ordered = np.argsort(np.asarray(energy))
    peaks, what = find_peaks(np.diff(energy[ordered]))

    unpump_1 = unpump_1[ordered]
    Izero_unpump = Izero_unpump[ordered]

    starts = np.append(0, peaks)
    ends = np.append(peaks, None)

    GS, err_GS, filtered = ([] for i in range(3))
    
    for s, e in zip(starts, ends):
        unpump  = unpump_1[s:e]
        Izero_u = Izero_unpump[s:e]
        
        ratio_u = unpump/Izero_u
        filterIu = Izero_u > (np.nanmedian(Izero_u) - n_sigma*np.std(Izero_u))
        thresh   = (Izero_u > threshold)
        filtervals = create_corr_condition_u(ratio_u, quantile)

        unpump_1_filter     = unpump[filterIu & thresh & filtervals]
        Izero_unpump_filter = Izero_u[filterIu & thresh & filtervals]
        
        unpump_filter = unpump_1_filter / Izero_unpump_filter
        
        GS_shot = np.mean(unpump_filter)
        errGS_shot = np.nanstd(unpump_filter)/np.sqrt(len(unpump_filter))
        GS.append(GS_shot)
        err_GS.append(errGS_shot)
        
        filtered.extend(filtervals)
    filtered = np.array(filtered)

    print (len(peaks), len(readbacks), np.shape(GS))

    GS = np.reshape(np.array(GS), (len(readbacks),-1)) 
    err_GS = np.reshape(np.array(err_GS), (len(readbacks), -1))

    nscans = np.shape(GS)[1]
    
    err_GS2 = np.nanstd(GS, axis=1)
    GS = np.nanmean(GS, axis=1)
    err_GS = np.sqrt(np.sum(err_GS**2, axis=1))
    #print(sum(filtered)/len(pump_1)*100)

    print ('{} shots out of {} survived'.format(np.sum(filtered), len(unpump_1)))

    res = {'GS': np.array(GS), 'err_GS': np.array(err_GS), 'err_GS2': np.array(err_GS2), 'filtered': filtered}
    
    return res

######################################

def create_corr_condition_u(unpump, quantile):
    
    qnt_low_u  = np.nanquantile(unpump, 0.5 - quantile/2)
    qnt_high_u = np.nanquantile(unpump, 0.5 + quantile/2)

    filtervals_u_l = unpump > qnt_low_u
    filtervals_u_h = unpump < qnt_high_u

    condition = filtervals_u_l & filtervals_u_h
    return condition

######################################

def Rebin_energyscans_PP(pump, unpump, Ipump, Iunpump, scanvar, readbacks, threshold=0):
    
    ordered = np.argsort(np.asarray(scanvar))
    peaks, what = find_peaks(np.diff(scanvar[ordered]))
    
    pump = pump[ordered]
    unpump = unpump[ordered]
    Ipump = Ipump[ordered]
    Iunpump = Iunpump[ordered]

    starts = np.append(0, peaks)
    ends = np.append(peaks, None)

    pp, GS, ES, err_pp, err_GS, err_ES = ([] for i in range(6))
    filtered = []

    for s, e in zip(starts, ends):
        pump_ebin    = pump[s:e]
        unpump_ebin  = unpump[s:e]
        Izero_p_ebin = Ipump[s:e]
        Izero_u_ebin = Iunpump[s:e]
        
        pump_ebin = pump_ebin / Izero_p_ebin
        unpump_ebin = unpump_ebin / Izero_u_ebin

        thresh   = (Izero_p_ebin>threshold) & (Izero_u_ebin>threshold)
        pp_shot = pump_ebin[thresh] - unpump_ebin[thresh]

        #pp_shot = np.log10(pump_ebin[thresh]/unpump_ebin[thresh])
        #pp_shot = np.log10(pump_ebin/unpump_ebin)
        #pp_shot = pump_ebin - unpump_ebin

        GS.append(np.nanmean(unpump_ebin))
        ES.append(np.nanmean(pump_ebin))
        pp.append(np.nanmean(pp_shot))
        err_GS.append(np.nanstd(unpump_ebin)/np.sqrt(len(unpump_ebin)))
        err_ES.append(np.nanstd(pump_ebin)/np.sqrt(len(pump_ebin)))
        err_pp.append(np.nanstd(pp_shot)/np.sqrt(len(pp_shot)))

    return np.array(pp), np.array(GS), np.array(ES), np.array(err_pp), np.array(err_GS), np.array(err_ES)

######################################

def Rebin_energyscans_PP_noPair(pump, Ipump, lights, darks, scanvar, readbacks, threshold=0):
    
    ordered = np.argsort(np.asarray(scanvar))
    peaks, what = find_peaks(np.diff(scanvar[ordered]))
    
    pump = pump[ordered]
    Ipump = Ipump[ordered]

    lights = lights[ordered]
    darks = darks[ordered]

    starts = np.append(0, peaks)
    ends = np.append(peaks, None)

    pp, GS, ES, err_pp, err_GS, err_ES = ([] for i in range(6))
    filtered = []

    for s, e in zip(starts, ends):
        pump_ebin1    = pump[s:e]
        Izero_p_ebin1 = Ipump[s:e]
        lights_ebin  = lights[s:e]
        darks_ebin   = darks[s:e]   

        thresh = (Izero_p_ebin1>threshold)

        pump_ebin1    = pump_ebin1[thresh]
        Izero_p_ebin1 = Izero_p_ebin1[thresh]
        lights_ebin   = lights_ebin[thresh]
        darks_ebin    = darks_ebin[thresh]

        pump_ebin    = pump_ebin1[lights_ebin]
        unpump_ebin  = pump_ebin1[darks_ebin]
        Izero_p_ebin = Izero_p_ebin1[lights_ebin]
        Izero_u_ebin = Izero_p_ebin1[darks_ebin]

        pump_ebin = pump_ebin / Izero_p_ebin
        unpump_ebin = unpump_ebin / Izero_u_ebin

        #pp_shot = np.log10(pump_ebin[thresh]/unpump_ebin[thresh])
        #pp_shot = np.log10(pump_ebin/unpump_ebin)
        #pp_shot = pump_ebin - unpump_ebin
        pp_ebin = np.nanmean(pump_ebin) - np.nanmean(unpump_ebin)

        GS.append(np.nanmean(unpump_ebin))
        ES.append(np.nanmean(pump_ebin))
        pp.append(pp_ebin)
        GS_err = np.nanstd(unpump_ebin)/np.sqrt(len(unpump_ebin))
        ES_err = np.nanstd(pump_ebin)/np.sqrt(len(pump_ebin))
        err_GS.append(GS_err)
        err_ES.append(ES_err)
        err_pp.append(np.sqrt(GS_err**2 + ES_err**2))

    return np.array(pp), np.array(GS), np.array(ES), np.array(err_pp), np.array(err_GS), np.array(err_ES)


######################################

def Rebin_and_filter_energyscans_PP(data, quantile, readbacks, threshold=0, n_sigma=1, raw=True):
    
    for k,v in data.items():
        data[k] = v
    
    pump_1 = np.asarray(data['pump_1'])
    unpump_1 = np.asarray(data['unpump_1'])
    if raw: 
        pump_1 = np.asarray(data['pump_1_raw'])
        unpump_1 = np.asarray(data['unpump_1_raw'])
    Izero_pump = np.asarray(data['Izero_pump'])
    Izero_unpump = np.asarray(data['Izero_unpump'])
    energy = np.asarray(data['energypad'])
    
    ordered = np.argsort(np.asarray(energy))
    peaks, what = find_peaks(np.diff(energy[ordered]))
    
    pump_1 = pump_1[ordered]
    unpump_1 = unpump_1[ordered]
    Izero_pump = Izero_pump[ordered]
    Izero_unpump = Izero_unpump[ordered]

    starts = np.append(0, peaks)
    ends = np.append(peaks, None)

    pp, GS, ES, err_pp, err_GS, err_ES = ([] for i in range(6))
    filtered = []

    for s, e in zip(starts, ends):
        pump    = pump_1[s:e]
        unpump  = unpump_1[s:e]
        Izero_p = Izero_pump[s:e]
        Izero_u = Izero_unpump[s:e]
        
        thresh   = (Izero_p > threshold) & (Izero_u > threshold)
        filterIp = Izero_p > (np.nanmedian(Izero_p) - n_sigma*np.std(Izero_p))
        filterIu = Izero_u > (np.nanmedian(Izero_u) - n_sigma*np.std(Izero_u))

        pump    = pump[filterIp & filterIu & thresh]
        unpump  = unpump[filterIp & filterIu & thresh]
        Izero_p = Izero_p[filterIp & filterIu & thresh]
        Izero_u = Izero_u[filterIp & filterIu & thresh]
        
        ratio_p = pump/Izero_p
        ratio_u = unpump/Izero_u
    
        filtervals = create_corr_condition(ratio_p, ratio_u, quantile)
        filtered.extend(filtervals)

        pump_1_filter       = pump[filtervals]
        unpump_1_filter     = unpump[filtervals]
        Izero_pump_filter   = Izero_p[filtervals]
        Izero_unpump_filter = Izero_u[filtervals]

        #unpump_1_filter = unpump_1_filter1/np.nanmean(unpump_1_filter1)

        pump_filter = pump_1_filter / Izero_pump_filter
        unpump_filter = unpump_1_filter / Izero_unpump_filter
        #pp_shot = np.log10(pump_filter/unpump_filter)
        pp_shot = pump_filter - unpump_filter

        GS.append(np.nanmean(unpump_filter))
        ES.append(np.nanmean(pump_filter))
        pp.append(np.nanmean(pp_shot))
        err_GS.append(np.nanstd(unpump_filter)/np.sqrt(len(unpump_filter)))
        err_ES.append(np.nanstd(pump_filter)/np.sqrt(len(pump_filter)))
        err_pp.append(np.nanstd(pp_shot)/np.sqrt(len(pp_shot)))

    print (len(peaks), len(readbacks), len(GS))

    GS = np.reshape(np.array(GS), (len(readbacks),-1)) 
    ES = np.reshape(np.array(ES), (len(readbacks), -1))
    pp = np.reshape(np.array(pp), (len(readbacks), -1))
    err_GS = np.reshape(np.array(err_GS), (len(readbacks), -1))
    err_ES = np.reshape(np.array(err_ES), (len(readbacks), -1))
    err_pp = np.reshape(np.array(err_pp), (len(readbacks), -1))

    nscans = np.shape(GS)[1]

    err_GS2 = np.nanstd(GS, axis=1)
    err_ES2 = np.nanstd(ES, axis=1)
    err_pp2 = np.nanstd(pp, axis=1)

    GS = np.nanmean(GS, axis=1)
    ES = np.nanmean(ES, axis=1)
    pp = np.nanmean(pp, axis=1)

    err_GS = np.sqrt(np.sum(err_GS**2, axis=1))
    err_ES = np.sqrt(np.sum(err_ES**2, axis=1))
    err_pp = np.sqrt(np.sum(err_pp**2, axis=1))

    print ('{} shots out of {} survived'.format(np.sum(filtered), len(pump_1)))
    res = {'pp': np.array(pp), 'GS': np.array(GS), 'ES': np.array(ES), 'err_pp': np.array(err_pp), 'err_GS': np.array(err_GS), 'err_ES': np.array(err_ES), 'err_pp2': np.array(err_pp2), 'err_GS2': np.array(err_GS2), 'err_ES2': np.array(err_ES2), 'filtered': filtered}
    return res
#np.array(pp), np.array(GS), np.array(ES), np.array(err_pp), np.array(err_GS), np.array(err_ES), np.array(err_pp2), np.array(err_GS2), np.array(err_ES2), filtered

######################################

def Rebin_and_filter_energyscans_PP_noPair(data, quantile, readbacks, threshold=0, n_sigma=1, raw=True):
    
    for k,v in data.items():
        data[k] = v
    
    pump_1 = np.asarray(data['pump_1'])
    if raw: 
        pump_1 = np.asarray(data['pump_1_raw'])
    Izero_pump = np.asarray(data['Izero_pump'])
    energy = np.asarray(data['energypad'])
    lights = np.asarray(data['lights'])
    darks  = np.asarray(data['darks'])    
    
    ordered = np.argsort(np.asarray(energy))
    peaks, what = find_peaks(np.diff(energy[ordered]))
    
    pump_1 = pump_1[ordered]
    Izero_pump = Izero_pump[ordered]
    lights = lights[ordered]
    darks = darks[ordered]

    starts = np.append(0, peaks)
    ends = np.append(peaks, None)

    pp, GS, ES, err_pp, err_GS, err_ES = ([] for i in range(6))
    filtered_p = []
    filtered_u = []

    for s, e in zip(starts, ends):
        pump1    = pump_1[s:e]
        Izero_p1 = Izero_pump[s:e]
        lights_e = lights[s:e]
        darks_e  = darks[s:e]       

        thresh  = Izero_p1 > threshold
        filterI = Izero_p1 > (np.nanmedian(Izero_p1) - n_sigma*np.std(Izero_p1))
        
        pump1    = pump1[filterI & thresh]
        Izero_p1 = Izero_p1[filterI & thresh]
        lights_e = lights_e[filterI & thresh]
        darks_e  = darks_e[filterI & thresh]

        pump    = pump1[lights_e]
        unpump  = pump1[darks_e]
        Izero_p = Izero_p1[lights_e]
        Izero_u = Izero_p1[darks_e]
        
        ratio_p = pump/Izero_p
        ratio_u = unpump/Izero_u
        
        filtervals_p, filtervals_u = create_corr_condition_noPair(ratio_p, ratio_u, quantile)
        filtered_p.extend(filtervals_p)
        filtered_u.extend(filtervals_u)
        
        pump_filter         = pump[filtervals_p]
        unpump_filter       = unpump[filtervals_u]
        Izero_pump_filter   = Izero_p[filtervals_p]
        Izero_unpump_filter = Izero_u[filtervals_u]

        pump_filter = pump_filter / Izero_pump_filter
        unpump_filter = unpump_filter / Izero_unpump_filter
        #pp_shot = np.log10(pump_filter/unpump_filter)
        pp_ebin = np.nanmean(pump_filter) - np.nanmean(unpump_filter)

        GS.append(np.nanmean(unpump_filter))
        ES.append(np.nanmean(pump_filter))
        pp.append(pp_ebin)
        GS_err = np.nanstd(unpump_filter)/np.sqrt(len(unpump_filter))
        ES_err = np.nanstd(pump_filter)/np.sqrt(len(pump_filter))

        err_GS.append(GS_err)
        err_ES.append(ES_err)
        err_pp.append(np.sqrt(GS_err**2 + ES_err**2))

    print (len(peaks), len(readbacks), len(GS))

    GS = np.reshape(np.array(GS), (len(readbacks),-1)) 
    ES = np.reshape(np.array(ES), (len(readbacks), -1))
    pp = np.reshape(np.array(pp), (len(readbacks), -1))
    err_GS = np.reshape(np.array(err_GS), (len(readbacks), -1))
    err_ES = np.reshape(np.array(err_ES), (len(readbacks), -1))
    err_pp = np.reshape(np.array(err_pp), (len(readbacks), -1))

    nscans = np.shape(GS)[1]

    err_GS2 = np.nanstd(GS, axis=1)
    err_ES2 = np.nanstd(ES, axis=1)
    err_pp2 = np.nanstd(pp, axis=1)

    GS = np.nanmean(GS, axis=1)
    ES = np.nanmean(ES, axis=1)
    pp = np.nanmean(pp, axis=1)

    err_GS = np.sqrt(np.sum(err_GS**2, axis=1))
    err_ES = np.sqrt(np.sum(err_ES**2, axis=1))
    err_pp = np.sqrt(np.sum(err_pp**2, axis=1))

    print ('{} shots out of {} survived'.format(np.sum(filtered_p)+np.sum(filtered_u), len(pump_1)))
    return np.array(pp), np.array(GS), np.array(ES), np.array(err_pp), np.array(err_GS), np.array(err_ES), np.array(err_pp2), np.array(err_GS2), np.array(err_ES2), filtered_p, filtered_u

######################################

def Rebin_timescans(pump, unpump, Ipump, Iunpump, delaystage, readbacks, threshold=0, varbin_t=False):

    pump = np.asarray(pump)
    unpump = np.asarray(unpump)
    Ipump = np.asarray(Ipump)
    Iunpump = np.asarray(Iunpump)
    delaystage = np.asarray(delaystage)

    stepsize = (readbacks[-1]-readbacks[0])/(len(readbacks)-1)
    binList = np.linspace(readbacks[0]-stepsize/2, readbacks[-1]+stepsize/2, len(readbacks)+1)
    if varbin_t:
        binList = histedges_equalN(delaystage, len(readbacks)+1)
    bin_centres = (binList[:-1] + binList[1:])/2

    delay_rebin = readbacks
    if varbin_t:
        delay_rebin = bin_centres

    GS = np.zeros(len(bin_centres))
    ES = np.zeros(len(bin_centres))
    pp_rebin = np.zeros(len(bin_centres))

    err_GS = np.zeros(len(bin_centres))
    err_ES = np.zeros(len(bin_centres))    
    err_pp = np.zeros(len(bin_centres))

    for i in range(len(bin_centres)):
        cond1 = delaystage >= binList[i]
        cond2 = delaystage < binList[i+1]
    
        idx = np.where(cond1*cond2)[0]
        delay_rebin[i] = np.average(delaystage[idx])
    
        p   = pump[idx]
        u   = unpump[idx]
        I_p = Ipump[idx]
        I_u = Iunpump[idx]

        p = p / I_p
        u = u / I_u
        thresh   = (I_p>threshold) & (I_u>threshold)
        #Pump_probe_shot = np.log10(p[thresh]/u[thresh])
        Pump_probe_shot = p[thresh] - u[thresh]
        
        GS[i] = np.nanmean(u)
        ES[i] = np.nanmean(p)
        pp_rebin[i]  = np.nanmean(Pump_probe_shot)
        err_GS = np.nanstd(u)/np.sqrt(len(u))
        err_ES = np.nanstd(p)/np.sqrt(len(p))
        err_pp[i] = np.nanstd(Pump_probe_shot)/np.sqrt(len(Pump_probe_shot))
    
    print('Time delay axis rebinned with delay stage data')
    
    return np.array(pp_rebin), np.array(GS), np.array(ES), np.array(err_pp), np.array(err_GS), np.array(err_ES), delay_rebin

######################################

def Rebin_timescans_noPair(pump, Ipump, lights, darks, delaystage, readbacks, threshold=0, varbin_t=False):

    pump = np.asarray(pump)
    Ipump = np.asarray(Ipump)
    lights = np.asarray(lights)
    darks = np.asarray(darks)
    delaystage = np.asarray(delaystage)

    stepsize = (readbacks[-1]-readbacks[0])/(len(readbacks)-1)
    binList = np.linspace(readbacks[0]-stepsize/2, readbacks[-1]+stepsize/2, len(readbacks)+1)
    if varbin_t:
        binList = histedges_equalN(delaystage, len(readbacks)+1)
    bin_centres = (binList[:-1] + binList[1:])/2

    delay_rebin = readbacks
    if varbin_t:
        delay_rebin = bin_centres

    GS = np.zeros(len(bin_centres))
    ES = np.zeros(len(bin_centres))
    pp_rebin = np.zeros(len(bin_centres))

    err_GS = np.zeros(len(bin_centres))
    err_ES = np.zeros(len(bin_centres))    
    err_pp = np.zeros(len(bin_centres))

    for i in range(len(bin_centres)):
        cond1 = delaystage >= binList[i]
        cond2 = delaystage < binList[i+1]
    
        idx = np.where(cond1*cond2)[0]
        delay_rebin[i] = np.average(delaystage[idx])
    
        p1  = pump[idx]
        I_p1 = Ipump[idx]
        lights_t = lights[idx]
        darks_t = darks[idx]

        thresh   = (I_p1>threshold)

        p1 = p1[thresh]
        I_p1 = I_p1[thresh]
        lights_t = lights_t[thresh]
        darks_t = darks_t[thresh]

        p   = p1[lights_t]
        u   = p1[darks_t]
        I_p = I_p1[lights_t]
        I_u = I_p1[darks_t]

        p = p / I_p
        u = u / I_u
        
        #Pump_probe_shot = np.log10(p[thresh]/u[thresh])
        Pump_probe = np.nanmean(p) - np.nanmean(u)
        
        GS[i] = np.nanmean(u)
        ES[i] = np.nanmean(p)
        pp_rebin[i]  = Pump_probe
        err_GS = np.nanstd(u)/np.sqrt(len(u))
        err_ES = np.nanstd(p)/np.sqrt(len(p))
        err_pp[i] = np.sqrt(err_GS**2 + err_ES**2)
    
    print('Time delay axis rebinned with delay stage data')
    
    return np.array(pp_rebin), np.array(GS), np.array(ES), np.array(err_pp), np.array(err_GS), np.array(err_ES), delay_rebin

######################################

def Rebin_and_filter_timescans(data, binsize, minvalue, maxvalue, quantile, withTT, threshold=0, n_sigma=1, raw=True, numbins=None, varbin_t=False):

    for k,v in data.items():
        data[k] = v
    
    pump_1 = np.asarray(data['pump_1'])
    unpump_1 = np.asarray(data['unpump_1'])
    if raw: 
        pump_1 = np.asarray(data['pump_1_raw'])
        unpump_1 = np.asarray(data['unpump_1_raw'])
    Izero_pump = np.asarray(data['Izero_pump'])
    Izero_unpump = np.asarray(data['Izero_unpump'])
    Delays_stage = np.asarray(data['Delays_stage'])
    Delays_corr = np.asarray(data['Delays_corr'])

    if withTT:
        Delays = Delays_corr
    else:
        Delays = Delays_stage

    binList = np.arange(minvalue, maxvalue, binsize)
    if varbin_t:
        binList = histedges_equalN(Delays, numbins)

    bin_centres = (binList[:-1] + binList[1:])/2
    delay_rebin = np.arange(minvalue + binsize/2, maxvalue - binsize/2, binsize)
    if varbin_t:
        delay_rebin = bin_centres

    pp_rebin = np.zeros(len(bin_centres))
    err_pp = np.zeros(len(bin_centres))

    totalshots = len(pump_1)
    howmany_before = []
    howmany = []

    for i in range(len(bin_centres)):
        cond1 = Delays >= binList[i]
        cond2 = Delays < binList[i+1]
    
        idx = np.where(cond1*cond2)[0]
        if varbin_t:
            delay_rebin[i] = np.average(Delays[idx])
    
        pump    = pump_1[idx]
        unpump  = unpump_1[idx]
        Izero_p = Izero_pump[idx]
        Izero_u = Izero_unpump[idx]

        thresh   = (Izero_p > threshold) & (Izero_u > threshold)
        filterIp = Izero_p > (np.nanmedian(Izero_p) - n_sigma*np.std(Izero_p))
        filterIu = Izero_u > (np.nanmedian(Izero_u) - n_sigma*np.std(Izero_u))

        pump    = pump[filterIp & filterIu & thresh]
        unpump  = unpump[filterIp & filterIu & thresh]
        Izero_p = Izero_p[filterIp & filterIu & thresh]
        Izero_u = Izero_u[filterIp & filterIu & thresh]
        
        howmany_before.append(len(Delays[idx]))
    
        pump_filter, unpump_filter, Izero_pump_filter, Izero_unpump_filter = \
        correlation_filter(pump, unpump, Izero_p, Izero_u, quantile)
        
        howmany.append(len(pump_filter))

        pump_filter = pump_filter / Izero_pump_filter
        unpump_filter = unpump_filter / Izero_unpump_filter
    
        #Pump_probe_shot = np.log10(pump_filter/unpump_filter)
        Pump_probe_shot = pump_filter - unpump_filter
    
        pp_rebin[i]  = np.nanmean(Pump_probe_shot)
        err_pp[i] = np.nanstd(Pump_probe_shot)/np.sqrt(len(Pump_probe_shot))
    if withTT:
        print('Time delay axis rebinned with TT data')
    else:
        print('Time delay axis rebinned with delay stage data')

    print ('{} shots out of {} survived (total shots: {})'.format(np.sum(howmany), np.sum(howmany_before), totalshots))

    res = {'pp': np.array(pp_rebin), 'err_pp': np.array(err_pp), 'Delay': np.array(delay_rebin), 'howmany': howmany}   
    return res

#    return pp_rebin, err_pp, delay_rebin, howmany

######################################

def Rebin_and_filter_timescans_noPair(data, binsize, minvalue, maxvalue, quantile, withTT, threshold=0, n_sigma=1, raw=True, numbins=None, varbin_t=False):
    for k,v in data.items():
        data[k] = v
        
    diode = np.asarray(data['pump_1'])
    if raw: 
        diode = np.asarray(data['pump_1_raw'])
    Izero = np.asarray(data['Izero_pump'])
    lights = np.asarray(data['lights'])
    darks = np.asarray(data['darks'])
    Delays_stage = np.asarray(data['Delays_stage'])
    Delays_corr = np.asarray(data['Delays_corr'])

    if withTT:
        Delays = Delays_corr#[lights]
    else:
        Delays = Delays_stage#[lights]

    binList = np.arange(minvalue, maxvalue, binsize)
    if varbin_t:
        binList = histedges_equalN(Delays, numbins)

    bin_centres = (binList[:-1] + binList[1:])/2
    delay_rebin = np.arange(minvalue + binsize/2, maxvalue - binsize/2, binsize)
    if varbin_t:
        delay_rebin = bin_centres

    pp_rebin = np.zeros(len(bin_centres))
    GS_rebin = np.zeros(len(bin_centres))
    ES_rebin = np.zeros(len(bin_centres))
    err_pp = np.zeros(len(bin_centres))

    totalshots = len(diode)
    howmany_before = []
    howmany = []
    filtered_p = []
    filtered_u = []

    for i in range(len(bin_centres)):
        cond1 = Delays >= binList[i]
        cond2 = Delays < binList[i+1]
    
        idx = np.where(cond1*cond2)[0]
        #delay_rebin[i] = np.average(Delays[idx])
        howmany_before.append(len(Delays[idx]))
    
        diode_t  = diode[idx]
        Izero_t  = Izero[idx]
        lights_t = lights[idx]
        darks_t  = darks[idx]
        delays_t = Delays[idx]

        thresh   = Izero_t > threshold
        filterI  = Izero_t > (np.nanmedian(Izero_t) - n_sigma*np.std(Izero_t))
 
        diode_t  = diode_t[filterI & thresh]
        Izero_t  = Izero_t[filterI & thresh]
        lights_t = lights_t[filterI & thresh]
        darks_t  = darks_t[filterI & thresh]
        delays_t = delays_t[filterI & thresh]

        pump     = diode_t[lights_t]
        unpump   = diode_t[darks_t]
        Izero_p  = Izero_t[lights_t]
        Izero_u  = Izero_t[darks_t]
        delays_p = delays_t[lights_t]

        ratio_p = pump/Izero_p
        ratio_u = unpump/Izero_u
        
        filtervals_p, filtervals_u = create_corr_condition_noPair(ratio_p, ratio_u, quantile)

        pump_filter         = pump[filtervals_p]
        unpump_filter       = unpump[filtervals_u]
        Izero_pump_filter   = Izero_p[filtervals_p]
        Izero_unpump_filter = Izero_u[filtervals_u]

        howmany.append(len(pump_filter)+len(unpump_filter))
        
        pump_filter = pump_filter / Izero_pump_filter
        unpump_filter = unpump_filter / Izero_unpump_filter

        delay_rebin[i] = np.average(delays_p)
        pp_rebin[i]  = np.nanmean(pump_filter) - np.nanmean(unpump_filter)
        GS_rebin[i]  = np.nanmean(unpump_filter)
        ES_rebin[i]  = np.nanmean(pump_filter)
        err_pp[i] = (np.sqrt(np.nanstd(pump_filter)**2 + np.nanstd(unpump_filter)**2))/np.sqrt(len(pump_filter))

    if withTT:
        print('Time delay axis rebinned with TT data')
    else:
        print('Time delay axis rebinned with delay stage data')
    print ('{} shots out of {} survived (total shots: {})'.format(np.sum(filtered_p)+np.sum(filtered_u), np.sum(howmany_before), totalshots))
    return pp_rebin, GS_rebin, ES_rebin, err_pp, delay_rebin, howmany

######################################

def Rebin_and_filter_2Dscans(data, binsize, minvalue, maxvalue, quantile, readbacks, withTT, threshold=0, n_sigma=1, raw=True, varbin_t=False, numbins=None):
    for k,v in data.items():
        data[k] = v

    pump_1 = np.asarray(data['pump_1'])
    unpump_1 = np.asarray(data['unpump_1'])
    if raw: 
        pump_1 = np.asarray(data['pump_1_raw'])
        unpump_1 = np.asarray(data['unpump_1_raw'])
    Izero_pump = np.asarray(data['Izero_pump'])
    Izero_unpump = np.asarray(data['Izero_unpump'])
    energypad = np.asarray(data['energypad'])
    Delays_stage = np.asarray(data['Delays_stage'])
    Delays_corr = np.asarray(data['Delays_corr'])

    if withTT:
        Delays = Delays_corr
    else:
        Delays = Delays_stage

    binList = np.arange(minvalue, maxvalue, binsize)
    if varbin_t:
        binList = histedges_equalN(Delays, numbins)
    
    bin_centres = (binList[:-1] + binList[1:])/2
    delay_rebin = np.arange(minvalue + binsize/2, maxvalue - binsize/2, binsize)
    if varbin_t:
        delay_rebin = bin_centres

    ordered = np.argsort(np.asarray(energypad))
    peaks, what = find_peaks(np.diff(energypad[ordered]))
    
    pump_1 = pump_1[ordered]
    unpump_1 = unpump_1[ordered]
    Izero_pump = Izero_pump[ordered]
    Izero_unpump = Izero_unpump[ordered]
    Delays = Delays[ordered]

    starts = np.append(0, peaks)
    ends = np.append(peaks, None)

    pump_in_ebin, unpump_in_ebin, Izero_p_in_ebin, Izero_u_in_ebin, Delays_in_ebin = ([] for i in range(5))

    pp_rebin = np.zeros((len(starts), len(bin_centres)))
    GS_rebin = np.zeros((len(starts), len(bin_centres)))
    ES_rebin = np.zeros((len(starts), len(bin_centres)))
    err_pp   = np.zeros((len(starts), len(bin_centres)))
    howmany_before = len(pump_1)
    howmany = []

    print (len(peaks), len(readbacks), len(bin_centres))
    for i, (s, e) in enumerate(zip(starts, ends)):
        pump_in_ebin = pump_1[s:e]
        unpump_in_ebin = unpump_1[s:e]
        Izero_p_in_ebin = Izero_pump[s:e]
        Izero_u_in_ebin = Izero_unpump[s:e]
        Delays_in_ebin = Delays[s:e]

        thresh   = (Izero_p_in_ebin > threshold) & (Izero_u_in_ebin > threshold)
        filterIp = Izero_p_in_ebin > (np.nanmedian(Izero_p_in_ebin) - n_sigma*np.std(Izero_p_in_ebin))
        filterIu = Izero_u_in_ebin > (np.nanmedian(Izero_u_in_ebin) - n_sigma*np.std(Izero_u_in_ebin))

        pump_in_ebin    = pump_in_ebin[filterIp & filterIu & thresh]
        unpump_in_ebin  = unpump_in_ebin[filterIp & filterIu & thresh]
        Izero_p_in_ebin = Izero_p_in_ebin[filterIp & filterIu & thresh]
        Izero_u_in_ebin = Izero_u_in_ebin[filterIp & filterIu & thresh]
        Delays_in_ebin  = Delays_in_ebin[filterIp & filterIu & thresh]

        for j in range(len(bin_centres)):
            cond1 = np.asarray(Delays_in_ebin) >= binList[j]
            cond2 = np.asarray(Delays_in_ebin) < binList[j+1]

            idx = np.where(cond1*cond2)[0]
            if varbin_t:
            	delay_rebin[j]  = np.nanmean(np.asarray(Delays_in_ebin)[idx])
            
            pump_in_tbin    = np.asarray(pump_in_ebin)[idx]
            unpump_in_tbin  = np.asarray(unpump_in_ebin)[idx]
            Izero_p_in_tbin = np.asarray(Izero_p_in_ebin)[idx]
            Izero_u_in_tbin = np.asarray(Izero_u_in_ebin)[idx]

            pump_filter, unpump_filter, Izero_pump_filter, Izero_unpump_filter = \
            correlation_filter(pump_in_tbin, unpump_in_tbin, Izero_p_in_tbin, Izero_u_in_tbin, quantile)

            howmany.append(len(pump_filter))

            pump_filter = pump_filter / Izero_pump_filter
            unpump_filter = unpump_filter / Izero_unpump_filter
                
            Pump_probe_shot = (pump_filter - unpump_filter)
            GS_shot = unpump_filter
            ES_shot = pump_filter
            
            pp_rebin[i, j] = np.nanmean(Pump_probe_shot)
            GS_rebin[i, j] = np.nanmean(GS_shot)
            ES_rebin[i, j] = np.nanmean(ES_shot)
            err_pp[i, j] = np.nanstd(Pump_probe_shot)/np.sqrt(len(Pump_probe_shot))

    pp_rebin = np.reshape(np.array(pp_rebin), (len(readbacks), -1, len(bin_centres)))
    pp_rebin = np.nanmean(pp_rebin, axis=1)
    GS_rebin = np.reshape(np.array(GS_rebin), (len(readbacks), -1, len(bin_centres)))
    GS_rebin = np.nanmean(GS_rebin, axis=1)
    ES_rebin = np.reshape(np.array(ES_rebin), (len(readbacks), -1, len(bin_centres)))
    ES_rebin = np.nanmean(ES_rebin, axis=1)
    err_pp = np.reshape(np.array(err_pp), (len(readbacks), -1, len(bin_centres)))
    err_pp = np.nanmean(err_pp, axis=1)

    if withTT:
        print('Time delay axis rebinned with TT data')
    else:
        print('Time delay axis rebinned with delay stage data')
    print ('{} shots out of {} survived ({:.2f}%)'.format(np.sum(howmany), np.sum(howmany_before), 100*np.sum(howmany)/np.sum(howmany_before)))

    res = {'GS': np.array(GS_rebin), 'ES': np.array(ES_rebin), 'pp': np.array(pp_rebin), 'err_pp': np.array(err_pp), 'Delay': np.array(delay_rebin), 'howmany': howmany}
    return res

#    return pp_rebin, err_pp, GS_rebin, ES_rebin, delay_rebin, howmany

######################################

def Rebin_and_filter_2Dscans_noPair(data, binsize, minvalue, maxvalue, quantile, readbacks, withTT, threshold=0, n_sigma=1, raw=True, varbin_t=False, numbins=None):
    for k,v in data.items():
        data[k] = v

    pump_1 = np.asarray(data['pump_1'])
    if raw: 
        pump_1 = np.asarray(data['pump_1_raw'])
    Izero_pump = np.asarray(data['Izero_pump'])
    lights = np.asarray(data['lights'])    
    darks = np.asarray(data['darks'])

    print (len(pump_1), len(Izero_pump), len(lights), len(darks))

    energypad = np.asarray(data['energypad'])
    Delays_stage = np.asarray(data['Delays_stage'])
    Delays_corr = np.asarray(data['Delays_corr'])

    if withTT:
        Delays = Delays_corr
    else:
        Delays = Delays_stage

    binList = np.arange(minvalue, maxvalue, binsize)
    if varbin_t:
        binList = histedges_equalN(Delays, numbins)
    
    bin_centres = (binList[:-1] + binList[1:])/2
    delay_rebin = np.arange(minvalue + binsize/2, maxvalue - binsize/2, binsize)
    if varbin_t:
        delay_rebin = bin_centres

    ordered = np.argsort(np.asarray(energypad))
    peaks, what = find_peaks(np.diff(energypad[ordered]))
    
    pump_1 = pump_1[ordered]    
    Izero_pump = Izero_pump[ordered]
    lights = lights[ordered]
    darks = darks[ordered]
    Delays = Delays[ordered]

    starts = np.append(0, peaks)
    ends = np.append(peaks, None)

    pump_in_ebin, Izero_p_in_ebin, Delays_in_ebin, lights_in_ebin, darks_in_ebin = ([] for i in range(5))

    pp_rebin = np.zeros((len(starts), len(bin_centres)))
    GS_rebin = np.zeros((len(starts), len(bin_centres)))
    ES_rebin = np.zeros((len(starts), len(bin_centres)))
    err_pp   = np.zeros((len(starts), len(bin_centres)))
    howmany_before = len(pump_1)
    howmany = []

    print (len(peaks), len(readbacks), len(bin_centres))
    for i, (s, e) in enumerate(zip(starts, ends)):
        pump_in_ebin = pump_1[s:e]
        Izero_p_in_ebin = Izero_pump[s:e]
        Delays_in_ebin = Delays[s:e]
        lights_in_ebin = lights[s:e]
        darks_in_ebin = darks[s:e]

        thresh   = (Izero_p_in_ebin > threshold)
        filterIp = Izero_p_in_ebin > (np.nanmedian(Izero_p_in_ebin) - n_sigma*np.std(Izero_p_in_ebin))

        pump_in_ebin    = pump_in_ebin[filterIp & thresh]
        Izero_p_in_ebin = Izero_p_in_ebin[filterIp & thresh]
        Delays_in_ebin  = Delays_in_ebin[filterIp & thresh]
        lights_in_ebin  = lights_in_ebin[filterIp & thresh]
        darks_in_ebin   = darks_in_ebin[filterIp & thresh]
        
        for j in range(len(bin_centres)):
            cond1 = np.asarray(Delays_in_ebin) >= binList[j]
            cond2 = np.asarray(Delays_in_ebin) < binList[j+1]

            idx = np.where(cond1*cond2)[0]
            #delay_rebin[j]  = np.average(np.asarray(Delays_in_ebin)[idx])
            
            pump_in_tbin    = np.asarray(pump_in_ebin)[idx]
            Izero_p_in_tbin = np.asarray(Izero_p_in_ebin)[idx]
            lights_in_tbin  = np.asarray(lights_in_ebin)[idx]
            darks_in_tbin   = np.asarray(darks_in_ebin)[idx]
            delays_in_tbin  = np.asarray(Delays_in_ebin)[idx]
        
            delay_rebin[j]  = np.average(delays_in_tbin)

            Diode_norm = pump_in_tbin / Izero_p_in_tbin

            qnt_low  = np.nanquantile(Diode_norm, 0.5 - quantile/2)
            qnt_high = np.nanquantile(Diode_norm, 0.5 + quantile/2)
    
            condition_low = Diode_norm > qnt_low
            condition_high = Diode_norm < qnt_high

            correlation_filter = condition_low & condition_high

            Diode_filter  = pump_in_tbin[correlation_filter]
            Izero_filter  = Izero_p_in_tbin[correlation_filter]
            darks_filter  = darks_in_tbin[correlation_filter]
            lights_filter = lights_in_tbin[correlation_filter]

            howmany.append(len(Diode_filter))

            p  = Diode_filter[lights_filter]
            u  = Diode_filter[darks_filter]
            Ip = Izero_filter[lights_filter]
            Iu = Izero_filter[darks_filter]

            p = p/Ip
            u = u/Iu

            pp_rebin[i, j]  = np.nanmean(p) - np.nanmean(u)
            GS_rebin[i, j] = np.nanmean(u)            
            ES_rebin[i, j] = np.nanmean(p)

            err_GS = np.nanstd(u)/np.sqrt(len(u))
            err_ES = np.nanstd(p)/np.sqrt(len(p)) 

            err_pp[i, j] = np.sqrt(err_GS**2 + err_ES**2)

    pp_rebin = np.reshape(np.array(pp_rebin), (len(readbacks), -1, len(bin_centres)))
    GS_rebin = np.reshape(np.array(GS_rebin), (len(readbacks), -1, len(bin_centres)))
    ES_rebin = np.reshape(np.array(ES_rebin), (len(readbacks), -1, len(bin_centres)))

    pp_rebin = np.nanmean(pp_rebin, axis=1)
    GS_rebin = np.nanmean(GS_rebin, axis=1)
    ES_rebin = np.nanmean(ES_rebin, axis=1)

    err_pp = np.reshape(np.array(err_pp), (len(readbacks), -1, len(bin_centres)))
    err_pp = np.nanmean(err_pp, axis=1)

    if withTT:
        print('Time delay axis rebinned with TT data')
    else:
        print('Time delay axis rebinned with delay stage data')
    print ('{} shots out of {} survived ({:.2f}%)'.format(np.sum(howmany), np.sum(howmany_before), 100*np.sum(howmany)/np.sum(howmany_before)))

    return pp_rebin, GS_rebin, ES_rebin, err_pp, delay_rebin, howmany



######################################
######################################
######################################

def Rebin_scans_PP(pump, unpump, Ipump, Iunpump, scanvar, readbacks):
    
    ordered = np.argsort(np.asarray(scanvar))
    peaks, what = find_peaks(np.diff(scanvar[ordered]))
    
    pump = pump[ordered]
    unpump = unpump[ordered]
    Ipump = Ipump[ordered]
    Iunpump = Iunpump[ordered]

    starts = np.append(0, peaks)
    ends = np.append(peaks, None)

    pp, GS, ES, err_pp, err_GS, err_ES = ([] for i in range(6))
    
    for s, e in zip(starts, ends):
        pump_ebin    = pump[s:e]
        unpump_ebin  = unpump[s:e]
        Izero_p_ebin = Ipump[s:e]
        Izero_u_ebin = Iunpump[s:e]
        
        pump_ebin = pump_ebin / Izero_p_ebin
        unpump_ebin = unpump_ebin / Izero_u_ebin
        #pp_shot = np.log10(pump_ebin/unpump_ebin)
        pp_shot = pump_ebin - unpump_ebin

        GS.append(np.nanmean(unpump_ebin))
        ES.append(np.nanmean(pump_ebin))
        pp.append(np.nanmean(pp_shot))
        err_GS.append(np.nanstd(unpump_ebin)/np.sqrt(len(unpump_ebin)))
        err_ES.append(np.nanstd(pump_ebin)/np.sqrt(len(pump_ebin)))
        err_pp.append(np.nanstd(pp_shot)/np.sqrt(len(pp_shot)))

    return np.array(pp), np.array(GS), np.array(ES), np.array(err_pp), np.array(err_GS), np.array(err_ES)

######################################

def Rebin_scans_PP_old(pump, unpump, Ipump, Iunpump, energy, readbacks, varbin_e=False):

    stepsize_energy = (readbacks[-1]-readbacks[0])/(len(readbacks)-1)
    binList_energy = np.linspace(readbacks[0]-stepsize_energy/2, readbacks[-1]+stepsize_energy/2, len(readbacks)+1)
    if varbin_e:
        binList_energy = histedges_equalN(energy, len(readbacks)+1)
    bin_centres_energy = (binList_energy[:-1] + binList_energy[1:])/2

    pp, GS, ES, err_pp, err_GS, err_ES = (np.zeros(len(bin_centres_energy)) for i in range(6))
    
    for i in range(len(bin_centres_energy)):
        cond1e = energy >= binList_energy[i]
        cond2e = energy < binList_energy[i+1]
        
        idx = np.where(cond1e*cond2e)[0]
        pump_ebin    = pump[idx]
        unpump_ebin  = unpump[idx]
        Izero_p_ebin  = Ipump[idx]
        Izero_u_ebin  = Iunpump[idx]

        pump_ebin   = pump_ebin / Izero_p_ebin
        unpump_ebin = unpump_ebin / Izero_u_ebin    
        #pp_ebin = np.log10(pump_ebin/unpump_ebin)
        pp_ebin = pump_ebin - unpump_ebin

        GS[i] = np.nanmean(unpump_ebin)
        err_GS[i] = np.nanstd(unpump_ebin)/np.sqrt(len(unpump_ebin))
        ES[i] = np.nanmean(pump_ebin)
        err_ES[i] = np.nanstd(pump_ebin)/np.sqrt(len(pump_ebin))
        pp[i] = np.nanmean(pp_ebin)
        err_pp[i] = np.nanstd(pp_ebin)/np.sqrt(len(pp_ebin))
    
    return pp, GS, ES, err_pp, err_GS, err_ES

######################################

def create_corr_condition(pump, unpump, quantile):

    qnt_low_p  = np.nanquantile(pump, 0.5 - quantile/2)
    qnt_high_p = np.nanquantile(pump, 0.5 + quantile/2)
    qnt_low_u  = np.nanquantile(unpump, 0.5 - quantile/2)
    qnt_high_u = np.nanquantile(unpump, 0.5 + quantile/2)

    filtervals_p_l = pump > qnt_low_p
    filtervals_p_h = pump < qnt_high_p
    filtervals_u_l = unpump > qnt_low_u
    filtervals_u_h = unpump < qnt_high_u

    condition = filtervals_p_l & filtervals_p_h & filtervals_u_l & filtervals_u_h
    return condition

######################################

def create_corr_condition_noPair(pump, unpump, quantile):

    qnt_low_p  = np.nanquantile(pump, 0.5 - quantile/2)
    qnt_high_p = np.nanquantile(pump, 0.5 + quantile/2)
    qnt_low_u  = np.nanquantile(unpump, 0.5 - quantile/2)
    qnt_high_u = np.nanquantile(unpump, 0.5 + quantile/2)

    filtervals_p_l = pump > qnt_low_p
    filtervals_p_h = pump < qnt_high_p
    filtervals_u_l = unpump > qnt_low_u
    filtervals_u_h = unpump < qnt_high_u

    condition_p = filtervals_p_l & filtervals_p_h
    condition_u = filtervals_u_l & filtervals_u_h
    return condition_p, condition_u


######################################
######################################
######################################

def Rebin_and_filter_energyscans_old(data, quantile, readbacks, varbin_e=False):
    
    for k,v in data.items():
        data[k] = v
    
    pump_1 = np.asarray(data['pump_1'])
    unpump_1 = np.asarray(data['unpump_1'])
    Izero_pump = np.asarray(data['Izero_pump'])
    Izero_unpump = np.asarray(data['Izero_unpump'])
    energy = np.asarray(data['energy'])

    stepsize_energy = (readbacks[-1]-readbacks[0])/(len(readbacks)-1)
    binList_energy = np.linspace(readbacks[0]-stepsize_energy/2, readbacks[-1]+stepsize_energy/2, len(readbacks)+1)
    if varbin_e:
        binList_energy = histedges_equalN(energy, len(readbacks)+1)
    bin_centres_energy = (binList_energy[:-1] + binList_energy[1:])/2

    pp_rebin = np.zeros(len(bin_centres_energy))
    GS       = np.zeros(len(bin_centres_energy))
    ES       = np.zeros(len(bin_centres_energy))
    err_pp   = np.zeros(len(bin_centres_energy))
    err_GS   = np.zeros(len(bin_centres_energy))
    err_ES   = np.zeros(len(bin_centres_energy))

    totalshots = len(pump_1)
    howmany_before = []
    howmany = []

    for i in range(len(bin_centres_energy)):
        cond1e = energy >= binList_energy[i]
        cond2e = energy < binList_energy[i+1]
        
        idx = np.where(cond1e*cond2e)[0]
        pump_ebin    = pump_1[idx]
        unpump_ebin  = unpump_1[idx]
        Izero_p_ebin = Izero_pump[idx]
        Izero_u_ebin = Izero_unpump[idx]

        howmany_before.append(len(energy[idx]))
    
        pump_filter, unpump_filter, Izero_pump_filter, Izero_unpump_filter = \
        correlation_filter(pump_ebin, unpump_ebin, Izero_p_ebin, Izero_u_ebin, quantile)

        howmany.append(len(pump_filter))

        pump_filter = pump_filter / Izero_pump_filter
        unpump_filter = unpump_filter / Izero_unpump_filter
        #Pump_probe_shot = np.log10(pump_filter/unpump_filter)
        Pump_probe_shot = pump_filter - unpump_filter

        GS[i]     = np.nanmean(unpump_filter)
        err_GS[i] = np.nanstd(unpump_filter)/np.sqrt(len(unpump_filter))
        ES[i]     = np.nanmean(pump_filter)
        err_ES[i] = np.nanstd(pump_filter)/np.sqrt(len(pump_filter))
        pp_rebin[i]  = np.nanmean(Pump_probe_shot)
        err_pp[i] = np.nanstd(Pump_probe_shot)/np.sqrt(len(Pump_probe_shot))

    print ('{} shots out of {} survived (total shots: {})'.format(np.sum(howmany), np.sum(howmany_before), totalshots))
    return pp_rebin, GS, ES, err_pp, err_GS, err_ES, howmany
    
######################################



######################################

def Rebin_and_filter_2Dscans_old(data, binsize, minvalue, maxvalue, quantile, readbacks, withTT, varbin_e=False, varbin_t=False, numbins=None):
    for k,v in data.items():
        data[k] = v

    pump_1 = np.asarray(data['pump_1'])
    unpump_1 = np.asarray(data['unpump_1'])
    Izero_pump = np.asarray(data['Izero_pump'])
    Izero_unpump = np.asarray(data['Izero_unpump'])
    energy = np.asarray(data['energy'])
    Delays_stage = np.asarray(data['Delays_stage'])
    Delays_corr = np.asarray(data['Delays_corr'])

    stepsize_energy = (readbacks[-1]-readbacks[0])/(len(readbacks)-1)
    binList_energy = np.linspace(readbacks[0]-stepsize_energy/2, readbacks[-1]+stepsize_energy/2, len(readbacks)+1)
    if varbin_e:
        binList_energy = histedges_equalN(energy, len(readbacks)+1)
    bin_centres_energy = (binList_energy[:-1] + binList_energy[1:])/2

    if withTT:
        Delays = Delays_corr
    else:
        Delays = Delays_stage

    binList = np.arange(minvalue, maxvalue, binsize)
    if varbin_t:
        binList = histedges_equalN(Delays, numbins)
    
    bin_centres = (binList[:-1] + binList[1:])/2
    delay_rebin = np.arange(minvalue + binsize/2, maxvalue - binsize/2, binsize)
    if varbin_t:
        delay_rebin = bin_centres

    pump_in_ebin, unpump_in_ebin, Izero_p_in_ebin, Izero_u_in_ebin, Delays_in_ebin = ([] for i in range(5))

    pp_rebin = np.zeros((len(bin_centres_energy), len(bin_centres)))
    err_pp   = np.zeros((len(bin_centres_energy), len(bin_centres)))

    for i in range(len(bin_centres_energy)):
        cond1e = energy >= binList_energy[i]
        cond2e = energy < binList_energy[i+1]
    
        idx = np.where(cond1e*cond2e)[0]
        pump_in_ebin.append(pump_1[idx])
        unpump_in_ebin.append(unpump_1[idx])
        Izero_p_in_ebin.append(Izero_pump[idx])
        Izero_u_in_ebin.append(Izero_unpump[idx])
    
        Delays_in_ebin.append(Delays[idx])

    howmany_before = []
    howmany = []

    for i, en in enumerate(readbacks):
        howmany_before.append(len(pump_in_ebin[i]))
        for j in range(len(bin_centres)):
            cond1 = np.asarray(Delays_in_ebin[i]) >= binList[j]
            cond2 = np.asarray(Delays_in_ebin[i]) < binList[j+1]
    
            idx = np.where(cond1*cond2)[0]
            delay_rebin[j] = np.average(np.asarray(Delays_in_ebin[i])[idx])
    
            pump_in_tbin       = np.asarray(pump_in_ebin[i])[idx]
            unpump_in_tbin     = np.asarray(unpump_in_ebin[i])[idx]
            
            Izero_p_in_tbin   = np.asarray(Izero_p_in_ebin[i])[idx]
            Izero_u_in_tbin = np.asarray(Izero_u_in_ebin[i])[idx]

            pump_filter, unpump_filter, Izero_pump_filter, Izero_unpump_filter = \
            correlation_filter(pump_in_tbin, unpump_in_tbin, Izero_p_in_tbin, Izero_u_in_tbin, quantile)

            howmany.append(len(pump_filter))    

            pump_filter = pump_filter / Izero_pump_filter
            unpump_filter = unpump_filter / Izero_unpump_filter
                
            #Pump_probe_shot = np.log10(pump_filter/unpump_filter)
            Pump_probe_shot = pump_filter - unpump_filter
            
            pp_rebin[i, j]  = np.nanmean(Pump_probe_shot)
            err_pp[i, j] = np.nanstd(Pump_probe_shot)/np.sqrt(len(Pump_probe_shot))

    if withTT:
        print('Time delay axis rebinned with TT data')
    else:
        print('Time delay axis rebinned with delay stage data')
    print ('{} shots out of {} survived ({:.2f}%)'.format(np.sum(howmany), np.sum(howmany_before), 100*np.sum(howmany)/np.sum(howmany_before)))

    return pp_rebin, err_pp, delay_rebin, howmany









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

    condition_Izero_pump = Izero_pump > 0.05
    condition_Izero_unpump = Izero_unpump > 0.05
    
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

    correlation_filter = condition_pump_low & condition_pump_high & condition_unpump_low & condition_unpump_high & condition_Izero_pump & condition_Izero_unpump
    
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
            delay_shot_fs = delay_shot #mm2fs(delay_shot, timezero_mm)
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
            delay_shot_fs = delay_shot #mm2fs(delay_shot, timezero_mm)
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


######################################

def rebin_XANES(Pump_probe_scan, Delays_corr_scan, Delays_fs_scan, variable_bins, withTT, numbins, binsize, min_delay, max_delay):
    from scipy.stats import binned_statistic
    print (np.shape(Pump_probe_scan),np.shape(Delays_corr_scan), np.shape(Delays_fs_scan))
    
    if variable_bins:
        binsize = 'variable'
        #print ('Rebin with {} bins of variable size'.format(numbins))
        if not withTT:
            #print ('Rebin delay stage positions')
            binList = histedges_equalN(Delays_fs_scan, numbins)
            pp_TT, binEdges, binNumber = binned_statistic(Delays_fs_scan, Pump_probe_scan, statistic='mean', bins=binList)
            pp_std, _, _ = binned_statistic(Delays_fs_scan, Pump_probe_scan, statistic='std', bins=binList)
        else:
            #print ('Rebin TT corrected delays')
            binList = histedges_equalN(Delays_corr_scan, numbins)
            pp_TT, binEdges, binNumber = binned_statistic(Delays_corr_scan, Pump_probe_scan, statistic='mean', bins=binList)
            pp_std, _, _ = binned_statistic(Delays_fs_scan, Pump_probe_scan, statistic='std', bins=binList)

        bin_centres = (binList[:-1] + binList[1:])/2
        Delay_fs_TT = np.copy(bin_centres)
    else: 
        binList = np.arange(min_delay, max_delay, binsize)
        #print ('Rebin with {} bins of {} fs'.format(len(binList), binsize))
        if not withTT:
            #print ('Rebin delay stage positions')
            pp_TT, binEdges, binNumber = binned_statistic(Delays_fs_scan, Pump_probe_scan, statistic='mean', bins=binList)
            pp_std, _, _ = binned_statistic(Delays_fs_scan, Pump_probe_scan, statistic='std', bins=binList)
        else:
            #print ('Rebin TT corrected delays')
            pp_TT, binEdges, binNumber = binned_statistic(Delays_corr_scan, Pump_probe_scan, statistic='mean', bins=binList)
            pp_std, _, _ = binned_statistic(Delays_corr_scan, Pump_probe_scan, statistic='std', bins=binList)

        bin_centres = (binList[:-1] + binList[1:])/2
        Delay_fs_TT = np.arange(min_delay + binsize/2, max_delay - binsize/2, binsize)
    
    count = []
    for index in range(len(bin_centres)):
        count.append(np.count_nonzero(binNumber == index+1))
    err_pp = pp_std/np.sqrt(np.array(count))
    
    print ('Rebin with {} bins of variable size'.format(numbins) if variable_bins else 'Rebin with {} bins of {} fs'.format(len(binList), binsize))
    print ('Rebin TT corrected delays' if withTT else 'Rebin delay stage positions')
    
    return Delay_fs_TT, pp_TT, pp_std, err_pp, count, binsize

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

def save_reduced_data_scanPP(reducedir, run_name, scan, D1p, D1u, D2p, D2u, D1p_raw, D1u_raw, D2p_raw, D2u_raw, I0p, I0u, delaystage, arrTimes, delaycorr, energy, energypad, rbk, c1, c2):
    readbacks = scan.readbacks
    setValues = scan.values
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                         "pump_1": D1p,
                                         "unpump_1": D1u,
                                         "pump_2": D2p,
                                         "unpump_2": D2u, 
                                         "pump_1_raw": D1p_raw, 
                                         "unpump_1_raw": D1u_raw, 
                                         "pump_2_raw": D2p_raw, 
                                         "unpump_2_raw": D2u_raw,
                                         "Izero_pump": I0p,
                                         "Izero_unpump": I0u,
                                         "Delays_stage" :delaystage, 
                                         "arrTimes": arrTimes,
                                         "Delays_corr": delaycorr,
                                         "energy": energy,
                                         "energypad": energypad,
                                         "readbacks": rbk, 
                                         "corr1": c1,
                                         "corr2": c2}
                                         
    np.save(reducedir+run_name+'/run_array', run_array)

################################################

def save_reduced_data_scanPP_noPair(reducedir, run_name, scan, D1p, D2p, D1p_raw, D2p_raw, I0p, delaystage, arrTimes, delaycorr, energy, energypad, rbk, c1, c2, lights, darks):
    readbacks = scan.readbacks
    setValues = scan.values
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,
                                         "pump_1": D1p,                                
                                         "pump_2": D2p,
                                         "pump_1_raw": D1p_raw, 
                                         "pump_2_raw": D2p_raw, 
                                         "Izero_pump": I0p,
                                         "Delays_stage" :delaystage, 
                                         "arrTimes": arrTimes,
                                         "Delays_corr": delaycorr,
                                         "energy": energy,
                                         "energypad": energypad,
                                         "readbacks": rbk, 
                                         "corr1": c1,
                                         "corr2": c2, 
                                         "lights": lights,
                                         "darks": darks}
                                         
    np.save(reducedir+run_name+'/run_array', run_array)

################################################

def save_reduced_data_scan_static(reducedir, run_name, scan, D1u, D2u, D1u_raw, D2u_raw, I0u, energy, energypad, rbk, c1, c2):
    readbacks = scan.readbacks
    setValues = scan.values
    run_array = {}
    run_array[run_name.split('-')[0]] = {"name": run_name,           
                                         "unpump_1": D1u,
                                         "unpump_2": D2u,
                                         "unpump_1_raw": D1u_raw, 
                                         "unpump_2_raw": D2u_raw,
                                         "Izero_unpump": I0u,
                                         "energy": energy, 
                                         "energypad": energypad, 
                                         "readbacks": rbk, 
                                         "corr1": c1,
                                         "corr2": c2}
                                         
    np.save(reducedir+run_name+'/run_array', run_array)

################################################

def use_two_diodes(d):
    d2 = defaultdict(list)
    for k,v in d.items():
        if k == 'name':
            continue
        else:
            d2[k] = d[k] + d[k]
    return d2

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


