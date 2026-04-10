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


class XANES_reduce():

    def __init__(self, pgroup, reducedir, run):
        self.pgroup = pgroup
        self.run = run
        self.reducedir = reducedir
    
        from sfdata import SFScanInfo
        self.jsonfile = ''     
        runname = []
        try:
            self.jsonfile = glob.glob('/sf/alvra/data/{}/raw/*{:04d}*/meta/scan.json'.format(self.pgroup, self.run))[0]
            runname.append(self.jsonfile.split('/')[6])
            print ("will reduce run {}: {}".format(self.run, runname))
            self.titlestring = self.pgroup + ' --- ' +str(self.run)    
        except IndexError:
            print ("Could not find run {} in pgroup {}".format(self.run, pgroup))
        

    def reduce_scan(self, TT, motor, diode1, diode2, Izero, saveflag, tolerance=0.05, shots2average=None):
        self.saveflag = saveflag
        self.TT = TT
        self.motor = motor
        self.diode1 = diode1
        self.diode2 = diode2
        self.Izero = Izero
        self.tolerance = tolerance
        self.shots2average = shots2average

        if self.TT == TT_PSEN124:
            self.TT = [channel_PSEN124_arrTimes, channel_PSEN124_arrTimesAmp]
            self.channel_arrTimes = channel_PSEN124_arrTimes
            self.channel_arrTimesAmp = channel_PSEN124_arrTimesAmp
        elif self.TT == TT_PSEN126:
            self.TT = [channel_PSEN126_arrTimes, channel_PSEN126_arrTimesAmp]
            self.channel_arrTimes = channel_PSEN126_arrTimes
            self.channel_arrTimesAmp = channel_PSEN126_arrTimesAmp
        elif self.TT == None:
            self.TT = [self.motor, self.motor]
            self.channel_arrTimes = self.motor
            self.channel_arrTimesAmp = self.motor

        self.channels_pp = [channel_Events, self.diode1, self.diode2, self.Izero, self.motor, channel_monoEnergy] + self.TT
        self.channels_all = self.channels_pp
    
        from sfdata import SFScanInfo
        runname = self.jsonfile.split('/')[-3]
        self.scan = SFScanInfo(self.jsonfile)
        self.rbk = self.scan.values

        unique = np.roll(np.diff(self.rbk, prepend=1)>self.tolerance, -1)
        unique[-1] = True
        if self.scan.parameters['Id'] == ['dummy']:
            unique = np.full(len(rbk), True)
        self.rbk = self.rbk[unique]

        p1, u1, p2, u2, Ip, Iu, ds, aT, dc, en, sv, c1, c2 = ([] for i in range(13))

        for i, step in enumerate(self.scan):
            check_files_and_data(step)
            check = get_filesize_diff(step)  
            go = unique[i]

            if check & go:
                clear_output(wait=True)
                filename = self.scan.files[i][0].split('/')[-1].split('.')[0]
                print (self.jsonfile)
                print ('Step {} of {}: Processing {}'.format(i+1, len(self.scan.files), filename))
    
                resultsPP, results, _, _ = load_data_compact_pump_probe(self.channels_pp, self.channels_all, step)

                p1.extend(resultsPP[self.diode1].pump)
                u1.extend(resultsPP[self.diode1].unpump)
                p2.extend(resultsPP[self.diode2].pump)
                u2.extend(resultsPP[self.diode2].unpump)
                Ip.extend(resultsPP[self.Izero].pump)
                Iu.extend(resultsPP[self.Izero].unpump)
                ds.extend(resultsPP[self.motor].pump)
                aT.extend(resultsPP[self.channel_arrTimes].pump)
                dc.extend(resultsPP[self.motor].pump + resultsPP[self.channel_arrTimes].pump)

                enshot = resultsPP[channel_monoEnergy].pump
                en.extend(enshot)
                sv = np.pad(sv, (0,len(enshot)), constant_values=(self.rbk[i]))

                pearsonr1 = pearsonr(resultsPP[self.diode1].unpump,resultsPP[self.Izero].unpump)[0]
                pearsonr2 = pearsonr(resultsPP[self.diode2].unpump,resultsPP[self.Izero].unpump)[0]

                c1.append(pearsonr1)
                c2.append(pearsonr2)

                print ("correlation Diode1 (dark shots) = {}".format(pearsonr1))
                print ("correlation Diode2 (dark shots) = {}".format(pearsonr2))

        u1_raw = u1
        p1_raw = p1
        u2_raw = u2
        p2_raw = p2

        if self.saveflag:
            os.makedirs(self.reducedir+runname, mode=0o775, exist_ok=True)
            save_reduced_data_scanPP(self.reducedir, runname, self.scan, p1, u1, p2, u2, p1_raw, u1_raw, p2_raw, u2_raw, Ip, Iu, ds, aT, dc, en, sv, self.rbk, c1, c2)

        self.pump_1       = np.array(p1)
        self.unpump_1     = np.array(u1)
        self.pump_2       = np.array(p2)
        self.unpump_2     = np.array(u2)
        self.Izero_pump   = np.array(Ip)
        self.Izero_unpump = np.array(Iu)
        self.Delays_stage = np.array(ds)
        self.arrTimes     = np.array(aT)
        self.Delays_corr  = np.array(dc)
        self.energy       = np.array(en)
        self.scanvar      = np.array(sv)
        self.corr1        = np.array(c1)
        self.corr2        = np.array(c2)

        print ('----------------------------')
        print ('Loaded {} total on/off pairs'.format(len(self.pump_1)))

        res = {'pump_1': self.pump_1, 'unpump_1': self.unpump_1, 'pump_2': self.pump_2, 'unpump_2': self.unpump_2, 'Izero_pump': self.Izero_pump, 'Izero_unpump': self.Izero_unpump, 'Delays_stage': self.Delays_stage, 'arrTimes': self.arrTimes, 'Delays_corr': self.Delays_corr, 'energy': self.energy, 'scanvar': self.scanvar, 'rbk': self.rbk, 'corr1': self.corr1, 'corr2': self.corr2}

        return (res)

    
    def Plot_correlations(self, lowlim = 0.992):
        
        fig, ((ax1, ax3)) = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
        plt.suptitle(self.titlestring)

        self.xlabel = self.scan.parameters['name'][0]
        try:
            self.xunits = self.scan.parameters['units'][0]
        except:
            pass

        if self.scan.parameters['Id'] == ['dummy']:
            self.rbk = np.arange(1, len(scan.rbk)+1)
            self.xlabel = 'Acq number'
            self.xunits = 'N/A'

        ax1.plot(self.rbk, self.corr1, label='diode1 run{:04d}'.format(self.run))
        ax3.plot(self.rbk, self.corr2, label='diode2 run{:04d}'.format(self.run))
        ax1.legend()
        ax1.set_xlabel("{} ({})".format(self.xlabel, self.xunits))
        ax1.grid()
        ax3.legend()
        ax3.set_xlabel("{} ({})".format(self.xlabel, self.xunits))
        ax3.grid()
        ax1.set_ylim(lowlim,1)
        ax3.set_ylim(lowlim,1)
        plt.show()

    def Rebin_scanvar(self, pump, unpump, Ipump, Iunpump, threshold):
    
        ordered = np.argsort(self.scanvar)
        peaks, what = find_peaks(np.diff(self.scanvar[ordered]))
        
        pump = pump[ordered]
        unpump = unpump[ordered]
        Ipump = Ipump[ordered]
        Iunpump = Iunpump[ordered]

        starts = np.append(0, peaks)
        ends = np.append(peaks, None)

        pp, GS, ES, err_pp, err_GS, err_ES = ([] for i in range(6))
        filtered = []

        for s, e in zip(starts, ends):
            pump_bin    = pump[s:e]
            unpump_bin  = unpump[s:e]
            Izero_p_bin = Ipump[s:e]
            Izero_u_bin = Iunpump[s:e]
            
            pump_bin = pump_bin / Izero_p_bin
            unpump_bin = unpump_bin / Izero_u_bin

            thresh   = (Izero_p_bin>threshold) & (Izero_u_bin>threshold)
            pp_shot = pump_bin[thresh] - unpump_bin[thresh]

            #pp_shot = np.log10(pump_ebin[thresh]/unpump_ebin[thresh])
            #pp_shot = np.log10(pump_ebin/unpump_ebin)
            #pp_shot = pump_ebin - unpump_ebin

            GS.append(np.nanmean(unpump_bin))
            ES.append(np.nanmean(pump_bin))
            pp.append(np.nanmean(pp_shot))
            err_GS.append(np.nanstd(unpump_bin)/np.sqrt(len(unpump_bin)))
            err_ES.append(np.nanstd(pump_bin)/np.sqrt(len(pump_bin)))
            err_pp.append(np.nanstd(pp_shot)/np.sqrt(len(pp_shot)))

        return np.array(pp), np.array(GS), np.array(ES), np.array(err_pp), np.array(err_GS), np.array(err_ES)


    def Plot_scan_simple(self, threshold=0, rebin=False):

        runname = 'run{:04d}'.format(self.run) 
        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        plt.suptitle(self.titlestring)

        if self.scan.parameters['Id'] == ['dummy']:
            self.rbk = np.arange(1, len(scan.readbacks)+1)
        
        pp1, GS1, ES1, err_pp1, err_GS1, err_ES1 = self.Rebin_scanvar(self.pump_1, self.unpump_1, self.Izero_pump, self.Izero_unpump, threshold)
        pp2, GS2, ES2, err_pp2, err_GS2, err_ES2 = self.Rebin_scanvar(self.pump_2, self.unpump_2, self.Izero_pump, self.Izero_unpump, threshold)

        if self.scan.parameters['Id'] == ['dummy']:
            self.rbk = np.arange(1, len(self.scan.readbacks)+1)
            self.xlabel = 'Acq number'
            self.xunits = 'N/A'

        ax1.plot(self.rbk, ES1, label='ON 1 {}'.format(runname), color='royalblue', alpha = 0.8)
        ax1.fill_between(self.rbk, ES1-err_ES1, ES1+err_ES1, color='lightblue')
        ax1.plot(self.rbk, GS1, label='OFF 1 {}'.format(runname), color='orange', alpha = 0.8)
        ax1.fill_between(self.rbk, GS1-err_GS1, GS1+err_GS1, color='navajowhite')
        ax1.set_xlabel("{} ({})".format(self.xlabel, self.xunits))
        ax1.legend()

        ax2.plot(self.rbk, pp1, label='pp 1 {}'.format(runname), color='green', marker='.')
        ax2.fill_between(self.rbk, pp1-err_pp1, pp1+err_pp1, color='lightgreen')
        ax2.set_xlabel("{} ({})".format(self.xlabel, self.xunits))
        ax2.legend()

        ax3.plot(self.rbk, ES2, label='ON 2 {}'.format(runname), color='royalblue', alpha = 0.8)
        ax3.fill_between(self.rbk, ES2-err_ES2, ES2+err_ES2, color='lightblue')
        ax3.plot(self.rbk, GS2, label='OFF 2 {}'.format(runname), color='orange', alpha = 0.8)
        ax3.fill_between(self.rbk, GS2-err_GS2, GS2+err_GS2, color='navajowhite')
        ax3.set_xlabel("{} ({})".format(self.xlabel, self.xunits))
        ax3.legend()

        ax4.plot(self.rbk, pp2, label='pp 2 {}'.format(runname), color='green', marker='.')
        ax4.fill_between(self.rbk, pp2-err_pp2, pp2+err_pp2, color='lightgreen')
        ax4.set_xlabel("{} ({})".format(self.xlabel, self.xunits))
        ax4.legend()

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        plt.show()


class XANES_analysis():

    def __init__(self, pgroup, loaddir, runlist, withTT=False, t0_offset=None):
        if t0_offset == None:
            t0_offset = [0]*len(runlist)
        self.pgroup = pgroup
        self.runlist = runlist
        self.loaddir = loaddir
        self.withTT = withTT
        self.t0_offset = t0_offset

        runname = []
        for run in self.runlist:
            if glob.glob(loaddir + '/run{:04d}*/*run_array*'.format(run)):
                runname.append(run)
            else:
                print ("Could not find run {} reduced in pgroup {}".format(run, pgroup))

        print ("will load run(s): {}".format(runname))
        self.runlist = runname


    def load_reduced_data(self, switch_diodes=False):
        self.switch_diodes = switch_diodes
        from collections import defaultdict
        self.titlestring = self.pgroup + ' --- ' +str(self.runlist) + ' --- diode1'
        if switch_diodes:
            self.titlestring = self.pgroup + ' --- ' +str(self.runlist) + ' --- diode2'

        self.datadict = defaultdict(list)
        for i, run in enumerate(self.runlist):
            offset = np.asarray(self.t0_offset[i])
            #data = {}
            file = sorted(glob.glob(self.loaddir + '/run{:04d}*/*run_array*'.format(run)))
            run_array = np.load(file[0], allow_pickle=True).item()
            for k, v in run_array.items():
                for key, value in v.items():
                    #data[key] = value
                    if key =="name" or key=='readbacks':
                        self.datadict[key].append(value)
                    else:
                        if "Delay" in key:
                            self.datadict[key].extend(value+offset)
                            if offset != 0:
                                print ("Run {}, {} offset by {} fs".format(run, key, offset))
                        else:
                            self.datadict[key].extend(value)

        if switch_diodes:
            d3 = self.datadict.copy()
            for k in self.datadict.keys():
                if "1" in k:
                    d3[k] = self.datadict[k.replace('1', '2')]
                if "2" in k:
                    d3[k] = self.datadict[k.replace('2', '1')]
            self.datadict.update(d3)

        self.pump_1 = np.array(self.datadict['pump_1'])
        self.pump_2 = np.array(self.datadict['pump_2'])
        self.unpump_1 = np.array(self.datadict['unpump_1'])
        self.unpump_2 = np.array(self.datadict['unpump_2'])
        self.Izero_pump = np.array(self.datadict['Izero_pump'])
        self.Izero_unpump = np.array(self.datadict['Izero_unpump'])   
        self.Delays_stage = np.array(self.datadict['Delays_stage'])
        self.arrTimes = np.array(self.datadict['arrTimes'])
        self.Delays_corr = np.array(self.datadict['Delays_corr'])
        self.energy = np.array(self.datadict['energy'])
        self.scanvar = np.array(self.datadict['scanvar'])
        self.rbk = np.array(self.datadict['readbacks'], dtype=object)
        self.corr1 = np.array(self.datadict['corr1'])
        self.corr2 = np.array(self.datadict['corr2'])
        
        return self.datadict, self.titlestring
            

    def hist_reduced_data(self):
        if self.withTT:
            Delays = self.Delays_corr
        else:
            Delays = self.Delays_stage
        
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(10, 3), constrained_layout=True)
        fig.suptitle(self.titlestring)
        ax1.title.set_text('Izero')
        ax1.hist(self.Izero_pump, bins = 200, alpha=0.5, label = 'on')
        ax1.hist(self.Izero_unpump, bins = 200, alpha=0.5, label = 'off')
        ax1.legend(loc='best')
        ax2.title.set_text('Fluo')
        ax2.hist(self.pump_1, bins = 200, alpha=0.5, label = 'on')
        ax2.hist(self.unpump_1, bins = 200, alpha=0.5, label = 'off')
        ax2.legend(loc='best')
        ax3.title.set_text('Delays')
        ax3.hist(Delays, bins = 100)
        ax3.set_xlim(min(Delays), max(Delays))
        ax4.title.set_text('Energies')
        ax4.hist(self.energy, bins=100)
        ax4.set_xlim(min(self.energy), max(self.energy))
        ax5.title.set_text('Scanvar')
        ax5.hist(self.scanvar, bins=100)
        ax5.set_xlim(min(self.scanvar), max(self.scanvar))
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        ax5.grid()        
        plt.show()

        if self.withTT:
            print('Time delay axis rebinned with TT data')
        else:
            print('Time delay axis rebinned with delay stage data')
        
    def create_filter_condition_PP(self, p, u):

        qnt_low_p  = np.nanquantile(p, 0.5 - self.quantile/2)
        qnt_high_p = np.nanquantile(p, 0.5 + self.quantile/2)
        qnt_low_u  = np.nanquantile(u, 0.5 - self.quantile/2)
        qnt_high_u = np.nanquantile(u, 0.5 + self.quantile/2)

        filter_p_l = p > qnt_low_p
        filter_p_h = p < qnt_high_p
        filter_u_l = u > qnt_low_u
        filter_u_h = u < qnt_high_u

        condition = filter_p_l & filter_p_h & filter_u_l & filter_u_h
        return condition

    def Rebin_and_filter_scanvar(self, quantile, threshold, n_sigma):

        self.quantile = quantile
        self.threshold = threshold
        self.n_sigma = n_sigma
    
        ordered = np.argsort(self.scanvar)
        peaks, what = find_peaks(np.diff(np.array(self.scanvar)[ordered]))
        
        pump = self.pump_1[ordered]
        unpump = self.unpump_1[ordered]
        Ipump = self.Izero_pump[ordered]
        Iunpump = self.Izero_unpump[ordered]

        starts = np.append(0, peaks)
        ends = np.append(peaks, None)

        pp, GS, ES, err_pp, err_GS, err_ES = ([] for i in range(6))
        self.surviving = []

        for s, e in zip(starts, ends):
            pump_bin    = pump[s:e]
            unpump_bin  = unpump[s:e]
            Izero_p_bin = Ipump[s:e]
            Izero_u_bin = Iunpump[s:e]

            thresh   = (Izero_p_bin>threshold) & (Izero_u_bin>threshold)
            filterIp = Izero_p_bin > (np.nanmedian(Izero_p_bin) - n_sigma*np.std(Izero_p_bin))
            filterIu = Izero_u_bin > (np.nanmedian(Izero_u_bin) - n_sigma*np.std(Izero_u_bin))

            pump_bin    = pump_bin[filterIp & filterIu & thresh]
            unpump_bin  = unpump_bin[filterIp & filterIu & thresh]
            Izero_p_bin = Izero_p_bin[filterIp & filterIu & thresh]
            Izero_u_bin = Izero_u_bin[filterIp & filterIu & thresh]
            
            ratio_p = pump_bin/Izero_p_bin
            ratio_u = unpump_bin/Izero_u_bin
        
            filter_bin = self.create_filter_condition_PP(ratio_p, ratio_u)
            self.surviving.append(np.sum(filter_bin))

            pump_filter    = pump_bin[filter_bin]
            unpump_filter  = unpump_bin[filter_bin]
            Izero_p_filter = Izero_p_bin[filter_bin]
            Izero_u_filter = Izero_u_bin[filter_bin]

            p_filter = pump_filter / Izero_p_filter
            u_filter = unpump_filter / Izero_u_filter
            #pp_shot = np.log10(pump_bin/unpump_bin)
            pp_shot = p_filter - u_filter

            GS.append(np.nanmean(u_filter))
            ES.append(np.nanmean(p_filter))
            pp.append(np.nanmean(pp_shot))
            err_GS.append(np.nanstd(u_filter)/np.sqrt(len(u_filter)))
            err_ES.append(np.nanstd(p_filter)/np.sqrt(len(p_filter)))
            err_pp.append(np.nanstd(pp_shot)/np.sqrt(len(pp_shot)))

        self.GS = np.array(GS)
        self.ES = np.array(ES)
        self.pp = np.array(pp)
        self.err_GS = np.array(err_GS)
        self.err_ES = np.array(err_ES)
        self.err_pp = np.array(err_pp)

        print ('{} shots out of {} survived'.format(np.sum(self.surviving), len(self.pump_1)))

   
    def plot_filtered_data(self, data):

        rbk = np.array(self.rbk[0], dtype=float)
    
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        plt.suptitle(self.titlestring)

        xlabel = data['meta'][2][1][0]
        xunits = data['meta'][5][1][0]

        print (np.shape(rbk))
            
        ax1.fill_between(rbk, self.ES-self.err_ES, self.ES+self.err_ES, label='ON', color='royalblue', alpha = 0.8)
        ax1.fill_between(rbk, self.GS-self.err_GS, self.GS+self.err_GS, label='OFF',color='orange', alpha = 0.8)
        ax3.fill_between(rbk, self.pp-self.err_pp, self.pp+self.err_pp, label='pump probe',color='lightgreen')
        ax3.plot(rbk, self.pp, color='green', marker='.')
        
        ax1.set_xlabel("{} ({})".format(xlabel, xunits))
        ax1.set_ylabel ("XAS Diode")
        ax1.set_title('XAS (fluo)')
        ax1.legend(loc="best")
        ax1.grid()
        
        ax3.set_xlabel("{} ({})".format(xlabel, xunits))
        ax3.set_ylabel ("DeltaXAS")
        ax3.set_title('pump probe')
        ax3.legend(loc="best")
        ax3.grid()
        
        plt.show()


        
        
            
      

            





