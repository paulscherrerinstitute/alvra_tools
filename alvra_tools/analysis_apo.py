import numpy as np
from tqdm import tqdm
import pandas as pd
import os, glob, copy, numbers
import textwrap
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d    
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation

from alvra_tools.load_data import *
from alvra_tools.utils import *
from lmfit.models import PseudoVoigtModel


def get_jsonfile_from_run(pgroup, run, path='raw'):
    import glob
    jsonfile = glob.glob('/sf/alvra/data/{}/{}/*{:04d}*/meta/scan.json'.format(pgroup, path, run))[0]
    return jsonfile

def get_meta(filedict):
    m = {}
    with h5py.File(filedict, 'r') as d:
        m['xlabel'] = d['meta']['xlabel'].asstr()[()] if 'xlabel' in d['meta'] else None
        m['units']  = d['meta']['units'].asstr()[()] if 'units' in d['meta'] else None
    return m
    
def generate_filelistBS(pgroup, run):
    filelist = sorted(glob.glob('/sf/alvra/data/{}/res/processed/run{:04d}*/data/*acq*BSDATA*'.format(pgroup, run)))
    return filelist

def get_array(datadict, key):
        return np.asarray(datadict.get(key, []))

def build_scanvar(dict_item, rbk_value):
    scanvar = np.pad(scanvar, (0,len(dict_item)), constant_values=(rbk_value))
    return scanvar

def load_step_scan(pgroup, run, channel_list, quantile, norm=None, what='allShots'):
    filelist = generate_filelistBS(pgroup, run)
    if not filelist:
        print ('===> run{:04d} not found in res/processed/ folder! <==='.format(run))
        return None, None, None
    else: 
        readbacks = []
        channels = [[] for _ in range(len(channel_list))]
        meta = get_meta(filelist[0])
        meta['title'] = pgroup + ' --- ' + str(filelist[0].split('/')[-3])
        for i,step in enumerate(tqdm(filelist)):
            with h5py.File(step, 'r') as dd:
                #readbacks.extend(dd['meta']['readback_value'][:])
                readbacks.extend(dd['meta']['scan_value'][:])
                for j, ch in enumerate(channel_list):
                    if norm:
                        df = pd.DataFrame(np.array(dd[what][ch])/np.array(dd[what][str(norm)]))
                    else:
                        df = pd.DataFrame(np.array(dd[what][ch]))
                    channels[j].append(np.nanquantile(df, [0.5, 0.5 - quantile/2, 0.5 + quantile/2]))
    return np.asarray(channels), np.asarray(readbacks), meta

def merge_steps(pgroup, run, diode, offset):
    from collections import defaultdict
    filelist = generate_filelistBS(pgroup, run)
    meta = get_meta(filelist[0])
    d = defaultdict(list)
    sv=[]
    offset_channels = set()
    for i, acq in enumerate(tqdm(filelist)):
        with h5py.File(acq, 'r') as dd:
            #rbk_value = dd['meta']['readback_value']
            rbk_value = dd['meta']['scan_value']            
            d["readbacks"].extend(rbk_value)
            sv.extend(np.full(len(dd['OnOff']['diode1_pump']), rbk_value))
            for name, group in dd.items():
                if name == 'meta':
                    continue
                for key, value in group.items():
                    if "diode" in key and diode is not None:
                        if diode not in key:
                            continue
                        #key = key.replace(diode, "diode")

                    if "delay" in key or "SLAAR" in key:
                        d[key].extend(value + offset)
                        offset_channels.add(key)
                    else:
                        d[key].extend(value)
    if offset != 0:
        for key in offset_channels:
            print ("Run {}, {} offset by {} fs".format(run, key, offset))
    d["scanvar"] = sv

    return d, meta

def merge_multiple_runs(pgroup, runlist, diode=None, t0_offset=None):
    from collections import defaultdict
    if t0_offset == None:
        t0_offset = [0]*len(runlist)
    d = defaultdict(list)
    readbacks = []
    for i, r in enumerate(runlist):
        offset = np.asarray(t0_offset[i])
        print ('Processing run {}'.format(r))
        drun, meta = merge_steps(pgroup, r, diode, offset)
        for key, value in drun.items():
            d[key].extend(value)
    meta['title'] = pgroup + ' --- ' + str(runlist)

    return d, meta
    
def unwrap_data(data, channel_index=0):
    if data is None:
        return None, None, None
    else:
        data = data[channel_index]
        print ("Loaded {} steps".format(len(data)))
        signal   = data[:,0]
        err_low  = data[:,1]
        err_high = data[:,2]    
        return signal, err_low, err_high

def plot_merged_data(data, meta, Signal, Izero, TT, withTT=False, bins=100, figsize=(12, 3)):
    
    Izero_pump   = get_array(data, '{}_pump'.format(Izero))
    Izero_unpump = get_array(data, '{}_unpump'.format(Izero))
    pump_1       = get_array(data, '{}_pump'.format(Signal))
    unpump_1     = get_array(data, '{}_unpump'.format(Signal))
    delay_stage  = get_array(data, 'delay_motor_pump')
    arr_times    = get_array(data, 'arrTimes{}_pump'.format(TT))
    energy       = get_array(data, 'energy_pump')
    rbk          = get_array(data, 'readbacks')

    delay_corr = delay_stage + arr_times if len(arr_times) else delay_stage
    delays = delay_corr if withTT else delay_stage

    title = meta.get('title', '')
    title = title + ' --- ' + Signal
    title = "\n".join(textwrap.wrap(title))

    fig, axes = plt.subplots(1, 6, figsize=figsize, constrained_layout=True)
    fig.suptitle(title, fontsize=12)

    def hist_pair(ax, data_on, data_off, title):
        ax.set_title(title)
        if len(data_on) and not np.isnan(data_on).all():
            ax.hist(data_on, bins=bins, alpha=0.6, label='on')
        if len(data_off) and not np.isnan(data_off).all():
            ax.hist(data_off, bins=bins, alpha=0.6, label='off')
        ax.legend()
        ax.grid(True)

    # 1. Izero
    hist_pair(axes[0], Izero_pump, Izero_unpump, 'Izero')
    
    # 2. Fluorescence
    hist_pair(axes[1], pump_1, unpump_1, 'Signal')
    
    # 3. Delays
    axes[2].set_title('Delays')
    if len(delays) and not np.isnan(delays).all():
        axes[2].hist(delays, bins=bins)
        axes[2].set_xlim(np.nanmin(delays), np.nanmax(delays))
    axes[2].grid(True)

    # 4. Energy
    axes[3].set_title('Energy')
    if len(energy) and not np.isnan(energy).all():
        axes[3].hist(energy, bins=min(len(rbk), bins))
        axes[3].set_xlim(np.nanmin(energy), np.nanmax(energy))
    axes[3].grid(True)

    # 5. Readbacks
    axes[4].set_title('Readbacks')
    if len(rbk) and not np.isnan(rbk).all():
        axes[4].hist(rbk, bins=bins)
        axes[4].set_xlim(np.nanmin(rbk), np.nanmax(rbk))
    axes[4].set_xlabel('{} ({})'.format(meta.get('xlabel'), meta.get('units')))
    axes[4].grid(True)

    # 6. Timetool
    axes[5].set_title('TT')
    if len(arr_times) and not np.isnan(arr_times).all():
        axes[5].hist(arr_times, bins=bins)
        axes[5].set_xlim(np.nanmin(arr_times), np.nanmax(arr_times))
    axes[5].grid(True)
                                                                                                                                                                                                        
    plt.show()

def correlation_filter(*arrays, quantile):
    signals = arrays[0::2]
    normals = arrays[1::2]
    normed = [s / n for s,n in zip(signals, normals)]

    conditions = []
    for arr in normed:
        q_low  = np.nanquantile(arr, 0.5 - quantile/2)
        q_high = np.nanquantile(arr, 0.5 + quantile/2)
        conditions.append((arr > q_low) & (arr < q_high))
    mask = np.logical_and.reduce(conditions)
    return tuple(a[mask] for a in arrays)

def create_corr_condition(pump, unpump, quantile):

    qnt_low_p  = np.nanquantile(pump, 0.5 - quantile/2)
    qnt_high_p = np.nanquantile(pump, 0.5 + quantile/2)
    qnt_low_u  = np.nanquantile(unpump, 0.5 - quantile/2)
    qnt_high_u = np.nanquantile(unpump, 0.5 + quantile/2)

    filtervals_p_l = pump >= qnt_low_p
    filtervals_p_h = pump <= qnt_high_p
    filtervals_u_l = unpump >= qnt_low_u
    filtervals_u_h = unpump <= qnt_high_u

    condition = filtervals_p_l & filtervals_p_h & filtervals_u_l & filtervals_u_h
    return condition

def Rebin_with_scanvar_and_filter(data, quantile, signal, izero, TT, YAGscan=False, withTT=False, threshold=0):

    print ('{}_pump'.format(izero), '{}_pump'.format(signal))

    Izero_pump   = get_array(data, '{}_pump'.format(izero))
    Izero_unpump = get_array(data, '{}_unpump'.format(izero))
    pump         = get_array(data, '{}_pump'.format(signal))
    unpump       = get_array(data, '{}_unpump'.format(signal))
    arrTimes     = get_array(data, 'arrTimes{}_pump'.format(TT))
    scanvar      = get_array(data, 'scanvar')

    ordered = np.argsort(np.asarray(scanvar))
    peaks,_ = find_peaks(np.diff(scanvar[ordered]))

    if withTT:
        scanvar = scanvar + arrTimes

    Izero_pump = Izero_pump[ordered]
    Izero_unpump = Izero_unpump[ordered]
    pump = pump[ordered]
    unpump = unpump[ordered]
    scanvar = scanvar[ordered]

    Izero_mask = (Izero_pump > threshold) & (Izero_unpump > threshold)
    Izero_pump   = Izero_pump[Izero_mask]
    Izero_unpump = Izero_unpump[Izero_mask]
    pump         = pump[Izero_mask]
    unpump       = unpump[Izero_mask]
    scanvarf     = scanvar[Izero_mask]

    starts = np.concatenate(([0], peaks+1))
    ends   = np.concatenate((peaks+1, [len(scanvar)]))
    nbins = len(starts)

    GS, ES, pp, err_GS, err_ES, err_pp, scanvar_rebin = (np.empty(nbins) for _ in range(7))
    howmany = []
    
    for i, (s, e) in enumerate(zip(starts, ends)):
        pump_bin = pump[s:e]
        unpump_bin = unpump[s:e]
        Izero_pump_bin = Izero_pump[s:e]
        Izero_unpump_bin = Izero_unpump[s:e]
        scanvar_bin = scanvarf[s:e]

        ratio_p = np.divide(pump_bin, Izero_pump_bin)
        ratio_u = np.divide(unpump_bin, Izero_unpump_bin)
    
        correlation_mask = create_corr_condition(ratio_p, ratio_u, quantile)
        pump_bin = pump_bin[correlation_mask]
        unpump_bin = unpump_bin[correlation_mask]
        Izero_pump_bin = Izero_pump_bin[correlation_mask]
        Izero_unpump_bin = Izero_unpump_bin[correlation_mask]
        scanvar_bin = scanvar_bin[correlation_mask]
        howmany.append(len(scanvar_bin))

        pp_bin = pump_bin/Izero_pump_bin - unpump_bin/Izero_unpump_bin
        if YAGscan:
            pp_bin = -np.log10(pump_bin/unpump_bin)/Izero_pump_bin
        
        scanvar_rebin[i] = np.average(scanvar_bin)
        GS[i] = np.nanmedian(unpump_bin/Izero_unpump_bin)
        ES[i] = np.nanmedian(pump_bin/Izero_pump_bin)
        pp[i] = np.nanmedian(pp_bin)

        err_GS[i] = median_abs_deviation(unpump_bin/Izero_unpump_bin)
        err_ES[i] = median_abs_deviation(pump_bin/Izero_pump_bin)
        err_pp[i] = median_abs_deviation(pp_bin)    
    
    print ('{} shots out of {} survived'.format(np.sum(howmany), len(scanvar)))
    results = {'GS': GS, 'ES':ES, 'pp': pp, 'err_GS': err_GS, 'err_ES': err_ES, 'err_pp': err_pp, 'scanvar_rebin': scanvar_rebin, 'howmany': howmany}   
    return results

def Rebin_and_filter(data, binsize, minvalue, maxvalue, quantile, signal, izero, TT, diode='diode1', YAGscan=False, withTT=False, threshold=0, numbins=None):

    Izero_pump   = get_array(data, '{}_pump'.format(izero))
    Izero_unpump = get_array(data, '{}_unpump'.format(izero))
    pump         = get_array(data, '{}_pump'.format(signal))
    unpump       = get_array(data, '{}_unpump'.format(signal))
    arrTimes     = get_array(data, 'arrTimes{}_pump'.format(TT))
    scanvar      = get_array(data, 'scanvar')
    
    if np.sum(scanvar) == 0:
        scanvar  = get_array(data, 'delay_motor_pump') 
    
    if withTT:
        scanvar = scanvar + arrTimes

    binList = np.arange(minvalue, maxvalue, binsize)
    if numbins:
        binList = histedges_equalN(scanvar, numbins)

    bin_centres = (binList[:-1] + binList[1:])/2
    scanvar_rebin = np.arange(minvalue + binsize/2, maxvalue - binsize/2, binsize)
    nbins = len(bin_centres)
    if numbins:
        scanvar_rebin = bin_centres

    Izero_mask = (Izero_pump > threshold) & (Izero_unpump > threshold)
    Izero_pump   = Izero_pump[Izero_mask]
    Izero_unpump = Izero_unpump[Izero_mask]
    pump         = pump[Izero_mask]
    unpump       = unpump[Izero_mask]
    scanvarf     = scanvar[Izero_mask]

    GS, ES, pp, err_GS, err_ES, err_pp, scanvar_rebin = (np.empty(nbins) for _ in range(7))
    howmany = []

    for i in range(len(bin_centres)):
        cond1 = scanvarf >= binList[i]
        cond2 = scanvarf < binList[i+1]
    
        idx = np.where(cond1 & cond2)[0]

        pump_bin = pump[idx]
        unpump_bin = unpump[idx]
        Izero_pump_bin = Izero_pump[idx]
        Izero_unpump_bin = Izero_unpump[idx]
        scanvar_bin = scanvarf[idx]

        ratio_p = np.divide(pump_bin, Izero_pump_bin)
        ratio_u = np.divide(unpump_bin, Izero_unpump_bin)
    
        correlation_mask = create_corr_condition(ratio_p, ratio_u, quantile)
        pump_bin = pump_bin[correlation_mask]
        unpump_bin = unpump_bin[correlation_mask]
        Izero_pump_bin = Izero_pump_bin[correlation_mask]
        Izero_unpump_bin = Izero_unpump_bin[correlation_mask]
        scanvar_bin = scanvar_bin[correlation_mask]
        howmany.append(len(scanvar_bin))

        pp_bin = pump_bin/Izero_pump_bin - unpump_bin/Izero_unpump_bin
        if YAGscan:
            pp_bin = -np.log10(pump_bin/unpump_bin)/Izero_pump_bin

        scanvar_rebin[i] = np.average(scanvar_bin)
        GS[i] = np.nanmedian(unpump_bin/Izero_unpump_bin)
        ES[i] = np.nanmedian(pump_bin/Izero_pump_bin)
        pp[i] = np.nanmedian(pp_bin)

        err_GS[i] = median_abs_deviation(unpump_bin/Izero_unpump_bin)
        err_ES[i] = median_abs_deviation(pump_bin/Izero_pump_bin)
        err_pp[i] = median_abs_deviation(pp_bin)
        
    print ('{} shots out of {} survived'.format(np.sum(howmany), len(scanvar)))
    results = {'GS': GS, 'ES':ES, 'pp': pp, 'err_GS': err_GS, 'err_ES': err_ES, 'err_pp': err_pp, 'scanvar_rebin': scanvar_rebin, 'howmany': howmany}   
    return results

def Rebin_and_filter_2Dscans(data, binsize, minvalue, maxvalue, quantile, signal, izero, TT, withTT=False, threshold=0, numbins=None):
    
    Izero_pump   = get_array(data, '{}_pump'.format(izero))
    Izero_unpump = get_array(data, '{}_unpump'.format(izero))
    pump         = get_array(data, '{}_pump'.format(signal))
    unpump       = get_array(data, '{}_unpump'.format(signal))
    arrTimes     = get_array(data, 'arrTimes{}_pump'.format(TT))
    scanvar_e    = get_array(data, 'scanvar')
    scanvar_t    = get_array(data, 'delay_motor_pump')
    
    if withTT:
        scanvar_t = scanvar_t + arrTimes

    binList = np.arange(minvalue, maxvalue, binsize)
    if numbins:
        binList = histedges_equalN(scanvar_t, numbins)

    bin_centres = (binList[:-1] + binList[1:])/2
    delay_rebin = np.arange(minvalue + binsize/2, maxvalue - binsize/2, binsize)
    nbinsY = len(bin_centres)
    if numbins:
        delay_rebin = bin_centres
    
    ordered = np.argsort(np.asarray(scanvar_e))
    peaks,_ = find_peaks(np.diff(scanvar_e[ordered]))

    Izero_pump = Izero_pump[ordered]
    Izero_unpump = Izero_unpump[ordered]
    pump = pump[ordered]
    unpump = unpump[ordered]
    scanvar_t = scanvar_t[ordered]
    scanvar_e = scanvar_e[ordered]

    starts = np.concatenate(([0], peaks+1))
    ends   = np.concatenate((peaks+1, [len(scanvar_e)]))
    nbinsX = len(starts)
    
    scanvar_rebin = np.empty(nbinsX)
    GS, ES, pp, err_GS, err_ES, err_pp = (np.empty((nbinsX, nbinsY)) for _ in range(6))
    howmany = []
    
    for i, (s, e) in enumerate(zip(starts, ends)):
        pump_ebin = pump[s:e]
        unpump_ebin = unpump[s:e]
        Izero_pump_ebin = Izero_pump[s:e]
        Izero_unpump_ebin = Izero_unpump[s:e]
        scanvar_t_in_tbin = scanvar_t[s:e]
        scanvar_e_in_tbin = scanvar_e[s:e]

        scanvar_rebin[i] = np.nanmean(scanvar_e_in_tbin)

        Izero_mask = (Izero_pump_ebin > threshold) & (Izero_unpump_ebin > threshold)
        
        pump_ebin         = pump_ebin[Izero_mask]
        unpump_ebin       = unpump_ebin[Izero_mask]
        Izero_pump_ebin   = Izero_pump_ebin[Izero_mask]
        Izero_unpump_ebin = Izero_unpump_ebin[Izero_mask]
        scanvar_tbin      = scanvar_t_in_tbin[Izero_mask]
        scanvar_ebin      = scanvar_e_in_tbin[Izero_mask]

        for j in range(len(bin_centres)):
            cond1 = scanvar_tbin >= binList[j]
            cond2 = scanvar_tbin < binList[j+1]
            
            idx = np.where(cond1 & cond2)[0]
            
            if numbins:
                delay_rebin[j]  = np.nanmean(scanvar_tbin)[idx]

            pump_tebin = pump_ebin[idx]
            unpump_tebin = unpump_ebin[idx]
            Izero_pump_tebin = Izero_pump_ebin[idx]
            Izero_unpump_tebin = Izero_unpump_ebin[idx]
            scanvar_tebin = scanvar_tbin[idx]
            scanvar_te = scanvar_ebin[idx]

            ratio_p = np.divide(pump_tebin, Izero_pump_tebin)
            ratio_u = np.divide(unpump_tebin, Izero_unpump_tebin)

            correlation_mask = create_corr_condition(ratio_p, ratio_u, quantile)
            pump_tebin = pump_tebin[correlation_mask]
            unpump_tebin = unpump_tebin[correlation_mask]
            Izero_pump_tebin = Izero_pump_tebin[correlation_mask]
            Izero_unpump_tebin = Izero_unpump_tebin[correlation_mask]
            scanvar_tebin = scanvar_tebin[correlation_mask]

            howmany.append(len(scanvar_tebin))
    
            pp_tebin = pump_tebin/Izero_pump_tebin - unpump_tebin/Izero_unpump_tebin

            #delay_rebin[j] = np.average(scanvar_tebin)

            GS[i, j] = np.nanmedian(unpump_tebin/Izero_unpump_tebin)
            ES[i, j] = np.nanmedian(pump_tebin/Izero_pump_tebin)
            pp[i, j] = np.nanmedian(pp_tebin)
            err_GS[i, j] = median_abs_deviation(unpump_tebin/Izero_unpump_tebin)
            err_ES[i, j] = median_abs_deviation(pump_tebin/Izero_pump_tebin)
            err_pp[i, j] = median_abs_deviation(pp_tebin)
            #err_pp[i, j] = np.sqrt(err_GS[i, j]**2 + err_ES[i, j]**2)

    print ('2D scan: {} shots out of {} survived'.format(np.sum(howmany), len(scanvar_e)))
    results = {'GS': GS, 'ES':ES, 'pp': pp, 'err_GS': err_GS, 'err_ES': err_ES, 'err_pp': err_pp, 'scanvar_rebin': scanvar_rebin, 'delay_rebin': delay_rebin, 'howmany': howmany}   
    return results 



def plot_bins_populations(results, meta):
    scanvar_rebin = results['scanvar_rebin']
    howmany       = results['howmany']
    fig = plt.figure(figsize = (7,5))
    fig.suptitle("\n".join(wrap(meta[''])))
    ax1 = fig.add_subplot(111)
    ax2 = plt.twinx(ax1)
    
    delayrange = np.arange(0, len(Delay_rebin), 1)
    ax1.plot(Delay_rebin, howmany, color = 'darkorange')
    
    ax2.scatter(Delay_rebin, delayrange, s = 5)

    ax1.grid()
    plt.show()

def normalize(data):
    res = copy.deepcopy(data)
    norm = np.nanmean(res['results']['GS'])
    for key in ['GS', 'ES', 'err_GS', 'err_ES']:
        res['results'][key] = data['results'][key]/norm

    return res

def average_two_diodes(data1, data2):
    d1 = normalize(data1)
    d2 = normalize(data2)

    res = copy.deepcopy(d1)

    def combine(value1, value2, err1, err2):
        mean = (value1 + value2)/2
        err  = np.sqrt(err1**2 + err2**2)/2
        return mean, err

    GS, err_GS = combine(d1['results']['GS'], d2['results']['GS'],
                         d1['results']['err_GS'], d2['results']['err_GS'])

    ES, err_ES = combine(d1['results']['ES'], d2['results']['ES'],
                         d1['results']['err_ES'], d2['results']['err_ES'])

    pp, err_pp = combine(d1['results']['pp'], d2['results']['pp'],
                         d1['results']['err_pp'], d2['results']['err_pp'])

    res['results'].update({
        'GS':GS, 'err_GS': err_GS,
        'ES':ES, 'err_ES': err_ES,
        'pp':pp, 'err_pp': err_pp})

    res['params'].update({
        'signal1': 'both diodes',
        'signal2': 'both diodes'})

    return res


class plotter:

    @staticmethod
    def dofits(datadict):
        rbk = datadict['scanvar_rebin']
        signal = datadict['pp']
        err_signal = datadict['err_pp']

        index = ~(np.isnan(rbk) | np.isnan(signal))
        rbk = rbk[index]
        signal=  signal[index]
        err_signal = err_signal[index]
        
        datadict.update({'scanvar_rebin':rbk, 'pp': signal, 'err_pp': err_signal})
        
        # Fit the curve
        fit = Fit(conv_exp_gauss_heaviside, estimate_conv_exp_gauss_heaviside_parameters)
        fit.estimate(rbk, signal)
        fit.p0 = better_p0(fit.p0, 4, 200)
        fit.fit(rbk,signal, maxfev=200000)
        sig_fit = fit.eval(rbk)

        sig_derivative = gaussian_filter1d(signal,2, order = 1)

        # Fit derivative Gaussian [x0, amplitude, sigma, offset]
        params,_ = curve_fit(gaussian, rbk, sig_derivative, p0 = [rbk.mean(), -sig_derivative.max(), np.diff(rbk).mean(), sig_derivative.min()])

        # Fit derivative pseudo Voigt
        mod = PseudoVoigtModel()
        pars = mod.guess(sig_derivative, x=rbk)
        init = mod.eval(pars, x=rbk)
        #out = mod.fit(sig_derivative, amplitude=-0.0006, center= 0, sigma=0.1, fraction=0.5, x=rbk)
        out = mod.fit(sig_derivative, amplitude=-sig_derivative.max(), center= rbk.mean(), sigma = np.diff(rbk).mean(), fraction=0.5, x=rbk)

        return datadict, sig_derivative, sig_fit, fit.popt, params, out

    @staticmethod
    def doGaussianfit(datadict):
        rbk = datadict['scanvar_rebin']
        signal = datadict['pp']
        err_signal = datadict['err_pp']

        index = ~(np.isnan(rbk) | np.isnan(signal))
        rbk = rbk[index]
        signal=  signal[index]
        err_signal = err_signal[index]
        
        datadict.update({'scanvar_rebin':rbk, 'pp': signal, 'err_pp': err_signal})
        
        # Fit derivative Gaussian [x0, amplitude, sigma, offset]
        params,_ = curve_fit(gaussian, rbk, signal, p0 = [rbk.mean(), -signal.max(), np.diff(rbk).mean(), signal.min()])

        return datadict, params

    @staticmethod
    def doGaussianfit_array(array, rbk):
        fit = Fit(gaussian, estimate_gaussian_parameters)
        fit.estimate(rbk, array)
        fit.fit(rbk, array)    

        fit_curve = fit.eval(rbk)
        center = fit.popt[0]
        width  = np.abs(fit.popt[2]) * 2.355
        
        return fit_curve, center, width

    @classmethod
    def general_scan(self, data, meta, rbk, figsize=(6, 5)):
        
        title = meta['title']
        title = "\n".join(textwrap.wrap(title))
        Int, err_low, err_high = unwrap_data(data)

        fig, (ax1) = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plt.suptitle(title)
      
        ax1.fill_between(rbk, err_low, err_high, color='lightblue', alpha = 0.8)
        ax1.plot(rbk, Int, marker='.')

        ax1.set(xlabel="{} ({})".format(meta['xlabel'], meta['units']),
                ylabel="Intensity")
        ax1.grid()

        return fig, (ax1)


    @classmethod
    def energy_scans(self, data, meta, figsize=(10, 4)):

        xlabel = meta.get('xlabel','')
        xunits = meta.get('units','')
        title  = meta.get('title','')

        r = data['results']
        p = data.get('params', {})
        w = data.get('which', None)
        s = p.get(w,"")
        title = title + ' --- ' + s
        title = "\n".join(textwrap.wrap(title))
        rbk = r['scanvar_rebin']

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        plt.suptitle(title)
      
        ax1.fill_between(rbk, r['ES']-r['err_ES'], r['ES']+r['err_ES'], color='lightblue', alpha = 0.8)
        ax1.plot(rbk, r['ES'], color='blue', marker='.', label='ON')
        ax1.fill_between(rbk, r['GS']-r['err_GS'], r['GS']+r['err_GS'], color='navajowhite', alpha = 0.8)
        ax1.plot(rbk, r['GS'], color='orange', marker='.', label='OFF')

        ax1.set(xlabel="{} ({})".format(xlabel, xunits),
                ylabel="XAS Diode",
                title="XAS (fluo)")
        ax1.legend()
        ax1.grid()
        
        ax3.fill_between(rbk, r['pp']-r['err_pp'], r['pp']+r['err_pp'], label='pump probe',color='lightgreen')
        ax3.plot(rbk, r['pp'], color='green', marker='.')
        
        ax3.set(xlabel="{} ({})".format(xlabel, xunits),
                ylabel="DeltaXAS",
                title="Pump probe")
        ax3.legend()
        ax3.grid()
        
        return fig, (ax1, ax3)

    @classmethod
    def delay_scans(self, data, meta, figsize=(10, 4)):

        xlabel = meta.get('xlabel','')
        if xlabel in [None, "None"]:
            xlabel = "pp delay - continuous"
        xunits = meta.get('units','')
        if xunits in [None, "None"]:
            xunits = "fs"
        title  = meta.get('title','')

        r = data['results']
        p = data.get('params', {})
        w = data.get('which', None)
        s = p.get(w,"")
        title = title + ' --- ' + s
        title = "\n".join(textwrap.wrap(title))

        rbk = r['scanvar_rebin']

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        plt.suptitle(title)
      
        ax1.fill_between(rbk, r['ES']-r['err_ES'], r['ES']+r['err_ES'], label='ON', color='lightblue', alpha = 0.8)
        ax1.plot(rbk, r['ES'], color='blue', marker='.', label='ON')
        ax1.fill_between(rbk, r['GS']-r['err_GS'], r['GS']+r['err_GS'], label='OFF', color='navajowhite', alpha = 0.8)
        ax1.plot(rbk, r['GS'], color='orange', marker='.', label='OFF')

        ax1.set(xlabel="{} ({})".format(xlabel, xunits),
                ylabel="XAS Diode",
                title="XAS (fluo)")
        ax1.legend()
        ax1.grid()
        
        ax3.fill_between(rbk, r['pp']-r['err_pp'], r['pp']+r['err_pp'], label='pump probe',color='lightgreen')
        ax3.plot(rbk, r['pp'], color='green', marker='.')
        
        ax3.set(xlabel="{} ({})".format(xlabel, xunits),
                ylabel="DeltaXAS",
                title="Pump probe")
        ax3.legend()
        ax3.grid()
        
        return fig, (ax1, ax3)

    @classmethod
    def overlap_scans(cls, data, meta, figsize=(10, 4)):
        from matplotlib.ticker import MaxNLocator

        xlabel = meta.get('xlabel','')
        xunits = meta.get('units','')
        title  = meta.get('title','')

        r = data['results']
        p = data.get('params', {})
        w = data.get('which', None)
        s = p.get(w,"")
        title = title + ' --- ' + s
        title = "\n".join(textwrap.wrap(title))

        r, params_gauss = cls.doGaussianfit(r)

        rbk = r['scanvar_rebin']

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        plt.suptitle(title)
      
        ax1.fill_between(rbk, r['ES']-r['err_ES'], r['ES']+r['err_ES'], color='lightblue', alpha = 0.8)
        ax1.plot(rbk, r['ES'], color='blue', marker='.', label='ON')
        ax1.fill_between(rbk, r['GS']-r['err_GS'], r['GS']+r['err_GS'], color='navajowhite', alpha = 0.8)
        ax1.plot(rbk, r['GS'], color='orange', marker='.', label='OFF')

        ax1.set(xlabel="{} ({})".format(xlabel, xunits),
                ylabel="XAS Diode",
                title="XAS (fluo)")
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax1.legend()
        ax1.grid()
        
        ax3.fill_between(rbk, r['pp']-r['err_pp'], r['pp']+r['err_pp'], label='pump probe',color='lightgreen')
        ax3.plot(rbk, r['pp'], color='green', marker='.')
        ax3.plot(rbk, gaussian(rbk,*params_gauss), color='red', label = 'fit Gauss, w= {:.4f} {}'.format(np.abs(params_gauss[2]*2.355), xunits[0]))
        
        ax3.set(xlabel="{} ({})".format(xlabel, xunits),
                ylabel="DeltaXAS",
                title='center = {:.4f} {}'.format(params_gauss[0], xunits[0]))
        ax3.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax3.legend()
        ax3.grid()
        
        return fig, (ax1, ax3)

    @classmethod
    def jet_scans(cls, data, meta, rbk, twodiodes=False, figsize=(6, 5)):

        title = meta['title']

        if not twodiodes:
            Int, err_low, err_high = unwrap_data(data)
            fit_curve, center, width = cls.doGaussianfit_array(Int, rbk)

            fig, (ax1) = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
            title = title + ' --- center = {:.3f} um'.format(center)
            title = "\n".join(textwrap.wrap(title))
            plt.suptitle(title)

            ax1.fill_between(rbk, err_low, err_high, color='lightblue', alpha = 0.8)
            ax1.plot(rbk, Int, marker='.', label='width = {:.3f} um'.format(width*1000)) 
            ax1.plot(rbk, fit_curve)

            ax1.set(xlabel="{} ({})".format(meta['xlabel'], meta['units']),
                ylabel="Intensity")
            ax1.legend(loc='upper right')
            ax1.grid()
            return fig, (ax1)
    
        else:
            Int1, err_low1, err_high1 = unwrap_data(data)
            Int2, err_low2, err_high2 = unwrap_data(data, 1)

            _, center1, width1 = cls.doGaussianfit_array(Int1, rbk)
            _, center2, width2 = cls.doGaussianfit_array(Int2, rbk)
            
            fig, (ax1) = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
            title = title + ' --- centers = {:.3f} & {:.3f} um'.format(center1, center2)
            title = "\n".join(textwrap.wrap(title))
            plt.suptitle(title)

            ax2 = ax1.twinx()
            
            lns1 = ax1.plot(rbk, Int1, label='width = {:.3f} um'.format(width1*1000), marker='.', color='blue')
            ax1.fill_between(rbk, err_low1, err_high1, color='lightblue', alpha = 0.8)

            lns2 = ax2.plot(rbk, Int2, label='width = {:.3f} um'.format(width2*1000), marker='.', color='orange')
            ax2.fill_between(rbk, err_low2, err_high2, color='navajowhite', alpha = 0.8)

            ax1.set_ylabel('Diode1', color='blue')
            ax1.tick_params(axis='y', colors='blue')

            ax2.set_ylabel('Diode2', color='darkorange')
            ax2.tick_params(axis='y', colors='darkorange')

            lns = lns1 + lns2
            labels = [l.get_label() for l in lns]
            ax1.set(xlabel="{} ({})".format(meta['xlabel'], meta['units']))
            ax1.legend(lns, labels, loc='best')

            ax1.grid()
            return fig, (ax1, ax2)

    @classmethod
    def fit_risetime(cls, data, meta, fitflag=True, figsize=(10,4)):

        xlabel = meta.get('xlabel','')
        if xlabel in [None, "None"]:
            xlabel = "pp delay - continuous"
        xunits = meta.get('units','')
        if xunits in [None, "None"]:
            xunits = "fs"
        title  = meta.get('title','')

        r = data['results']
        p = data.get('params', {})
        w = data.get('which', None)
        s = p.get(w,"")
        title = title + ' --- ' + s
        if 'binsize' in p:
            title = title + ' --- ' +'binsize {} fs --- {} on/off pairs'.format(int(p['binsize']), np.sum(r['howmany']))
        title = "\n".join(textwrap.wrap(title))

        t0_fit   = np.nan
        t0_Voigt = np.nan
        t0_Gauss = np.nan
        _, sig_derivative, _, _, _, _ = cls.dofits(r)
        
        if fitflag:
            r, sig_derivative, sig_fit, params_fit, params_gauss, out_voigt = cls.dofits(r)
            t0_Voigt = out_voigt.params.get('center').value
            t0_Gauss = params_gauss[0]
            t0_fit   = params_fit[0]

        rbk = r['scanvar_rebin']

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        plt.suptitle(title)

        ax1.fill_between(rbk, r['pp']-r['err_pp'], r['pp']+r['err_pp'],color='lightgreen')
        ax1.plot(rbk, r['pp'], color='green', marker='.', label='pump probe')
        if fitflag:
            ax1.plot(rbk, sig_fit, color='red', label = 'fit, w = {:.2f} fs'.format(np.abs(params_fit[2])))

        ax1.set(xlabel="{} ({})".format(xlabel, xunits),
                title='t0 = {:.2f} fs'.format(t0_fit))
        ax1.legend()
        ax1.grid()

        ax2.scatter(rbk, sig_derivative, color='limegreen', label = 'derivative', s = 5)
        if fitflag:
            ax2.plot(rbk, out_voigt.best_fit, color='red', label = 'fit psVoigt, w = {:.2f} fs'.format(np.abs(out_voigt.params.get('fwhm').value)))
        
        ax2.set(xlabel="{} ({})".format(xlabel, xunits),
                title='t0 = {:.2f} fs'.format(t0_Voigt))
        ax2.legend()
        ax2.grid()

        ax3.scatter(rbk, sig_derivative, color='limegreen', label = 'derivative', s = 5)
        if fitflag:
            ax3.plot(rbk, gaussian(rbk,*params_gauss), color='red', label = 'fit Gauss, w= {:.2f} fs'.format(np.abs(params_gauss[2]*2.355)))

        ax3.set(xlabel="{} ({})".format(xlabel, xunits),
                title='t0 = {:.2f} fs'.format(t0_Gauss))
        ax3.legend()
        ax3.grid()
      
        return fig, (ax1, ax2, ax3)

    @classmethod
    def fit_decay(cls, data, meta, fitfunction, p0=None, figsize=(6,4)):
        def get_default_p0(fitfunction):
            if fitfunction.__name__ == 'model_decay_1exp':
                       #x0, sigma, amp1, tau1,  C
                return [0,  40,    200,  1000,  800]
            elif fitfunction.__name__ == 'model_decay_2exp':
                       #x0, sigma, amp1, tau1,  C
                return [0,  40,    200,  1000,  800, 10, 1000]
            else:
                raise ValueError("Unknown fit function")

        xlabel = meta.get('xlabel','')
        if xlabel in [None, "None"]:
            xlabel = "pp delay - continuous"
        xunits = meta.get('units','')
        if xunits in [None, "None"]:
            xunits = "fs"
        title  = meta.get('title','')

        r = data['results']
        p = data.get('params', {})
        w = data.get('which', None)
        s = p.get(w,"")
        title = title + ' --- ' + s
        title = "\n".join(textwrap.wrap(title))

        rbk = r['scanvar_rebin']
        
        if p0 is None:
            p0 = get_default_p0(fitfunction)
        popt,_  = curve_fit(fitfunction, rbk, r['pp'], p0=p0, maxfev=40000)
        sig_fit = fitfunction(rbk, *popt)

        popt = np.pad(popt, (0, 7-len(popt)), 'constant', constant_values=np.nan)

        t0_fit = popt[0]
        IRF    = popt[1]*2.355
        tau1   = popt[3]
        tau2   = popt[6]

        fig, (ax1) = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plt.suptitle(title)
    
        ax1.fill_between(rbk, r['pp']-r['err_pp'], r['pp']+r['err_pp'],color='lightgreen')
        ax1.plot(rbk, r['pp'], color='green', marker='.', label='pump probe')
        ax1.plot(rbk, sig_fit, color='red', label = f'IRF = {IRF:.2f} {xunits}\n' \
                                                    f'tau1 = {tau1:.2f} {xunits}\n' \
                                                    f'tau2 = {tau2:.2f} {xunits}')


        ax1.set(xlabel="{} ({})".format(xlabel, xunits),
                title='t0 = {:.2f} {}'.format(t0_fit, xunits))
        ax1.legend()
        ax1.grid()
      
        return fig, (ax1)

    @classmethod
    def fluence_scans(self, data, meta, params=None, figsize=(12, 4)):

        xlabel = meta.get('xlabel','')
        xunits = meta.get('units','')
        title  = meta.get('title','')

        r = data['results']
        p = data.get('params', {})
        w = data.get('which', None)
        s = p.get(w,"")
        title = title + ' --- ' + s
        title = "\n".join(textwrap.wrap(title))

        rbk = r['scanvar_rebin']
        intensity = rbk.copy()

        if params is not None:
            p = np.poly1d(params)
            intensity = p(rbk)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        plt.suptitle(title)
      
        ax1.fill_between(rbk, r['ES']-r['err_ES'], r['ES']+r['err_ES'], color='lightblue', alpha = 0.8)
        ax1.plot(rbk, r['ES'], color='blue', marker='.', label='ON')
        ax1.fill_between(rbk, r['GS']-r['err_GS'], r['GS']+r['err_GS'], color='navajowhite', alpha = 0.6)
        ax1.plot(rbk, r['GS'], color='orange', marker='.', label='OFF')

        ax1.set(xlabel="{} ({})".format(xlabel, xunits),
                ylabel="XAS Diode",
                title="XAS (fluo)")
        ax1.legend()
        ax1.grid()
        
        ax2.fill_between(rbk, r['pp']-r['err_pp'], r['pp']+r['err_pp'],color='lightgreen')
        ax2.plot(rbk, r['pp'], color='green', marker='.', label='pump probe')
        
        ax2.set(xlabel="{} ({})".format(xlabel, xunits),
                ylabel="DeltaXAS",
                title="Pump probe")
        ax2.legend()
        ax2.grid()

        ax3.fill_between(intensity, r['pp']-r['err_pp'], r['pp']+r['err_pp'],color='lightgreen')
        ax3.plot(intensity, r['pp'], color='green', marker='.', label='pump probe')
        
        ax3.set(xlabel="Pulse Energy (uJ)",
                ylabel="DeltaXAS",
                title="Pump probe")
        ax3.legend()
        ax3.grid()
        
        return fig, (ax1, ax2, ax3)

    @classmethod
    def KnifeEdge_scans(self, data, meta, rbk, figsize=(6, 5)):
        
        title = meta['title']
        Int, err_low, err_high = unwrap_data(data)

        fit = Fit(errfunc_fwhm, estimate_errfunc_parameters)
        fit.estimate(rbk, Int)
        fit.fit(rbk,Int)
        sig_fit = fit.eval(rbk)

        centerpos_mm = fit.popt[0]
        width_mm = np.abs(fit.popt[2])
        title = title + ' --- fwhm = {:.2f} um'.format(width_mm*1000)
        title = "\n".join(textwrap.wrap(title))

        fig, (ax1) = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plt.suptitle(title)
      
        ax1.fill_between(rbk, err_low, err_high, color='lightblue', alpha = 0.8)
        ax1.plot(rbk, Int, marker='.', label=meta['xlabel'])
        ax1.plot(rbk, sig_fit)

        ax1.set(xlabel="{} ({})".format(meta['xlabel'], meta['units']),
                ylabel="Intensity")
        ax1.legend()
        ax1.grid()

        return fig, (ax1)


    @classmethod
    def TwoD_scans(self, data, meta, vmin=None, vmax=None, what='pp', figsize=(10, 4), show=True):
       
        xlabel = meta.get('xlabel','')
        xunits = meta.get('units','')
        title  = meta.get('title','')

        r = data['results']
        p = data.get('params', {})
        w = data.get('which', None)
        s = p.get(w,"")
        title = title + ' --- ' + s
        title = "\n".join(textwrap.wrap(title))

        rbk = r['scanvar_rebin']
        delays = r['delay_rebin']

        whatplot = np.ma.masked_invalid(r[what])
        cmap = plt.get_cmap('turbo').copy()
        cmap.set_bad(color='white')

        if vmin is None:
            vmin = np.nanmin(r[what])
        if vmax is None:
            vmax = np.nanmax(r[what])

        levels = np.linspace(vmin, vmax, 31)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        plt.suptitle(title)
      
        pcm = ax1.pcolormesh(delays, rbk, whatplot, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        con = ax2.contourf(delays, rbk, whatplot, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax, extend='both')
       
        ax1.set(ylabel="{} ({})".format(xlabel, xunits),
                xlabel="Delays (fs)")
        
        ax1.axvline(x = 0, color = 'k', linestyle = '--')

        ax2.set(xlabel="Delays (fs)")

        ax2.axvline(x = 0, color = 'k', linestyle = '--')

        cbar = fig.colorbar(pcm, ax=[ax1,ax2], fraction=0.05)
        cbar.set_label('Difference', rotation=270, labelpad=25)

        if show:
            plt.show()

        return fig, (ax1, ax2)

    @classmethod
    def TwoD_scans_lineouts(self, data, meta, energylist, delayslist, delay_int, energy_int, vmin=None, vmax=None, figsize=(9, 7), show=True):
        import matplotlib.gridspec as gridspec
        energy_int = energy_int / 2
        delay_int = delay_int /2

        xlabel = meta.get('xlabel','')
        xunits = meta.get('units','')
        title  = meta.get('title','')

        r = data['results']
        p = data.get('params', {})
        w = data.get('which', None)
        s = p.get(w,"")
        title = title + ' --- ' + s
        title = "\n".join(textwrap.wrap(title))

        rbk = r['scanvar_rebin']
        delays = r['delay_rebin']

        pp = np.ma.masked_invalid(r['pp'])
        cmap = plt.get_cmap('turbo').copy()
        cmap.set_bad(color='white')

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = gridspec.GridSpec(2,2, height_ratios=[2,2], width_ratios=[2,2])

        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[1,0])    
        ax3 = plt.subplot(gs[0,1])

        pcm = ax1.pcolormesh(delays, rbk, pp, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

        for energy in energylist:
            idx = np.argmin(np.abs(np.array(rbk) - energy))
            mask_e = np.abs(np.array(rbk) - energy) <= energy_int
            if not np.any(mask_e):
                print ('Energy integration range smaller than delay bin')
                mask_e[idx] = True

            cut_e = np.nanmean(pp[mask_e, :], axis=0)

            ax2.plot(delays, cut_e, label = '{:.2f}'.format(np.array(rbk)[idx]))
            y0 = energy - energy_int
            y1 = energy + energy_int
            ax1.axhspan(y0, y1, alpha=0.3)
            ax1.hlines(energy, xmin=min(delays), xmax=max(delays), linestyles='--', color='black')

        ax2.set_xlabel("Delay (fs)")
        ax2.legend()
        ax2.grid()

        for delay in delayslist:
            idx = np.argmin(np.abs(np.array(delays) - delay))
            mask_d = np.abs(np.array(delays) - delay) <= delay_int
            if not np.any(mask_d):
                print ('Delay integration range smaller than delay bin')
                mask_d[idx] = True

            cut_d = np.nanmean(pp[:, mask_d], axis=1)

            ax3.plot(rbk, cut_d, label = '{}'.format(np.array(delays)[idx]))
            x0 = delay - delay_int
            x1 = delay + delay_int
            ax1.axvspan(x0, x1, alpha=0.3)
            ax1.vlines(delay, ymin=min(rbk), ymax=max(rbk), linestyles='--', color='gray')

        ax3.set_xlabel("Energy (eV)")
        ax3.legend()
        ax3.grid()

        #plt.tight_layout()

        if show:
            plt.show()

        return fig, (ax1, ax2, ax3)

    @classmethod
    def bins_population(self, data, meta, figsize=(7, 5), show=True):

        xlabel = meta.get('xlabel','')
        if xlabel in [None, "None"]:
            xlabel = "pp delay"
        xunits = meta.get('units','')
        if xunits in [None, "None"]:
            xunits = "fs"
        title  = meta.get('title','')

        r = data['results']
        p = data.get('params', {})
        w = data.get('which', None)
        s = p.get(w,"")
        title = title + ' --- ' + s
        title = "\n".join(textwrap.wrap(title))

        rbk = r['scanvar_rebin']
        howmany = r['howmany']

        fig, (ax1) = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plt.suptitle(title)

        ax2 = plt.twinx(ax1)
        xaxisrange = np.arange(0, len(rbk), 1)

        ax1.plot(rbk, howmany, color = 'darkorange')
        ax2.scatter(rbk, xaxisrange, s = 5)

        ax1.set(xlabel="{} ({})".format(xlabel, xunits),
                ylabel="Number of shots per bin")

        ax2.set(ylabel="Step number")

        ax1.grid()
        return fig, ax1

    @classmethod
    def shot_noise(self, data, meta, figsize=(7, 5), show=True):

        xlabel = meta.get('xlabel','')
        if xlabel in [None, "None"]:
            xlabel = "pp delay"
        xunits = meta.get('units','')
        if xunits in [None, "None"]:
            xunits = "fs"
        title  = meta.get('title','')

        r = data['results']
        p = data.get('params', {})
        w = data.get('which', None)
        s = p.get(w,"")
        title = title + ' --- ' + s
        title = "\n".join(textwrap.wrap(title))

        rbk = r['scanvar_rebin']
        howmany = r['howmany']

        fig, (ax1) = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plt.suptitle(title)

        ax2 = plt.twinx(ax1)

        ax1.plot(rbk, r['err_GS']/r['GS']*100)
        ax2.plot(rbk, howmany, color = 'darkorange')

        ax1.set_xlabel("{} ({})".format(xlabel, xunits))
        ax1.set_ylabel("std/mean (%)", color='blue')
        ax2.set_ylabel("Number of shots per bin", color = 'orange')
    
        ax1.grid()
        return fig, ax1
        

def SaveData(SaveDir, runlist, plot1=None, plot2=None, plot_both=None):
    import numbers
    savedir = SaveDir + '_singlerun/'
    if len(runlist) == 1:
        runname2save = 'run{:04d}'.format(runlist[0])
    else:
        savedir = SaveDir+'_multiruns/'
        runname2save = 'run' + '_'.join("{:04d}".format(x) for x in runlist)

    savedir = savedir + runname2save

    save_dict = {}

    if plot1 is not None:
        save_dict['plot1'] = plot1
    if plot2 is not None:
        save_dict['plot2'] = plot2
    if plot_both is not None:
        save_dict['plot_both'] = plot_both

    os.makedirs(savedir, mode=0o775, exist_ok=True)
    np.savez(savedir+'/data.npz', **save_dict)
    os.chmod(savedir+'/data.npz', 0o775)
    print('Data saved in {}/'.format(savedir))

def unpack(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        return x.item()
    return x

def extract_diode(data, which_plot):
    if which_plot == 'plot1':
        return unpack(data['plot1'])
    elif which_plot == 'plot2':
        return unpack(data['plot2'])
    elif which_plot == 'both':
        return unpack(data['plot_both'])
    else:
        raise ValueError("which_plot must be 'plot1', 'plot2', or 'both'")
    return result

def LoadDataFlexible(SaveDir, runlists, which_folder='_singlerun', which_plot='both'):
    if isinstance(runlists[0], int):
        runlists = [runlists]
    datasets = []
    runnames = []

    # --- MULTIRUNS ---
    if which_folder == '_multiruns':
        base = SaveDir + '_multiruns/'
        for runlist in runlists:

            pattern = "run" + "_".join(f"{r:04d}" for r in runlist)
            search_path = os.path.join(base, pattern, "data.npz")

            matches = glob.glob(search_path)

            if not matches:
                print("Missing multirun {}".format(runlist))
                continue

            path = matches[0]
            runnames.append(path.split('/')[-2])
            
            data = np.load(path, allow_pickle=True)
            d = extract_diode(data, which_plot)
            p = d.get('params', {})
            w = d.get('which', None)
            plot = p.get(w,"")
            print("Loaded __{}__ from: {}".format(plot, path))
            datasets.append(extract_diode(data, which_plot))

    # --- SINGLERUNS ---
    elif which_folder == '_singlerun':
        base = SaveDir + '_singlerun/'
        for runlist in runlists:
            for r in runlist:

                pattern = os.path.join(base, f"run{r:04d}", "data.npz")
                matches = glob.glob(pattern)

                if not matches:
                    print("Missing run {}".format(r))
                    continue

                path = matches[0]
                runnames.append(path.split('/')[-2])

                data = np.load(path, allow_pickle=True)
                d = extract_diode(data, which_plot)
                p = d.get('params', {})
                w = d.get('which', None)
                plot = p.get(w,"")
                print("Loaded __{}__ from: {}".format(plot, path))
                datasets.append(extract_diode(data, which_plot))

    else:
        raise ValueError("which_folder must be '_singlerun' or '_multiruns'")

    return datasets, runnames, plot


def LoadDataAuto(SaveDir, runlists, which_plot='both'):
    if isinstance(runlists[0], int):
        runlists = [runlists]

    datasets = []
    runnames = []
    plot = ""

    base_single = SaveDir + '_singlerun/'
    base_multi  = SaveDir + '_multiruns/'

    for runlist in runlists:

        # --- Try MULTIRUN first ---
        pattern_multi = os.path.join(
            base_multi,
            "run" + "_".join(f"{r:04d}" for r in runlist),
            "data.npz"
        )
        match_multi = glob.glob(pattern_multi)

        if match_multi:
            path = match_multi[0]
        else:
            # --- Fall back to SINGLE RUNS ---
            if len(runlist) == 1:
                r = runlist[0]
                pattern_single = os.path.join(base_single, f"run{r:04d}", "data.npz")
                match_single = glob.glob(pattern_single)

                if not match_single:
                    print(f"Missing run {r}")
                    continue

                path = match_single[0]

            else:
                # multiple runs but no multirun found → load individually
                for r in runlist:
                    pattern_single = os.path.join(base_single, f"run{r:04d}", "data.npz")
                    match_single = glob.glob(pattern_single)

                    if not match_single:
                        print(f"Missing run {r}")
                        continue

                    path = match_single[0]

                    runnames.append(path.split('/')[-2])

                    data = np.load(path, allow_pickle=True)
                    d = extract_diode(data, which_plot)

                    p = d.get('params', {})
                    w = d.get('which', None)
                    plot = p.get(w, "")

                    print(f"Loaded __{plot}__ from: {path}")
                    datasets.append(d)

                continue  # skip rest of loop

        # --- Common load path (multirun OR single fallback) ---
        runnames.append(path.split('/')[-2])

        data = np.load(path, allow_pickle=True)
        d = extract_diode(data, which_plot)

        p = d.get('params', {})
        w = d.get('which', None)
        plot = p.get(w, "")

        print(f"Loaded __{plot}__ from: {path}")
        datasets.append(d)

    return datasets, runnames, plot



