import numpy as np
import json, glob
import os, math
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec

from IPython.display import clear_output, display
from datetime import datetime
from scipy.stats import pearsonr
import itertools

from alvra_tools.load_data import *
from alvra_tools.channels import *
from alvra_tools.utils import *
from alvra_tools.timing_tool import *

from sfdata import SFScanInfo

def check_BSchannel(pgroup, runnumber, channel, acq=None, pp=True):
    channels_pp  = [channel_Events] + channel
    channels_fel = channels_pp
    jsonfile = glob.glob('/sf/alvra/data/{}/raw/*{:04d}*/meta/scan.json'.format(pgroup, runnumber))[0]
    scan = SFScanInfo(jsonfile)
    run_name = jsonfile.split('/')[-3]
    title = str( pgroup + ' --- ' +run_name)

    data = []
    
    if acq!=None:
        title = title + str(' --- acq {}'.format(acq) )
        s = scan[acq-1]
        check_files_and_data(s)
        check = get_filesize_diff(s)
        if check:
            filename = scan.files[acq-1][0].split('/')[-1].split('.')[0]
            print ('Processing: {}'.format(run_name))
            print ('Acquisition: {}'.format(filename))
            resultsPP, results, _, _ = load_data_compact_pump_probe(channels_pp, channels_fel, s)
    else:
        for i, step in enumerate(scan):
            check_files_and_data(step)
            check = get_filesize_diff(step)
            if check:
                clear_output(wait=True)
                filename = scan.files[i][0].split('/')[-1].split('.')[0]
                print ('Processing: {}'.format(run_name))
                print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))
                resultsPP, results , _, _ = load_data_compact_pump_probe(channels_pp, channels_fel, step)
    for ch in channel:
        if pp:
            data.append(resultsPP[ch].pump)
        else:
            data.append(results[ch])
    meta = [title, channel]
    return data, meta

def plot_checkedBSchannel(data, meta):
    data = np.asarray(data)
    plt.figure(figsize=(12,4))
    plt.suptitle(meta[0])
    threeplots = False
    if len(data) == 2:
        threeplots = True

    gs = gridspec.GridSpec(1,3)

    if not threeplots:
        ax1 = plt.subplot(gs[:,0])
        ax2 = plt.subplot(gs[:,1])
        for ch in range(len(data)):
            ax1.plot(data[ch].ravel(), label=meta[1][ch])
            ax2.hist(data[ch].ravel(), label=meta[1][ch], bins=100)
    else:
        ax1 = plt.subplot(gs[:,0])
        ax2 = plt.subplot(gs[:,1])
        ax3 = plt.subplot(gs[:,2])
        for ch in range(len(data)):
            ax1.plot(data[ch].ravel(), label=meta[1][ch])
            ax2.hist(data[ch].ravel(), label=meta[1][ch], bins=100)
        corr = pearsonr(data[0].ravel(), data[1].ravel())[0]
        ax3.plot(data[0].ravel(), data[1].ravel(), '.', ms=0.5, label='{:.5f}'.format(corr))
        ax3.set_ylabel(meta[1][1])
        ax3.set_xlabel(meta[1][0])        
        ax3.legend(loc='best')
        ax3.title.set_text('Correlation')
        ax3.grid()

    ax1.set_ylabel('Intensity')
    ax1.set_xlabel('Number of shots')
    ax1.legend(loc='best')
    ax1.grid()

    ax2.set_ylabel('counts')
    ax2.set_xlabel('Intensity')
    ax2.legend(loc='best')
    ax2.grid()
    
    plt.tight_layout()

def plot_checkedBSchannel_2D(data, meta):
    data = np.asarray(data)
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
    plt.suptitle(meta[0])
    
    pos = ax1.pcolormesh(np.squeeze(data), label=meta[1])
    fig.colorbar(pos, ax=ax1)
    ax2.hist(np.mean(np.squeeze(data), axis=0), label=meta[1], bins=100)

    ax1.set_ylabel('Intensity')
    ax1.set_xlabel('Number of shots')
    ax1.legend(loc='best')
    ax1.grid()

    ax2.set_ylabel('counts')
    ax2.set_xlabel('Intensity')
    ax2.legend(loc='best')
    ax2.grid()
    plt.tight_layout()

def creation_date(path_to_file):
    stat = os.stat(path_to_file)
    return stat.st_mtime


def check_JF_backlog(detector, pgroup, run):
    nfiles = len(list(glob.glob('/sf/alvra/data/{}/raw/*{:04d}*/meta/acq*'.format(pgroup, run))))
    print ('Run {}: {} acquisitions found'.format(run, nfiles))

    acqall = list(np.arange(1, nfiles+1, 1))
    BSonly = []
    JFonly = []
    reqtime = []
    jsononly = []
    trun = []
    flag_noBS = False
    for acq in acqall:
        ff = glob.glob('/sf/alvra/data/{}/raw/*{:04d}*/meta/*{:04d}*json*'.format(pgroup, run, acq))[0]
        #tdate.append(json.load(open(ff))['request_time'])
        t = time.mktime(time.strptime(json.load(open(ff))['request_time'],'%Y-%m-%d %H:%M:%S.%f'))
        reqtime.append(t)
        if acq==1:
            trun.append(t)
        
        jsonf = glob.glob('/sf/alvra/data/{}/raw/*{:04d}*/meta/*{:04d}*json*'.format(pgroup, run, acq))[0]
        try:
            JFf = glob.glob('/sf/alvra/data/{}/raw/*{:04d}*/data/*{:04d}*{}.h5*'.format(pgroup, run, acq, detector))[0]
            JFonly.append(creation_date(JFf))
        except:
            print("Acq {} not ready yet or no JF file present".format(acq))
            JFonly.append(np.nan)
        try:
            BSf = glob.glob('/sf/alvra/data/{}/raw/*{:04d}*/data/*{:04d}*BS*'.format(pgroup, run, acq))[0]
            BSonly.append(creation_date(BSf))
        except:
            flag_noBS=True
            BSonly.append(np.nan) #np.nan
        
        #JFonly.append(creation_date(JFf))
        jsononly.append(creation_date(jsonf))
    if flag_noBS:
        print ('No BSDATA files found')

    return reqtime, JFonly, BSonly, jsononly, trun

def plot_JF_backlog(reqtime, JFonly, BSonly, jsononly, pgroup, run):

    timeaxis = np.array(reqtime)-np.array(reqtime[0])
    index = ~(np.isnan(timeaxis) | np.isnan(JFonly))
    timeaxis = np.asarray(timeaxis)[index]
 
    reqtime = np.asarray(reqtime)[index]
    JFonly =  np.asarray(JFonly)[index]
    BSonly = np.asarray(BSonly)[index]
    jsononly = np.asarray(jsononly)[index]

    m_JF,b_JF = np.polyfit(timeaxis, np.array(JFonly)- np.array(reqtime), 1)
 
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    plt.suptitle ('{} -- run {}'.format(pgroup, run))

    offset = reqtime[0]

    ax1.plot(timeaxis, reqtime - offset, label ='req time')
    ax1.plot(timeaxis, JFonly - offset, '.', label ='JF mtime')
    ax1.plot(timeaxis, BSonly - offset, '.', label ='BS mtime')
    ax1.plot(timeaxis, jsononly - offset, '.', label ='json file mtime')

    ax1.grid()
    ax1.legend(loc='best')
    ax1.set_ylabel('mtime (s)')

    ax2.plot(timeaxis, np.array(JFonly)- np.array(reqtime), '.', label='JF - req time: {:.2f} s'.format(np.median(np.array(JFonly)- np.array(reqtime))), color = '#ff7f0e')
    ax2.plot(timeaxis, np.poly1d(np.polyfit(timeaxis, np.array(JFonly)- np.array(reqtime), 1))(timeaxis), color = 'black', linestyle = '--', label='slope={:.2f}'.format(m_JF))
    ax2.plot(timeaxis, np.array(BSonly)-np.array(reqtime), '.', label='BS - req time: {:.2f} s'.format(np.median(np.array(BSonly)- np.array(reqtime))), color = '#2ca02c')
    ax2.plot(timeaxis, np.array(jsononly)- np.array(reqtime), '.', label='json mtime - req time: {:.2f} s'.format(np.mean(np.array(jsononly)- np.array(reqtime))), color='#d62728')
    ax2.legend(loc='best')
    ax2.grid()
    ax2.set_ylabel('Delta mtime (s)')

    plt.xlabel('Lab time (s)')
    plt.tight_layout()
    plt.show()

def check_JF_backlog_loop(detector, pgroup, runlist):

    BSonly = []
    JFonly = []
    reqtime = []
    jsononly = []
    trun = []
    tdates = []

    for run in runlist:
        reqt, JF, BS, json, t = check_JF_backlog(detector, pgroup, run)
        reqtime.extend(reqt)
        BSonly.extend(BS)
        JFonly.extend(JF)
        jsononly.extend(json)
        trun.extend(t)
        
    return reqtime, JFonly, BSonly, jsononly, trun

def plot_JF_backlog_loop(reqtime, trun, JFonly, BSonly, jsononly, pgroup, runlist):
    
    timeaxis = (np.array(reqtime)-np.array(reqtime[0]))/3600
    index = ~(np.isnan(timeaxis) | np.isnan(JFonly))
    #index1 = ~(np.isnan(trun) | np.isnan(JFonly))

    timeaxis2 = np.asarray(timeaxis)[index]
    reqtime2 = np.asarray(reqtime)[index]
    JFonly =  np.asarray(JFonly)[index]
    #BSonly = np.asarray(BSonly)[index]
    #jsononly = np.asarray(jsononly)[index]
    
    #trunaxis = (np.array(trun)-np.array(trun[0]))/3600
    #trunaxis = np.asarray(trunaxis)[index2]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True, gridspec_kw={'width_ratios': [3, 1]})
    plt.suptitle ('{} -- {}'.format(pgroup, runlist))

    JFtimes = (np.array(JFonly)- np.array(reqtime2))/60
    BStimes = (np.array(BSonly)-np.array(reqtime))/60
    jsontimes = (np.array(jsononly)- np.array(reqtime))

    ax2.hist(JFtimes, bins = 50, orientation=u'horizontal', label='JF - req time: {:.2f} min'.format(np.median(JFtimes)), color='#ff7f0e')
    ax2.axhline(y= np.median(JFtimes), color='black', linestyle='--')
    ax2.legend(loc='best')
    ax2.grid()

    ax1.plot(timeaxis, BStimes, '.', ms=3, label='BS - req time: {:.2f} s'.format(np.median(BStimes)*60), color = '#2ca02c')
    ax1.plot(timeaxis, jsontimes, '.',  ms=3, label='json mtime - req time: {:.2f} s'.format(np.median(jsontimes)), color='#d62728')
    ax1.plot(timeaxis2, JFtimes, '.',  ms=3, label='JF - req time: {:.2f} min'.format(np.median(JFtimes)), color = '#ff7f0e')
    ax1.legend(loc='best')
    ax1.grid()
    ax1.set_ylabel('Delta mtime (min)')

    ax1.set_xlabel('Lab time (hour)')

    plt.tight_layout()
    plt.show()

def timestamp(json_file):
    file_split = json_file.split('/')[:-1]
    path_to_bsdata = '/'.join([*file_split[:-1], 'data', '*BSDATA.h5'])
    
    timestamp_s = []
    
    for file in glob.glob(path_to_bsdata):
        with h5py.File(file) as f:
            timestamp_ns = f[channel_Events]['timestamp'][:]
            timestamp_s.append(np.mean(timestamp_ns) * 1e-9)
    timestamp_s = np.mean(timestamp_s)
    timestamp_datetime = datetime.fromtimestamp(timestamp_s)
    return np.datetime64(timestamp_datetime)

def timestamp_step(step):
    bsdata = step.fnames[0]
    timestamp_s = []
    with h5py.File(bsdata) as f:
        timestamp_ns = f[channel_Events]['timestamp'][:]
        timestamp_s.append(np.mean(timestamp_ns) * 1e-9)
    timestamp_s = np.mean(timestamp_s)
    timestamp_datetime = datetime.fromtimestamp(timestamp_s)
    return np.datetime64(timestamp_datetime)

def check_psss(pgroup, runlist):
    datafiles = []
    print (runlist)
    for run in runlist:
        file = glob.glob('/sf/alvra/data/{}/raw/*{:04d}*/meta/scan.json'.format(pgroup, run))
        datafiles.extend(file)

    channel_list = ['SARFE10-PSSS059:FIT-COM', "SARFE10-PBPG050:HAMP-INTENSITY-CAL"]
    from sfdata import SFScanInfo

    all_results = []
    psss_com = []
    timestamps_npy = []
    pulseEnergy = []
    timestamps = []

    for j,json_file in enumerate(datafiles):
        scan = SFScanInfo(json_file)
        psss_com_run=[]
        pulseEnergy_run = []
        timestamps_run = []
        for i, step in enumerate(scan):
            check_files_and_data(step)
	        check = get_filesize_diff(step) 
	        if check:
                clear_output(wait=True)
	            print ('{}/{}: {}'.format(j+1, len(datafiles), json_file))
	            timestamps_run.append( timestamp_step(step) )
	            filename = scan.files[i][0].split('/')[-1].split('.')[0]
	            print ('Step {} of {}: Processing {}'.format(i+1, len(scan.files), filename))

	            results,pids = load_data_compact(channel_list, step)
	            temp = results['SARFE10-PSSS059:FIT-COM']
	            psss_com_run.append(np.average(results['SARFE10-PSSS059:FIT-COM']))
	            psss_com.append(np.average(results['SARFE10-PSSS059:FIT-COM']))
	            
	            pulseEnergy.append(np.average(results["SARFE10-PBPG050:HAMP-INTENSITY-CAL"]))
	            pulseEnergy_run.append(np.average(results["SARFE10-PBPG050:HAMP-INTENSITY-CAL"]))
	            timestamps.append(timestamp_step(step))

    return timestamps, psss_com, pulseEnergy




    


