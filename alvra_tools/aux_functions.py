import numpy as np
import json, glob
import os, math
from matplotlib import pyplot as plt
from IPython.display import clear_output, display
from datetime import datetime
from scipy.stats.stats import pearsonr
import itertools

from alvra_tools.load_data import *
from alvra_tools.channels import *
from alvra_tools.utils import *
from alvra_tools.timing_tool import *

from sfdata import SFScanInfo

def check_BSchannel(pgroup, runnumber, channel, acq=None, pp=True):
    channels_pp  = [channel_Events, channel]
    channels_fel = channels_pp
    jsonfile = glob.glob('/sf/alvra/data/{}/raw/*{:04d}*/meta/scan.json'.format(pgroup, runnumber))[0]
    
    scan = SFScanInfo(jsonfile)
    run_name = jsonfile.split('/')[-3]
    title = str( pgroup + ' --- ' +run_name)

    data = []
    
    if acq!=None:
        title = title + str(' --- acq {}'.format(acq) )
        s = scan[acq]
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
    if pp:
        data.extend(resultsPP[channel].pump)
    else:
        data.extend(results[channel])
    meta = [title, channel]
    return np.asarray(data), meta


def plot_checkedBSchannel(data, meta):
    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))

    plt.suptitle(meta[0])
    ax1.plot(data.ravel(), label=meta[1])
    ax1.set_ylabel('Intensity')
    ax1.set_xlabel('Number of shots')
    ax1.legend(loc='best')
    ax1.grid()

    ax2.hist(data.ravel(), label=meta[1])
    ax2.set_ylabel('counts')
    ax2.set_xlabel('Intensity')
    ax2.legend(loc='best')
    ax2.grid()
