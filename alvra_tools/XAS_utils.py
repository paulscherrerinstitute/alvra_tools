import numpy as np
from matplotlib import pyplot as plt
from alvra_tools.XAS_functions import *

################################################

def Plot_correlation(titlestring, scan, data, quantile, det1, det2, timescan, xlim=None, ylim=None, ylim_pp=None):
    
    readbacks    = np.asarray(data["readbacks"])
    timezero_mm  = np.asarray(data["timezero_mm"])

    if timescan:
        Delay_mm, readbacks = adjust_delayaxis(scan.parameters,readbacks,timezero_mm[0])

    plt.figure(figsize = (7,5))
    plt.suptitle(titlestring, fontsize = 12)

    units = scan.parameters['units'][0]
    label = scan.parameters['name'][0]

    plt.plot(readbacks, np.asarray(data["correlation1"]), label='{}, {}%'.format(det1.split(':')[-1], quantile*100),marker='.', color = 'orange')
    try:
        plt.plot(readbacks, np.asarray(data["correlation2"]), label='{}, {}%'.format(det2.split(':')[-1], quantile*100),marker='.', color = 'blue')
    except:
        print ('plotting only 1 diode')

    plt.xlabel("{} ({})".format(label, units))
    plt.ylabel ("XAS (I0 norm)")
    plt.gca().set_title('correlation')
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.show()

################################################

def Plot_1diode(titlestring, scan, data, quantile, det1, timescan, xlim=None, ylim=None, ylim_pp=None):

    DataDiode1_pump   = np.asarray(data["DataDiode1_pump"])
    DataDiode1_unpump = np.asarray(data["DataDiode1_unpump"])
    Pump_probe_Diode1 = np.asarray(data["Pump_probe_Diode1"])
    goodshots1        = np.asarray(data["goodshots1"])
    readbacks         = np.asarray(data["readbacks"])
    timezero_mm       = np.asarray(data["timezero_mm"])
    
    if timescan:
        Delay_mm, readbacks = adjust_delayaxis(scan.parameters,readbacks,timezero_mm[0])

    #### CH1 ####
    XAS1_pump = DataDiode1_pump[:,0]
    err1_low_pump = DataDiode1_pump[:,1]
    err1_high_pump = DataDiode1_pump[:,2]
    XAS1_unpump = DataDiode1_unpump[:,0]
    err1_low_unpump = DataDiode1_unpump[:,1]
    err1_high_unpump = DataDiode1_unpump[:,2]
    XAS1_pump_probe = Pump_probe_Diode1[:,0]
    err1_low_pump_probe = Pump_probe_Diode1[:,1]
    err1_high_pump_probe = Pump_probe_Diode1[:,2]

    ### plot 1 diode ###
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True)
    plt.suptitle(titlestring, fontsize = 12)
    
    units = scan.parameters['units'][0]
    label = scan.parameters['name'][0]
    
    ax1.plot(readbacks, XAS1_pump, label='ON, {}, {}%'.format(det1.split(':')[-1], quantile*100),marker='.')
    ax1.fill_between(readbacks, err1_low_pump, err1_high_pump, color='lightblue')

    ax1.plot(readbacks, XAS1_unpump, label='OFF, {}, {}%'.format(det1.split(':')[-1], quantile*100),marker='.')
    ax1.fill_between(readbacks, err1_low_unpump, err1_high_unpump, color='navajowhite')

    ax1.set_xlabel("{} ({})".format(label, units))
    ax1.set_ylabel ("XAS Diode 1(I0 norm)")
    ax1.set_title('XAS (fluo)')
    ax1.legend(loc="best")
    ax1.grid()

    ax3.plot(readbacks, XAS1_pump_probe, label='ON, {}, {}%'.format(det1.split(':')[-1], quantile*100),color='green',marker='.')
    ax3.fill_between(readbacks, err1_low_pump_probe, err1_high_pump_probe, alpha = 0.7, color='lightgreen')

    ax3.set_xlabel("{} ({})".format(label, units))
    ax3.set_ylabel ("XAS Diode 2(I0 norm)")
    ax3.set_title('pump probe')
    ax3.legend(loc="best")
    ax3.grid()
    
    ax1.set_xlim(xlim)
    ax3.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax3.set_ylim(ylim_pp)
    
    plt.show()

    return XAS1_pump_probe, readbacks

################################################

def Plot_2diodes_3figs(titlestring, scan, data, quantile, det1, det2, timescan, xlim=None, ylim=None, ylim_pp=None):

    DataDiode1_pump   = np.asarray(data["DataDiode1_pump"])
    DataDiode2_pump   = np.asarray(data["DataDiode2_pump"])
    Pump_probe_Diode1 = np.asarray(data["Pump_probe_Diode1"])
    DataDiode1_unpump = np.asarray(data["DataDiode1_unpump"])
    DataDiode2_unpump = np.asarray(data["DataDiode2_unpump"])
    Pump_probe_Diode2 = np.asarray(data["Pump_probe_Diode2"])
    goodshots1        = np.asarray(data["goodshots1"])
    goodshots2        = np.asarray(data["goodshots2"])
    readbacks         = np.asarray(data["readbacks"])
    timezero_mm       = np.asarray(data["timezero_mm"])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True)

    if timescan:
        Delay_mm, readbacks = adjust_delayaxis(scan.parameters,readbacks,timezero_mm[0])

    #### CH1 ####
    XAS1_pump = DataDiode1_pump[:,0]
    err1_low_pump = DataDiode1_pump[:,1]
    err1_high_pump = DataDiode1_pump[:,2]
    XAS1_unpump = DataDiode1_unpump[:,0]
    err1_low_unpump = DataDiode1_unpump[:,1]
    err1_high_unpump = DataDiode1_unpump[:,2]
    XAS1_pump_probe = Pump_probe_Diode1[:,0]
    err1_low_pump_probe = Pump_probe_Diode1[:,1]
    err1_high_pump_probe = Pump_probe_Diode1[:,2]

    #### CH2 ####
    XAS2_pump = DataDiode2_pump[:,0]
    err2_low_pump = DataDiode2_pump[:,1]
    err2_high_pump = DataDiode2_pump[:,2]
    XAS2_unpump = DataDiode2_unpump[:,0]
    err2_low_unpump = DataDiode2_unpump[:,1]
    err2_high_unpump = DataDiode2_unpump[:,2]
    XAS2_pump_probe = Pump_probe_Diode2[:,0]
    err2_low_pump_probe = Pump_probe_Diode2[:,1]
    err2_high_pump_probe = Pump_probe_Diode2[:,2]

    ### plot 2 diodes ###

    
    plt.suptitle(titlestring, fontsize = 12)
    
    units = scan.parameters['units'][0]
    label = scan.parameters['name'][0]
    
    ax1.plot(readbacks, XAS1_pump, label='ON, {}, {}%'.format(det1.split(':')[-1], quantile*100),marker='.')
    ax1.fill_between(readbacks, err1_low_pump, err1_high_pump, color='lightblue')

    ax1.plot(readbacks, XAS1_unpump, label='OFF, {}, {}%'.format(det1.split(':')[-1], quantile*100),marker='.')
    ax1.fill_between(readbacks, err1_low_unpump, err1_high_unpump, color='navajowhite')

    ax1.set_xlabel("{} ({})".format(label, units))
    ax1.set_ylabel ("XAS Diode 1(I0 norm)")
    ax1.set_title('XAS (fluo)')
    ax1.legend(loc="best")
    ax1.grid()

    ax2.plot(readbacks, XAS2_pump, label='ON, {}, {}%'.format(det2.split(':')[-1], quantile*100),marker='.')
    ax2.fill_between(readbacks, err2_low_pump, err2_high_pump, color='lightblue')

    ax2.plot(readbacks, XAS2_unpump, label='OFF, {}, {}%'.format(det2.split(':')[-1], quantile*100),marker='.')
    ax2.fill_between(readbacks, err2_low_unpump, err2_high_unpump, color='navajowhite')

    ax2.set_xlabel("{} ({})".format(label, units))
    ax2.set_ylabel ("XAS Diode 2(I0 norm)")
    ax2.set_title('XAS (fluo)')
    ax2.legend(loc="best")
    ax2.grid()

    ax3.plot(readbacks, XAS1_pump_probe, label='ON, {}, {}%'.format(det1.split(':')[-1], quantile*100),color='green',marker='.')
    ax3.fill_between(readbacks, err1_low_pump_probe, err1_high_pump_probe, alpha = 0.7, color='lightgreen')

    ax3.plot(readbacks, XAS2_pump_probe, label='ON, {}, {}%'.format(det2.split(':')[-1], quantile*100),color='purple',marker='.')
    ax3.fill_between(readbacks, err2_low_pump_probe, err2_high_pump_probe, alpha = 0.7, color='lavender')

    ax3.set_xlabel("{} ({})".format(label, units))
    ax3.set_ylabel ("XAS Diode 2(I0 norm)")
    ax3.set_title('pump probe')
    ax3.legend(loc="best")
    ax3.grid()
    
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax3.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_ylim(ylim)
    ax3.set_ylim(ylim_pp)
    
    plt.show()

    return XAS1_pump_probe, XAS2_pump_probe, readbacks

#######################################################

def Plot_2diodes_4figs(titlestring, scan, data, quantile, det1, det2, timescan, xlim=None, ylim=None, ylim_pp=None):

    DataDiode1_pump   = np.asarray(data["DataDiode1_pump"])
    DataDiode2_pump   = np.asarray(data["DataDiode2_pump"])
    Pump_probe_Diode1 = np.asarray(data["Pump_probe_Diode1"])
    DataDiode1_unpump = np.asarray(data["DataDiode1_unpump"])
    DataDiode2_unpump = np.asarray(data["DataDiode2_unpump"])
    Pump_probe_Diode2 = np.asarray(data["Pump_probe_Diode2"])
    goodshots1        = np.asarray(data["goodshots1"])
    goodshots2        = np.asarray(data["goodshots2"])
    readbacks         = np.asarray(data["readbacks"])
    timezero_mm       = np.asarray(data["timezero_mm"])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    
    if timescan:
        Delay_mm, readbacks = adjust_delayaxis(scan.parameters,readbacks,timezero_mm[0])

    #### CH1 ####
    XAS1_pump = DataDiode1_pump[:,0]
    err1_low_pump = DataDiode1_pump[:,1]
    err1_high_pump = DataDiode1_pump[:,2]
    XAS1_unpump = DataDiode1_unpump[:,0]
    err1_low_unpump = DataDiode1_unpump[:,1]
    err1_high_unpump = DataDiode1_unpump[:,2]
    XAS1_pump_probe = Pump_probe_Diode1[:,0]
    err1_low_pump_probe = Pump_probe_Diode1[:,1]
    err1_high_pump_probe = Pump_probe_Diode1[:,2]

    #### CH2 ####
    XAS2_pump = DataDiode2_pump[:,0]
    err2_low_pump = DataDiode2_pump[:,1]
    err2_high_pump = DataDiode2_pump[:,2]
    XAS2_unpump = DataDiode2_unpump[:,0]
    err2_low_unpump = DataDiode2_unpump[:,1]
    err2_high_unpump = DataDiode2_unpump[:,2]
    XAS2_pump_probe = Pump_probe_Diode2[:,0]
    err2_low_pump_probe = Pump_probe_Diode2[:,1]
    err2_high_pump_probe = Pump_probe_Diode2[:,2]

    ### plot 2 diodes ###

    
    plt.suptitle(titlestring, fontsize = 12)
    
    units = scan.parameters['units'][0]
    label = scan.parameters['name'][0]
    
    ax1.plot(readbacks, XAS1_pump, label='ON, {}, {}%'.format(det1.split(':')[-1], quantile*100),marker='.')
    ax1.fill_between(readbacks, err1_low_pump, err1_high_pump, color='lightblue')

    ax1.plot(readbacks, XAS1_unpump, label='OFF, {}, {}%'.format(det1.split(':')[-1], quantile*100),marker='.')
    ax1.fill_between(readbacks, err1_low_unpump, err1_high_unpump, color='navajowhite')

    ax1.set_xlabel("{} ({})".format(label, units))
    ax1.set_ylabel ("XAS Diode 1(I0 norm)")
    ax1.set_title('XAS (fluo)')
    ax1.legend(loc="best")
    ax1.grid()

    ax2.plot(readbacks, XAS2_pump, label='ON, {}, {}%'.format(det2.split(':')[-1], quantile*100),marker='.')
    ax2.fill_between(readbacks, err2_low_pump, err2_high_pump, color='lightblue')

    ax2.plot(readbacks, XAS2_unpump, label='OFF, {}, {}%'.format(det2.split(':')[-1], quantile*100),marker='.')
    ax2.fill_between(readbacks, err2_low_unpump, err2_high_unpump, color='navajowhite')

    ax2.set_xlabel("{} ({})".format(label, units))
    ax2.set_ylabel ("XAS Diode 2(I0 norm)")
    ax2.set_title('XAS (fluo)')
    ax2.legend(loc="best")
    ax2.grid()

    ax3.plot(readbacks, XAS1_pump_probe, label='pump-probe, {}, {}%'.format(det1.split(':')[-1], quantile*100),color='green',marker='.')
    ax3.fill_between(readbacks, err1_low_pump_probe, err1_high_pump_probe, alpha = 0.7, color='lightgreen')
    
    ax3.set_xlabel("{} ({})".format(label, units))
    ax3.set_ylabel ("XAS difference")
    ax3.set_title('pump probe')
    ax3.legend(loc="best")
    ax3.grid()

    ax4.plot(readbacks, XAS2_pump_probe, label='pump-probe, {}, {}%'.format(det2.split(':')[-1], quantile*100), color='purple',marker='.')
    ax4.fill_between(readbacks, err2_low_pump_probe, err2_high_pump_probe, alpha = 0.7, color='lavender')

    ax4.set_xlabel("{} ({})".format(label, units))
    ax4.set_ylabel ("XAS difference")
    ax4.set_title('pump probe')
    ax4.legend(loc="best")
    ax4.grid()
    
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax3.set_xlim(xlim)
    ax4.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_ylim(ylim)
    ax3.set_ylim(ylim_pp)
    
    plt.show()

    return XAS1_pump_probe, XAS2_pump_probe, readbacks

###############################################################################

def Plot_2diodes_2figs(titlestring, scan, data, quantile, det1, det2, timescan, xlim=None, ylim=None, ylim_pp=None):

    DataDiode1_pump   = np.asarray(data["DataDiode1_pump"])
    DataDiode2_pump   = np.asarray(data["DataDiode2_pump"])
    Pump_probe_Diode1 = np.asarray(data["Pump_probe_Diode1"])
    DataDiode1_unpump = np.asarray(data["DataDiode1_unpump"])
    DataDiode2_unpump = np.asarray(data["DataDiode2_unpump"])
    Pump_probe_Diode2 = np.asarray(data["Pump_probe_Diode2"])
    goodshots1        = np.asarray(data["goodshots1"])
    goodshots2        = np.asarray(data["goodshots2"])
    readbacks         = np.asarray(data["readbacks"])
    timezero_mm       = np.asarray(data["timezero_mm"])
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    
    if timescan:
        Delay_mm, readbacks = adjust_delayaxis(scan.parameters,readbacks,timezero_mm[0])

    #### CH1 ####
    XAS1_pump = DataDiode1_pump[:,0]
    err1_low_pump = DataDiode1_pump[:,1]
    err1_high_pump = DataDiode1_pump[:,2]
    XAS1_unpump = DataDiode1_unpump[:,0]
    err1_low_unpump = DataDiode1_unpump[:,1]
    err1_high_unpump = DataDiode1_unpump[:,2]
    XAS1_pump_probe = Pump_probe_Diode1[:,0]
    err1_low_pump_probe = Pump_probe_Diode1[:,1]
    err1_high_pump_probe = Pump_probe_Diode1[:,2]

    #### CH2 ####
    XAS2_pump = DataDiode2_pump[:,0]
    err2_low_pump = DataDiode2_pump[:,1]
    err2_high_pump = DataDiode2_pump[:,2]
    XAS2_unpump = DataDiode2_unpump[:,0]
    err2_low_unpump = DataDiode2_unpump[:,1]
    err2_high_unpump = DataDiode2_unpump[:,2]
    XAS2_pump_probe = Pump_probe_Diode2[:,0]
    err2_low_pump_probe = Pump_probe_Diode2[:,1]
    err2_high_pump_probe = Pump_probe_Diode2[:,2]

    ### plot 2 diodes ###
    
    plt.suptitle(titlestring, fontsize = 12)
    
    units = scan.parameters['units'][0]
    label = scan.parameters['name'][0]
    
    ax1.plot(readbacks, XAS1_pump, label='ON, {}, {}%'.format(det1.split(':')[-1], quantile*100),marker='.')
    ax1.fill_between(readbacks, err1_low_pump, err1_high_pump, color='lightblue')

    ax1.plot(readbacks, XAS1_unpump, label='OFF, {}, {}%'.format(det1.split(':')[-1], quantile*100),marker='.')
    ax1.fill_between(readbacks, err1_low_unpump, err1_high_unpump, color='navajowhite')

    ax1.set_xlabel("{} ({})".format(label, units))
    ax1.set_ylabel ("XAS Diode 1(I0 norm)")
    ax1.set_title('XAS (fluo)')
    ax1.legend(loc="best")
    ax1.grid()

    ax2.plot(readbacks, XAS1_pump_probe, label='pump-probe, {}, {}%'.format(det1.split(':')[-1], quantile*100),color='green',marker='.')
    ax2.fill_between(readbacks, err1_low_pump_probe, err1_high_pump_probe, alpha = 0.7, color='lightgreen')

    ax2.plot(readbacks, XAS2_pump_probe, label='pump-probe, {}, {}%'.format(det2.split(':')[-1], quantile*100), color='purple',marker='.')
    ax2.fill_between(readbacks, err2_low_pump_probe, err2_high_pump_probe, alpha = 0.7, color='lavender')

    ax2.set_xlabel("{} ({})".format(label, units))
    ax2.set_ylabel ("XAS difference")
    ax2.set_title('pump probe')
    ax2.legend(loc="best")
    ax2.grid()
    
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)

    ax1.set_ylim(ylim)
    ax2.set_ylim(ylim_pp)

    
    plt.show()

    return XAS1_pump_probe, XAS2_pump_probe, readbacks

################################################

def Plot_AveScans_1diode(titlestring, scan, data, nscans, timescan, xlim=None, ylim=None, ylim_pp=None):

    DataDiode1_pump   = np.asarray(data["DataDiode1_pump"])
    Pump_probe_Diode1 = np.asarray(data["Pump_probe_Diode1"])
    DataDiode1_unpump = np.asarray(data["DataDiode1_unpump"])
    goodshots1        = np.asarray(data["goodshots1"])
    readbacks         = np.asarray(data["readbacks"])
    timezero_mm       = np.asarray(data["timezero_mm"])

    #### CH1 ####
    DataDiode1_pump   = np.reshape(np.asarray(DataDiode1_pump), (nscans, -1, 3))
    DataDiode1_unpump = np.reshape(np.asarray(DataDiode1_unpump), (nscans, -1, 3))
    Pump_probe_Diode1 = np.reshape(np.asarray(Pump_probe_Diode1), (nscans, -1, 3))
    goodshots1        = np.reshape(np.asarray(goodshots1), (nscans, -1))
    
    XAS1_pump        = np.mean(DataDiode1_pump[:,:,0], axis=0)
    err1_low_pump    = np.mean((DataDiode1_pump[:,:,0]-DataDiode1_pump[:,:,1])/np.sqrt(goodshots1), axis=0)
    err1_high_pump   = np.mean((DataDiode1_pump[:,:,2]-DataDiode1_pump[:,:,0])/np.sqrt(goodshots1), axis=0)

    XAS1_unpump      = np.mean(DataDiode1_unpump[:,:,0], axis=0)
    err1_low_unpump  = np.mean((DataDiode1_unpump[:,:,0]-DataDiode1_unpump[:,:,1])/np.sqrt(goodshots1), axis=0)
    err1_high_unpump = np.mean((DataDiode1_unpump[:,:,2]-DataDiode1_unpump[:,:,0])/np.sqrt(goodshots1), axis=0)

    XAS1_pump_probe       = np.mean(Pump_probe_Diode1[:,:,0], axis=0)
    err1_low_pump_probe   = np.sqrt(err1_low_pump**2  + err1_low_unpump**2)
    err1_high_pump_probe  = np.sqrt(err1_high_pump**2 + err1_high_unpump**2)
    
    readbacks         = np.reshape(np.asarray(readbacks), (nscans, -1))[0]
    if timescan:
        Delay_mm, readbacks = adjust_delayaxis(scan.parameters,readbacks,timezero_mm[0])

    ### plot 1 diode ###

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    plt.suptitle(titlestring)

    ax1.fill_between(readbacks, XAS1_pump - err1_low_pump, XAS1_pump + err1_high_pump , label='diode 1 ON', color='royalblue', alpha = 0.8)
    ax1.fill_between(readbacks, XAS1_unpump - err1_low_unpump, XAS1_unpump + err1_high_unpump, label='diode 1 OFF',color='orange', alpha = 0.8)
    ax3.fill_between(readbacks, XAS1_pump_probe - err1_low_pump_probe, XAS1_pump_probe + err1_low_pump_probe, label='pump probe 1',color='limegreen')
    ax3.plot(readbacks, XAS1_pump_probe, color='green', marker='.')

    units = scan.parameters['units'][0]
    label = scan.parameters['name'][0]

    ax1.set_xlabel("{} ({})".format(label, units))
    ax1.set_ylabel ("XAS Diode 1 (I0 norm)")
    ax1.set_title('XAS (fluo)')
    ax1.legend(loc="best")
    ax1.grid()

    ax3.set_xlabel("{} ({})".format(label, units))
    ax3.set_ylabel ("DeltaXAS Diode 1")
    ax3.set_title('pump probe')
    ax3.legend(loc="best")
    ax3.grid()

    plt.show()

    return readbacks, DataDiode1_pump, DataDiode1_unpump, Pump_probe_Diode1, goodshots1

################################################

def Plot_AveScans_2diodes(titlestring, scan, data, nscans, timescan, xlim=None, ylim=None, ylim_pp=None):

    DataDiode1_pump   = np.asarray(data["DataDiode1_pump"])
    DataDiode2_pump   = np.asarray(data["DataDiode2_pump"])
    Pump_probe_Diode1 = np.asarray(data["Pump_probe_Diode1"])
    DataDiode1_unpump = np.asarray(data["DataDiode1_unpump"])
    DataDiode2_unpump = np.asarray(data["DataDiode2_unpump"])
    Pump_probe_Diode2 = np.asarray(data["Pump_probe_Diode2"])
    goodshots1        = np.asarray(data["goodshots1"])
    goodshots2        = np.asarray(data["goodshots2"])
    readbacks         = np.asarray(data["readbacks"])
    timezero_mm       = np.asarray(data["timezero_mm"])

    #### CH1 ####
    DataDiode1_pump   = np.reshape(np.asarray(DataDiode1_pump), (nscans, -1, 3))
    DataDiode1_unpump = np.reshape(np.asarray(DataDiode1_unpump), (nscans, -1, 3))
    Pump_probe_Diode1 = np.reshape(np.asarray(Pump_probe_Diode1), (nscans, -1, 3))
    goodshots1        = np.reshape(np.asarray(goodshots1), (nscans, -1))
    
    XAS1_pump        = np.mean(DataDiode1_pump[:,:,0], axis=0)
    err1_low_pump    = np.mean((DataDiode1_pump[:,:,0]-DataDiode1_pump[:,:,1])/np.sqrt(goodshots1), axis=0)
    err1_high_pump   = np.mean((DataDiode1_pump[:,:,2]-DataDiode1_pump[:,:,0])/np.sqrt(goodshots1), axis=0)

    XAS1_unpump      = np.mean(DataDiode1_unpump[:,:,0], axis=0)
    err1_low_unpump  = np.mean((DataDiode1_unpump[:,:,0]-DataDiode1_unpump[:,:,1])/np.sqrt(goodshots1), axis=0)
    err1_high_unpump = np.mean((DataDiode1_unpump[:,:,2]-DataDiode1_unpump[:,:,0])/np.sqrt(goodshots1), axis=0)

    XAS1_pump_probe       = np.mean(Pump_probe_Diode1[:,:,0], axis=0)
    err1_low_pump_probe   = np.sqrt(err1_low_pump**2  + err1_low_unpump**2)
    err1_high_pump_probe  = np.sqrt(err1_high_pump**2 + err1_high_unpump**2)

    #### CH2 ####
    DataDiode2_pump   = np.reshape(np.asarray(DataDiode2_pump), (nscans, -1, 3))
    DataDiode2_unpump = np.reshape(np.asarray(DataDiode2_unpump), (nscans, -1, 3))
    Pump_probe_Diode2 = np.reshape(np.asarray(Pump_probe_Diode2), (nscans, -1, 3))
    goodshots2        = np.reshape(np.asarray(goodshots2), (nscans, -1))

    XAS2_pump        = np.mean(DataDiode2_pump[:,:,0], axis=0)
    err2_low_pump    = np.mean((DataDiode2_pump[:,:,0]-DataDiode2_pump[:,:,1])/np.sqrt(goodshots2), axis=0)
    err2_high_pump   = np.mean((DataDiode2_pump[:,:,2]-DataDiode2_pump[:,:,0])/np.sqrt(goodshots2), axis=0)

    XAS2_unpump      = np.mean(DataDiode2_unpump[:,:,0], axis=0)
    err2_low_unpump  = np.mean((DataDiode2_unpump[:,:,0]-DataDiode2_unpump[:,:,1])/np.sqrt(goodshots2), axis=0)
    err2_high_unpump = np.mean((DataDiode2_unpump[:,:,2]-DataDiode2_unpump[:,:,0])/np.sqrt(goodshots2), axis=0)

    XAS2_pump_probe       = np.mean(Pump_probe_Diode2[:,:,0], axis=0)
    err2_low_pump_probe   = np.sqrt(err2_low_pump**2  + err2_low_unpump**2)
    err2_high_pump_probe  = np.sqrt(err2_high_pump**2 + err2_high_unpump**2)
    
    readbacks         = np.reshape(np.asarray(readbacks), (nscans, -1))[0]
    if timescan:
        Delay_mm, readbacks = adjust_delayaxis(scan.parameters,readbacks,timezero_mm[0])
    
    ### plot 2 diodes ###
    
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    plt.suptitle(titlestring)

    units = scan.parameters['units'][0]
    label = scan.parameters['name'][0]

    ax1.fill_between(readbacks, XAS1_pump - err1_low_pump, XAS1_pump + err1_high_pump , label='diode 1 ON', color='royalblue', alpha = 0.8)
    ax1.fill_between(readbacks, XAS1_unpump - err1_low_unpump, XAS1_unpump + err1_high_unpump, label='diode 1 OFF',color='orange', alpha = 0.8)
    ax3.fill_between(readbacks, XAS1_pump_probe - err1_low_pump_probe, XAS1_pump_probe + err1_low_pump_probe, label='pump probe 1',color='limegreen')
    ax3.plot(readbacks, XAS1_pump_probe, color='green', marker='.')

    ax2.fill_between(readbacks, XAS2_pump - err2_low_pump, XAS2_pump + err2_high_pump , label='diode 2 ON', color='royalblue', alpha = 0.8)
    ax2.fill_between(readbacks, XAS2_unpump - err2_low_unpump, XAS2_unpump + err2_high_unpump, label='diode 2 OFF',color='orange', alpha = 0.8)
    ax4.fill_between(readbacks, XAS2_pump_probe - err2_low_pump_probe, XAS2_pump_probe + err2_low_pump_probe, label='pump probe 2', color='limegreen')
    ax4.plot(readbacks, XAS2_pump_probe, color='green', marker='.')

    ax1.set_xlabel("{} ({})".format(label, units))
    ax1.set_ylabel ("XAS Diode 1 (I0 norm)")
    ax1.set_title('XAS (fluo)')
    ax1.legend(loc="best")
    ax1.grid()

    ax3.set_xlabel("{} ({})".format(label, units))
    ax3.set_ylabel ("DeltaXAS Diode 1")
    ax3.set_title('pump probe')
    ax3.legend(loc="best")
    ax3.grid()

    ax2.set_xlabel("{} ({})".format(label, units))
    ax2.set_ylabel ("XAS Diode 2 (I0 norm)")
    ax2.set_title('XAS (fluo)')
    ax2.legend(loc="best")
    ax2.grid()

    ax4.set_xlabel("{} ({})".format(label, units))
    ax4.set_ylabel ("DeltaXAS Diode 2")
    ax4.set_title('pump probe')
    ax4.legend(loc="best")
    ax4.grid()

    plt.show()

    return readbacks, DataDiode1_pump, DataDiode2_pump, DataDiode1_unpump, DataDiode2_unpump, Pump_probe_Diode1, Pump_probe_Diode2, goodshots1, goodshots2
    #return Delay_fs, XAS1_pump_probe, XAS2_pump_probe, XAS1_pump, XAS2_pump, XAS1_unpump, XAS2_unpump, goodshots1_avg, goodshots2_avg
    
################################################

def Plot_2diodes_Averaged_1fig(titlestring, scan, data, timescan, nscans=1, xlim=None, ylim=None, ylim_pp=None):

    DataDiode1_pump   = np.asarray(data["DataDiode1_pump"])
    DataDiode2_pump   = np.asarray(data["DataDiode2_pump"])
    Pump_probe_Diode1 = np.asarray(data["Pump_probe_Diode1"])
    DataDiode1_unpump = np.asarray(data["DataDiode1_unpump"])
    DataDiode2_unpump = np.asarray(data["DataDiode2_unpump"])
    Pump_probe_Diode2 = np.asarray(data["Pump_probe_Diode2"])
    goodshots1        = np.asarray(data["goodshots1"])
    goodshots2        = np.asarray(data["goodshots2"])
    readbacks         = np.asarray(data["readbacks"])
    timezero_mm       = np.asarray(data["timezero_mm"])

    if timescan:
        Delay_mm, readbacks = adjust_delayaxis(scan.parameters,readbacks,timezero_mm[0])

    readbacks         = np.reshape(np.asarray(readbacks), (nscans, -1))[0]

    #### CH1 ####
    DataDiode1_pump   = np.reshape(np.asarray(DataDiode1_pump), (nscans, -1, 3))
    DataDiode1_unpump = np.reshape(np.asarray(DataDiode1_unpump), (nscans, -1, 3))
    Pump_probe_Diode1 = np.reshape(np.asarray(Pump_probe_Diode1), (nscans, -1, 3))
    goodshots1        = np.reshape(np.asarray(goodshots1), (nscans, -1))
    
    XAS1_pump        = np.mean(DataDiode1_pump[:,:,0], axis=0)
    err1_low_pump    = np.mean((DataDiode1_pump[:,:,0]-DataDiode1_pump[:,:,1])/np.sqrt(goodshots1), axis=0)
    err1_high_pump   = np.mean((DataDiode1_pump[:,:,2]-DataDiode1_pump[:,:,0])/np.sqrt(goodshots1), axis=0)

    XAS1_unpump      = np.mean(DataDiode1_unpump[:,:,0], axis=0)
    err1_low_unpump  = np.mean((DataDiode1_unpump[:,:,0]-DataDiode1_unpump[:,:,1])/np.sqrt(goodshots1), axis=0)
    err1_high_unpump = np.mean((DataDiode1_unpump[:,:,2]-DataDiode1_unpump[:,:,0])/np.sqrt(goodshots1), axis=0)

    XAS1_pump_probe       = np.mean(Pump_probe_Diode1[:,:,0], axis=0)
    err1_low_pump_probe   = np.sqrt(err1_low_pump**2  + err1_low_unpump**2)
    err1_high_pump_probe  = np.sqrt(err1_high_pump**2 + err1_high_unpump**2)

    #### CH2 ####
    DataDiode2_pump   = np.reshape(np.asarray(DataDiode2_pump), (nscans, -1, 3))
    DataDiode2_unpump = np.reshape(np.asarray(DataDiode2_unpump), (nscans, -1, 3))
    Pump_probe_Diode2 = np.reshape(np.asarray(Pump_probe_Diode2), (nscans, -1, 3))
    goodshots2        = np.reshape(np.asarray(goodshots2), (nscans, -1))

    XAS2_pump        = np.mean(DataDiode2_pump[:,:,0], axis=0)
    err2_low_pump    = np.mean((DataDiode2_pump[:,:,0]-DataDiode2_pump[:,:,1])/np.sqrt(goodshots2), axis=0)
    err2_high_pump   = np.mean((DataDiode2_pump[:,:,2]-DataDiode2_pump[:,:,0])/np.sqrt(goodshots2), axis=0)

    XAS2_unpump      = np.mean(DataDiode2_unpump[:,:,0], axis=0)
    err2_low_unpump  = np.mean((DataDiode2_unpump[:,:,0]-DataDiode2_unpump[:,:,1])/np.sqrt(goodshots2), axis=0)
    err2_high_unpump = np.mean((DataDiode2_unpump[:,:,2]-DataDiode2_unpump[:,:,0])/np.sqrt(goodshots2), axis=0)

    XAS2_pump_probe       = np.mean(Pump_probe_Diode2[:,:,0], axis=0)
    err2_low_pump_probe   = np.sqrt(err2_low_pump**2  + err2_low_unpump**2)
    err2_high_pump_probe  = np.sqrt(err2_high_pump**2 + err2_high_unpump**2)

    ### Average the diodes ###

    XAS_mean_pump = (XAS1_pump+XAS2_pump)/2
    XAS_mean_unpump = (XAS1_unpump+XAS2_unpump)/2

    offset1 = np.average(XAS1_unpump[0:5])
    offset2 = np.average(XAS2_unpump[0:5])

    ave_unpump = (XAS1_unpump-offset1 + XAS2_unpump-offset2)/2
    ave_unpump_err_l = np.sqrt(err1_low_unpump**2+err2_low_unpump**2)
    ave_unpump_err_h = np.sqrt(err1_high_unpump**2+err1_high_unpump**2)

    ave_pump = (XAS1_pump-offset1 + XAS2_pump-offset2)/2
    ave_pump_err_l = np.sqrt(err1_low_pump**2+err2_low_pump**2)
    ave_pump_err_h = np.sqrt(err1_high_pump**2+err1_high_pump**2)

    ave_pp1 = ave_pump - ave_unpump
    ave_pp_err_l1 = np.sqrt(ave_unpump_err_l**2+ave_pump_err_l**2)
    ave_pp_err_h1 = np.sqrt(ave_unpump_err_h**2+ave_pump_err_h**2)

    fig, ax1= plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)
    plt.suptitle(titlestring)
    
    units = scan.parameters['units'][0]
    label = scan.parameters['name'][0]

    ax1.plot(readbacks,ave_pump,lw=1,marker='o',markersize=3,label='ON') 
    ax1.plot(readbacks,ave_unpump,lw=1,marker='o',markersize=3,label='ON') 
    ax1.set_xlabel("{} ({})".format(label, units))
    ax1.set_ylabel('XAS Intensity (a.u.)')
    ax1.grid()

    ax2 = plt.twinx(ax1)
    ax2.axes.errorbar(readbacks, ave_pp1, (ave_pp_err_l1, ave_pp_err_h1), 
                  lw=1,color='green', markersize=0,capsize=1,capthick=0.5,
                       ecolor='green',elinewidth=0.5,label='pump-probe')
    ax2.fill_between(readbacks,ave_pp1-ave_pp_err_l1, ave_pp1+ave_pp_err_h1,color='lightgreen', alpha=0.5)
    ax2.axhline(0,ls='--',c='k',lw=1)
    ax2.set_ylabel('Delta XAS Intensity (a.u.)')

    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim_pp)

    plt.show()

    return readbacks, ave_pp1

################################################

def Plot_2diodes_Averaged_2figs(titlestring, scan, data, timescan, nscans=1, xlim=None, ylim=None, ylim_pp=None):

    DataDiode1_pump   = np.asarray(data["DataDiode1_pump"])
    DataDiode2_pump   = np.asarray(data["DataDiode2_pump"])
    Pump_probe_Diode1 = np.asarray(data["Pump_probe_Diode1"])
    DataDiode1_unpump = np.asarray(data["DataDiode1_unpump"])
    DataDiode2_unpump = np.asarray(data["DataDiode2_unpump"])
    Pump_probe_Diode2 = np.asarray(data["Pump_probe_Diode2"])
    goodshots1        = np.asarray(data["goodshots1"])
    goodshots2        = np.asarray(data["goodshots2"])
    readbacks         = np.asarray(data["readbacks"])
    timezero_mm       = np.asarray(data["timezero_mm"])

    if timescan:
        Delay_mm, readbacks = adjust_delayaxis(scan.parameters,readbacks,timezero_mm[0])

    readbacks         = np.reshape(np.asarray(readbacks), (nscans, -1))[0]

    #### CH1 ####
    DataDiode1_pump   = np.reshape(np.asarray(DataDiode1_pump), (nscans, -1, 3))
    DataDiode1_unpump = np.reshape(np.asarray(DataDiode1_unpump), (nscans, -1, 3))
    Pump_probe_Diode1 = np.reshape(np.asarray(Pump_probe_Diode1), (nscans, -1, 3))
    goodshots1        = np.reshape(np.asarray(goodshots1), (nscans, -1))
    
    XAS1_pump        = np.mean(DataDiode1_pump[:,:,0], axis=0)
    err1_low_pump    = np.mean((DataDiode1_pump[:,:,0]-DataDiode1_pump[:,:,1])/np.sqrt(goodshots1), axis=0)
    err1_high_pump   = np.mean((DataDiode1_pump[:,:,2]-DataDiode1_pump[:,:,0])/np.sqrt(goodshots1), axis=0)

    XAS1_unpump      = np.mean(DataDiode1_unpump[:,:,0], axis=0)
    err1_low_unpump  = np.mean((DataDiode1_unpump[:,:,0]-DataDiode1_unpump[:,:,1])/np.sqrt(goodshots1), axis=0)
    err1_high_unpump = np.mean((DataDiode1_unpump[:,:,2]-DataDiode1_unpump[:,:,0])/np.sqrt(goodshots1), axis=0)

    XAS1_pump_probe       = np.mean(Pump_probe_Diode1[:,:,0], axis=0)
    err1_low_pump_probe   = np.sqrt(err1_low_pump**2  + err1_low_unpump**2)
    err1_high_pump_probe  = np.sqrt(err1_high_pump**2 + err1_high_unpump**2)

    #### CH2 ####
    DataDiode2_pump   = np.reshape(np.asarray(DataDiode2_pump), (nscans, -1, 3))
    DataDiode2_unpump = np.reshape(np.asarray(DataDiode2_unpump), (nscans, -1, 3))
    Pump_probe_Diode2 = np.reshape(np.asarray(Pump_probe_Diode2), (nscans, -1, 3))
    goodshots2        = np.reshape(np.asarray(goodshots2), (nscans, -1))

    XAS2_pump        = np.mean(DataDiode2_pump[:,:,0], axis=0)
    err2_low_pump    = np.mean((DataDiode2_pump[:,:,0]-DataDiode2_pump[:,:,1])/np.sqrt(goodshots2), axis=0)
    err2_high_pump   = np.mean((DataDiode2_pump[:,:,2]-DataDiode2_pump[:,:,0])/np.sqrt(goodshots2), axis=0)

    XAS2_unpump      = np.mean(DataDiode2_unpump[:,:,0], axis=0)
    err2_low_unpump  = np.mean((DataDiode2_unpump[:,:,0]-DataDiode2_unpump[:,:,1])/np.sqrt(goodshots2), axis=0)
    err2_high_unpump = np.mean((DataDiode2_unpump[:,:,2]-DataDiode2_unpump[:,:,0])/np.sqrt(goodshots2), axis=0)

    XAS2_pump_probe       = np.mean(Pump_probe_Diode2[:,:,0], axis=0)
    err2_low_pump_probe   = np.sqrt(err2_low_pump**2  + err2_low_unpump**2)
    err2_high_pump_probe  = np.sqrt(err2_high_pump**2 + err2_high_unpump**2)

    ### Average the diodes ###

    XAS_mean_pump = (XAS1_pump+XAS2_pump)/2
    XAS_mean_unpump = (XAS1_unpump+XAS2_unpump)/2

    offset1 = np.average(XAS1_unpump[0:5])
    offset2 = np.average(XAS2_unpump[0:5])

    ave_unpump = (XAS1_unpump-offset1 + XAS2_unpump-offset2)/2
    ave_unpump_err_l = np.sqrt(err1_low_unpump**2+err2_low_unpump**2)
    ave_unpump_err_h = np.sqrt(err1_high_unpump**2+err1_high_unpump**2)

    ave_pump = (XAS1_pump-offset1 + XAS2_pump-offset2)/2
    ave_pump_err_l = np.sqrt(err1_low_pump**2+err2_low_pump**2)
    ave_pump_err_h = np.sqrt(err1_high_pump**2+err1_high_pump**2)

    ave_pp1 = ave_pump - ave_unpump
    ave_pp_err_l1 = np.sqrt(ave_unpump_err_l**2+ave_pump_err_l**2)
    ave_pp_err_h1 = np.sqrt(ave_unpump_err_h**2+ave_pump_err_h**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True)
    plt.suptitle(titlestring, fontsize = 12)

    units = scan.parameters['units'][0]
    label = scan.parameters['name'][0]

    ax1.fill_between(readbacks,ave_unpump-ave_unpump_err_l, ave_unpump+ave_unpump_err_h,color='royalblue',label='unpumped')
    ax1.fill_between(readbacks,ave_pump-ave_pump_err_l, ave_pump+ave_pump_err_h,color='orange',label='pumped')

    ax2.axes.errorbar(readbacks, ave_pp1, (ave_pp_err_l1, ave_pp_err_h1), 
                 lw=1,color='green', markersize=0,capsize=1,capthick=0.5,
                      ecolor='green',elinewidth=0.5,label='pump-probe')
    ax2.fill_between(readbacks,ave_pp1-ave_pp_err_l1, ave_pp1+ave_pp_err_h1,color='lightgreen')
    ax2.axhline(0,ls='--',c='k',lw=1)

    ax1.set_xlabel("{} ({})".format(label, units))
    ax1.set_ylabel('XAS Intensity (a.u.)')
    ax1.grid()
    ax1.legend(loc='best')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    ax2.set_xlabel("{} ({})".format(label, units))
    ax2.set_ylabel('XAS Intensity (a.u.)')
    ax2.grid()
    ax2.legend(loc='best')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim_pp)

    return readbacks, ave_pp1

################################################


