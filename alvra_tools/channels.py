detector_name = "JF02T09V02"

channel_JF_images      = "{}/data".format(detector_name)
channel_JF_pulse_ids   = "{}/pulse_id".format(detector_name)

channel_Events         = "SAR-CVME-TIFALL5:EvtSet/data"                  #Event code channel

channel_delay          = "SLAAR11-LMOT-M451:ENC_1_BS/data"               #Delay of pump-probe delay stage
channel_delay_NPP      = "SLAAR11-LTIM01-EVR0:DUMMY_PV5_NBS/data"        #Delay in mm from the NPP (spatial encoding TT) small stage

channel_BS_pulse_ids   = "SLAAR11-LMOT-M451:ENC_1_BS/pulse_id"           #Pulse ids taken from pump-probe delay stage (100 Hz)
channel_PIPS_trans     = "SARES11-GES1:CH1_VAL_GET/data"                 #X-ray TRANS diode to PRIME Keysight (channel 1)
channel_PIPS_fluo      = "SARES11-GES1:CH2_VAL_GET/data"                 #X-ray FLUO diode to PRIME Keysight (channel 2)

#channel_Izero          = "SARES11-LSCP10-FNS:CH1:VAL_GET/data"          #Izero diode to PRIME Ioxos (channel 1)
#channel_Izero2         = "SARES11-LSCP10-FNS:CH0:VAL_GET/data"          #Izero diode to PRIME Ioxos (channel 0)
#channel_Izero3         = "SARES11-LSCP10-FNS:CH3:VAL_GET/data"          #Izero diode to PRIME Ioxos (channel 3)
#channel_Izero4         = "SARES11-LSCP10-FNS:CH2:VAL_GET/data"          #Izero diode to PRIME Ioxos (channel 2)

channel_Izero          = "SAROP11-CVME-PBPS2:Lnk9Ch12-DATA-SUM/data"     #Izero diode to Wavedream PBPS117 (Up - PDU)
channel_Izero2         = "SAROP11-CVME-PBPS2:Lnk9Ch13-DATA-SUM/data"     #Izero diode to Wavedream PBPS117 (Down - PDD)
channel_Izero3         = "SAROP11-CVME-PBPS2:Lnk9Ch14-DATA-SUM/data"     #Izero diode to Wavedream PBPS117 (Right - PDR)
channel_Izero4         = "SAROP11-CVME-PBPS2:Lnk9Ch15-DATA-SUM/data"     #Izero diode to Wavedream PBPS117 (Left - PDL)

channel_LaserDiode     = "SLAAR11-LSCP1-FNS:CH0:VAL_GET/data"            #Laser diode to ESA Laser Ioxos 
channel_Laser_refDiode = "SLAAR11-LSCP1-FNS:CH2:VAL_GET/data"            #Laser diode leaking from beampath, to ESA Laser Ioxos 
channel_Laser_diag     = "SLAAR11-LSCP1-FNS:CH4:VAL_GET/data"            #Laser diode on the DIAG table to laser Ioxos

channel_PALM_eTOF      = "SAROP11-PALMK118:CH2_BUFFER/data"              #PALM eTof from Ch2
channel_PALM_drift     = "SAROP11-PALMK118:CH2_VAL_GET/data"             #PALM drift
channel_BAM            = "S10BC01-DBAM070:EOM1_T1/data"                  #BAM arrival time

#channel_laser_yaw      = "SLAAR11-LTIM01-EVR0:DUMMY_PV2_NBS/data"        #Laser mirror rotation Smaract motor
#channel_laser_pitch    = "SLAAR11-LTIM01-EVR0:DUMMY_PV4_NBS/data"        #Laser mirror pitch Smaract motor

channel_position_X     = "SLAAR11-LTIM01-EVR0:DUMMY_PV1_NBS/data"        #Huber stage X position. 
channel_position_Y     = "SLAAR11-LTIM01-EVR0:DUMMY_PV2_NBS/data"        #Huber stage Y position. 
channel_position_Z     = "SLAAR11-LTIM01-EVR0:DUMMY_PV3_NBS/data"        #Huber stage Z position. 
channel_energy         = "SLAAR11-LTIM01-EVR0:DUMMY_PV4_NBS/data"        #Mono energy in eV

#channel_LAM_delaystage = "SLAAR11-LTIM01-EVR0:DUMMY_PV2_NBS/data"        #LAM internal stage position in mm
#channel_LAM_stepper    = "SLAAR11-LTIM01-EVR0:DUMMY_PV10_NBS/data"       #LAM stepper motor (used for feedback)



