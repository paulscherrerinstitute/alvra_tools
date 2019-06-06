
channel_JF_images      = "data/JF02T09V01/data"
channel_JF_pulse_ids   = "data/JF02T09V01/pulse_id"

channel_Events         = "data/SAR-CVME-TIFALL5:EvtSet/data"                  #Event code channel

channel_delay          = "data/SLAAR11-LMOT-M451:ENC_1_BS/data"               #Delay of pump-probe delay stage
channel_delay_NPP      = "data/SLAAR11-LTIM01-EVR0:DUMMY_PV5_NBS/data"        #Delay in mm from the NPP (spatial encoding TT) small stage

channel_BS_pulse_ids   = "data/SLAAR11-LMOT-M451:ENC_1_BS/pulse_id"           #Pulse ids taken from pump-probe delay stage (100 Hz)
channel_PIPS_trans     = "data/SARES11-GES1:CH1_VAL_GET/data"                 #X-ray TRANS diode to PRIME Keysight (channel 1)
channel_PIPS_fluo      = "data/SARES11-GES1:CH2_VAL_GET/data"                 #X-ray FLUO diode to PRIME Keysight (channel 2)

#channel_Izero          = "data/SARES11-LSCP10-FNS:CH1:VAL_GET/data"          #Izero diode to PRIME Ioxos (channel 1)
#channel_Izero2         = "data/SARES11-LSCP10-FNS:CH0:VAL_GET/data"          #Izero diode to PRIME Ioxos (channel 0)
#channel_Izero3         = "data/SARES11-LSCP10-FNS:CH3:VAL_GET/data"          #Izero diode to PRIME Ioxos (channel 3)
#channel_Izero4         = "data/SARES11-LSCP10-FNS:CH2:VAL_GET/data"          #Izero diode to PRIME Ioxos (channel 2)

channel_Izero          = "data/SAROP11-CVME-PBPS2:Lnk9Ch12-DATA-SUM/data"     #Izero diode to Wavedream PBPS117 (Up - PDU)
channel_Izero2         = "data/SAROP11-CVME-PBPS2:Lnk9Ch13-DATA-SUM/data"     #Izero diode to Wavedream PBPS117 (Down - PDD)
channel_Izero3         = "data/SAROP11-CVME-PBPS2:Lnk9Ch14-DATA-SUM/data"     #Izero diode to Wavedream PBPS117 (Right - PDR)
channel_Izero4         = "data/SAROP11-CVME-PBPS2:Lnk9Ch15-DATA-SUM/data"     #Izero diode to Wavedream PBPS117 (Left - PDL)

channel_LaserDiode     = "data/SLAAR11-LSCP1-FNS:CH0:VAL_GET/data"            #Laser diode to ESA Laser Ioxos 
channel_Laser_refDiode = "data/SLAAR11-LSCP1-FNS:CH2:VAL_GET/data"            #Laser diode leaking from beampath, to ESA Laser Ioxos 
channel_Laser_diag     = "data/SLAAR11-LSCP1-FNS:CH4:VAL_GET/data"            #Laser diode on the DIAG table to laser Ioxos

channel_PALM_eTOF      = "data/SAROP11-PALMK118:CH2_BUFFER/data"              #PALM eTof from Ch2
channel_PALM_drift     = "data/SAROP11-PALMK118:CH2_VAL_GET/data"             #PALM drift
channel_BAM            = "data/S10BC01-DBAM070:EOM1_T1/data"                  #BAM arrival time

#channel_laser_yaw      = "data/SLAAR11-LTIM01-EVR0:DUMMY_PV2_NBS/data"        #Laser mirror rotation Smaract motor
#channel_laser_pitch    = "data/SLAAR11-LTIM01-EVR0:DUMMY_PV4_NBS/data"        #Laser mirror pitch Smaract motor

channel_position_X     = "data/SLAAR11-LTIM01-EVR0:DUMMY_PV1_NBS/data"        #Huber stage X position. 
channel_position_Y     = "data/SLAAR11-LTIM01-EVR0:DUMMY_PV2_NBS/data"        #Huber stage Y position. 
channel_position_Z     = "data/SLAAR11-LTIM01-EVR0:DUMMY_PV3_NBS/data"        #Huber stage Z position. 
channel_energy         = "data/SLAAR11-LTIM01-EVR0:DUMMY_PV4_NBS/data"        #Mono energy in eV

#channel_LAM_delaystage = "data/SLAAR11-LTIM01-EVR0:DUMMY_PV2_NBS/data"        #LAM internal stage position in mm
#channel_LAM_stepper    = "data/SLAAR11-LTIM01-EVR0:DUMMY_PV10_NBS/data"       #LAM stepper motor (used for feedback)



