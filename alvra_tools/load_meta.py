import h5py
import json
import numpy as np


def load_gain_data(fn):
    """
    Load the gains file
    (usually located in /sf/alvra/config/jungfrau/gainMaps/JF02T09V01/)
    """
    with h5py.File(fn, "r") as f:
        G = f["gains"][:]
    return G

def load_pede_data(fn):
    """
    Load the pedestal file
    (usually located in /sf/alvra/data/*pgroup*/res/JF_pedestals)
    """
    with h5py.File(fn, "r") as f:
        P = f["gains"][:]
        pixel_mask = f["pixel_mask"][:]
    return P, pixel_mask

def load_scan_data(fn):
    with open(fn) as f:
        data = json.load(f)
    return data["scan_files"]

def load_scan_readback(fn):
    with open(fn) as f:
        data = json.load(f)
    return np.array(data["scan_readbacks"]).ravel()



