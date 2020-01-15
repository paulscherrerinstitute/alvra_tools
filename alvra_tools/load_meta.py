import json
import numpy as np


def load_scan_data(fn):
    with open(fn) as f:
        data = json.load(f)
    return data["scan_files"]

def load_scan_readback(fn):
    with open(fn) as f:
        data = json.load(f)
    return np.array(data["scan_readbacks"]).ravel()
