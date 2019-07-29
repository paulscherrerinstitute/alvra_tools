#!/usr/bin/env python3

if __name__ == "__main__":
    import argparse

    description = "Prints the pedestal file that was recorded closest in time to a given data file."

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description)

    parser.add_argument("filename", help="Data file name", nargs="*")

    clargs = parser.parse_args()


#pede_path = "/sf/{beamline}/data/{pgroup}/res/JF_pedestals/"


import os
import pathlib


def _get_base_folder(fname):
    fname = fname.split(os.sep)
    return os.sep.join(fname[:5])

def _get_detector_name(fname):
    fname = fname.split(os.sep)
    detname = fname[-1].split(".")[-2]
    return detname


def find_pede(fname):
    fpath = pathlib.Path(fname)
    fmtime = fpath.stat().st_mtime

    pede_path = _get_base_folder(fname) + "/res/JF_pedestals"
    pede_path = pathlib.Path(pede_path)

    detector_name = _get_detector_name(fname)
    print(detector_name)
    pede = None
    min_time_diff = float('inf')
    for entry in pede_path.iterdir():
        if entry.is_file() and detector_name in entry.name:
            pmtime = entry.stat().st_mtime
            time_diff = abs(pmtime - fmtime)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                pede = entry
    return pede



if __name__ == "__main__":
    maxlen = max(len(fn) for fn in clargs.filename)
    for fn in clargs.filename:
        pede = find_pede(fn)
        print(fn.ljust(maxlen), pede)


