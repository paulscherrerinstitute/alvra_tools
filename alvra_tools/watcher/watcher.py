#!/usr/bin/env python3

# Command line argument handling
import argparse

description = """
Monitor the given pgroup's data folders and run runner.sh for every raw file without corresponding cropped file.
Inside runner.sh $source and $target may be used as arguments to a command.
Adjust convert_input_output() in convertio.py to reflect the raw-cropped filename transformation.
"""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description)

parser.add_argument("-p", "--pgroup",  help="pgroup", default="p17589")
parser.add_argument("-s", "--sleep",   help="Time to wait between checks in seconds", default=10, type=int)
parser.add_argument("-n", "--name",    help="Data-set name", default="*")
parser.add_argument("-r", "--reverse", help="Reverse file list", action="store_true")

clargs = parser.parse_args()


# Folders & filenames
pgroup = clargs.pgroup
name = clargs.name

dir_base = "/sf/alvra/data/"
#dir_data = dir_base + pgroup + "/raw/scan_data"
dir_data = dir_base + pgroup + "/raw"
dir_crop = dir_base + pgroup + "/res/cropped_data"

dir_data = dir_data + "/" + name
dir_crop = dir_crop + "/" + name

fnames_data = dir_data + "/*.h5"
fnames_crop = dir_crop + "/*.h5"
fnames_lock = dir_crop + "/*.h5.lock"
fnames_BS   = dir_data + "/*.BSREAD.h5"


################################################################################
import os
from time import sleep
from glob import glob
from convertio import convert_input_output, convert_JF_BS


while True:
    print("=" * 100)

    fnames_data_current = glob(fnames_data)
    fnames_crop_current = glob(fnames_crop)
    fnames_lock_current = glob(fnames_lock)
    fnames_BS_current   = glob(fnames_BS)

#    print(fnames_data_current, fnames_data)
#    print(fnames_crop_current, fnames_crop)
#    print(fnames_lock_current, fnames_lock)

    for fn in sorted(fnames_data_current, reverse=clargs.reverse):
        new_fn = convert_input_output(fn, dir_crop)
        if new_fn is None:
            continue
        fn_BS = convert_JF_BS(fn)
        if fn_BS not in fnames_BS_current:
            print("{} has no BSREAD file yet".format(new_fn))
            continue
        new_fn_lock = new_fn + ".lock"
        if new_fn in fnames_crop_current:
            if new_fn_lock in fnames_lock_current:
                print("{} lock removed".format(new_fn))
                os.system("rm \"{}\"".format(new_fn_lock))
            print("{} exists already".format(new_fn))
            continue
        if new_fn_lock in fnames_lock_current:
            print("{} is locked".format(new_fn))
            continue
        print("Process:", fn, "-->", new_fn)
#        continue
        os.environ["source"] = fn
        os.environ["target"] = new_fn
#        os.system("bash ./runner.sh")
        os.system("sbatch ./runner.sh")
        print("{} locked".format(new_fn))
        os.system("mkdir -p \"$(dirname {})\"".format(new_fn_lock))
        os.system("touch \"{}\"".format(new_fn + ".lock"))

    sleep(clargs.sleep)



