#!/usr/bin/env python3

# Command line argument handling
import argparse

description = """
Monitor the given pgroup's data folders and run runner.sh for every raw file without corresponding cropped file.
Inside runner.sh $script, $source and $target may be used to build a command.
Adjust convert_input_output() in convertio.py to reflect the raw-cropped filename transformation.
"""
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description)

parser.add_argument("-p", "--processor", help="Processor script (if None, dry-run is enabled)")
parser.add_argument("-g", "--group",     help="pgroup", default="p17589")
parser.add_argument("-n", "--name",      help="Data-set name", default="*")
parser.add_argument("-d", "--dry-run",   help="Simulate, don't actually run anything", action="store_true")
parser.add_argument("-r", "--reverse",   help="Reverse file list", action="store_true")
parser.add_argument("-s", "--sleep",     help="Time to wait between checks in seconds", default=10, type=int)

clargs = parser.parse_args()


if not clargs.dry_run and clargs.processor is None:
    print("No processor given, will dry run...")
    clargs.dry_run = True 


# Folders & filenames
name = clargs.name
pgroup = clargs.group

pgroup = pgroup if pgroup.startswith("p") else "p" + pgroup
pfolder = pgroup[:3]

dir_data = f"/sf/alvra/data/{pgroup}/raw"
dir_crop = f"/das/work/{pfolder}/{pgroup}/cropped_data"

dir_data_name = dir_data
dir_crop_name = dir_crop
if name is not "*":
    dir_data_name += "/" + name
    dir_crop_name += "/" + name

fnames_data = dir_data_name + "/**/*.JF02T09V*.h5"
fnames_crop = dir_crop_name + "/**/*.h5"
fnames_lock = dir_crop_name + "/**/*.h5.lock"
fnames_BS   = dir_data_name + "/**/*.BSREAD.h5"


################################################################################
import os
from time import sleep
from utils import rglob
from convertio import convert_input_output, convert_JF_BS
from find_pede import find_pede



if __name__ == "__main__":
    while True:
        print("=" * 100)

        fnames_data_current = rglob(fnames_data)
        fnames_crop_current = rglob(fnames_crop)
        fnames_lock_current = rglob(fnames_lock)
        fnames_BS_current   = rglob(fnames_BS)

#        print(fnames_data_current, fnames_data)
#        print(fnames_crop_current, fnames_crop)
#        print(fnames_lock_current, fnames_lock)
#        print(fnames_BS_current,   fnames_BS)

        for fn in sorted(fnames_data_current, reverse=clargs.reverse):
            new_fn = convert_input_output(fn, dir_data, dir_crop)
            if new_fn is None:
                continue
            fn_BS = convert_JF_BS(fn)
            if fn_BS not in fnames_BS_current:
                print("{} has no BSREAD file yet".format(fn))
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
            if clargs.dry_run:
                continue
            os.environ["script"] = clargs.processor
            os.environ["source"] = fn
            os.environ["target"] = new_fn

            pede = find_pede(fn)
            print("with pedestal:", pede)
            os.environ["options"] = "-p {}".format(pede)

#            os.system("bash ./runner.sh")
            exit_status = os.system("sbatch -p hour {}/runner.sh".format(os.path.realpath(os.path.dirname(__file__))))
            if exit_status != 0:
                continue
            print("{} locked".format(new_fn))
            os.system("mkdir -p \"$(dirname {})\"".format(new_fn_lock))
            os.system("touch \"{}\"".format(new_fn + ".lock"))

        sleep(clargs.sleep)



