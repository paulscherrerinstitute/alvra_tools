#!/usr/bin/env python3

# Command line argument handling
import argparse

description = "Show files listed in json (scan_info), but not existing as data (scan_data) as well as existing data missing in json."
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description)

parser.add_argument("-p", "--pgroup", help="pgroup", default="p17589")
parser.add_argument("-a", "--alphabetic-sort", help="Sort existing files alphabetically instead of by modification time", action="store_true")

clargs = parser.parse_args()


# Folders & filenames
pgroup = clargs.pgroup

dir_base = "/sf/alvra/data/"
dir_info = dir_base + pgroup + "/res/scan_info"
dir_data = dir_base + pgroup + "/raw/scan_data"

fnames_info = dir_info + "/*_scan_info.json"
fnames_data = dir_data + "/*/*.h5"



################################################################################
from glob import glob
from utils import list_scan_info_data, missing
from utils import sorted_time, print_red, print_lines


if clargs.alphabetic_sort:
    file_sorter = sorted
else:
    file_sorter = sorted_time


fnames_info = glob(fnames_info)
fnames_data = glob(fnames_data)

fnames_in_info = list_scan_info_data(fnames_info)

missing_data, missing_in_info = missing(fnames_data, fnames_in_info)

missing_data = sorted(missing_data)
missing_in_info = file_sorter(missing_in_info)


print_red("listed in json, but file missing:")
print_lines(missing_data)

print_red("file exists, but missing in json:")
print_lines(missing_in_info)



