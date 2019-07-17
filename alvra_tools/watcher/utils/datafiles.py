import json


def load_scan_info(fn):
    with open(fn) as f:
        data = json.load(f)
    return data["scan_files"]

def list_scan_info_data(fnames):
#    res = set()
#    for fn in fnames:
#        for entry in load_scan_info(fn):
#            res.update(entry)
#    return res
    return set().union(*(
        entry
        for fn in fnames
        for entry in load_scan_info(fn)
    ))


def missing(seq1, seq2):
    set1 = set(seq1)
    set2 = set(seq2)
    missing_in_1 = set2 - set1
    missing_in_2 = set1 - set2
    return missing_in_1, missing_in_2



