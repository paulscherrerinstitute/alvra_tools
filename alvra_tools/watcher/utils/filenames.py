#!/usr/bin/env python

import os


class DataFileName(object):
    def __init__(self, fn):
        parsed = fname_parse(fn)
        self.__dict__.update(parsed)

    def __str__(self):
        return self.fname

    @property
    def fname(self):
        return fname_assemble(self.__dict__)

    @property
    def atime(self):
        """last access time"""
        return os.path.getatime(self.fname)

    @property
    def ctime(self):
        """metadata change time"""
        return os.path.getctime(self.fname)

    @property
    def mtime(self):
        """last modification time"""
        return os.path.getmtime(self.fname)



def fname_parse(fn):
    splitted = fn.split("/")
    folder = "/".join(splitted[:-1])
    remainder = splitted[-1]

    splitted = remainder.split(".")
    remainder, typ, ext = splitted

    splitted = remainder.split("_")
    last = splitted[-1]
    if last.startswith("step"):
        step = last[4:]
        step = int(step)
        scanname = "_".join(splitted[:-1])
    else:
        step = None
        scanname = remainder

#    if folder.endswith("/" + scanname):
#        folder = folder[:-len(scanname)-1]

    res = {
        "folder": folder,
        "type": typ,
        "extension": ext,
        "scanname": scanname,
        "step": step
    }

    return res


def fname_assemble(infos):
    folder = infos["folder"] + "/" #+ infos["scanname"] + "/"
    if infos["step"] is not None:
        step = "step{:04}".format(infos["step"])
        fname = infos["scanname"] + "_" + step + "." + infos["type"] + "." + infos["extension"]
    else:
        fname = infos["scanname"] + "." + infos["type"] + "." + infos["extension"]
    return folder + fname





if __name__ == "__main__":
    from printing import print_dict

    fn = "/sf/alvra/data/p17589/raw/scan_data/monoscan_20uJ_144_Febpy_a/monoscan_20uJ_144_Febpy_a_step0028.JF02T09V01.h5"
    tmp = fname_parse(fn)
    print_dict(tmp)

    nfn = fname_assemble(tmp)

    output = "/output"
    length_folder = len(tmp["folder"])
    outputs = output * (length_folder / len(output) + 1)
    outputs = outputs[:length_folder]

    tmp["folder"] = outputs
    tmp["type"] += "crop"
    ofn = fname_assemble(tmp)

    print(fn)
    print(nfn)
    print(ofn)


    print()
    dfn = DataFileName(fn)
    print(dfn)
    dfn.folder = outputs
    dfn.type += "crop"
    print(dfn)



