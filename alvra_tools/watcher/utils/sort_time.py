import os


def sorted_atime(fnames, *args, **kwargs):
    """Sort filenames by last access time"""
    return sorted(fnames, key=os.path.getatime, *args, **kwargs)

def sorted_ctime(fnames):
    """Sort filenames by metadata change time"""
    return sorted(fnames, key=os.path.getctime)

def sorted_mtime(fnames, *args, **kwargs):
    """Sort filenames by last modification time"""
    return sorted(fnames, key=os.path.getmtime, *args, **kwargs)

sorted_time = sorted_mtime



