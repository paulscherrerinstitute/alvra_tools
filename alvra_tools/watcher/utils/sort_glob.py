import os
from glob import glob

def glob_sorted(path, *args, **kwargs):
    res = glob(path)
    res = sorted(res, *args, **kwargs)
    return res

def glob_sorted_time(path, *args, **kwargs):
    kwargs_init = {"key": os.path.getmtime}
    kwargs_init.update(kwargs)
    return glob_sorted(path, *args, **kwargs_init)

def rglob(*args, **kwargs):
    kwargs_init = {"recursive": True}
    kwargs_init.update(kwargs)
    return glob(*args, **kwargs_init)



