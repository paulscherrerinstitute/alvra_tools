from utils import DataFileName

def convert_input_output(source_fname, target_folder):
    try:
        fn = DataFileName(source_fname)
    except:
        return None
    if not fn.type.startswith("JF02"):
        return None
    fn.type += "crop"
    fn.folder = target_folder
    return fn.fname


def convert_JF_BS(source_fname):
    try:
        fn = DataFileName(source_fname)
    except:
        return None
    if not fn.type.startswith("JF02"):
        return None
    fn.type = "BSREAD"
    return fn.fname



