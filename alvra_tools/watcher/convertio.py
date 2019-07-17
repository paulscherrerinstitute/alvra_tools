from utils import DataFileName

def convert_input_output(source_fname, source_folder, target_folder):
    if not source_fname.startswith(source_folder):
        return None
    try:
        fn = DataFileName(source_fname)
    except:
        return None
    if not fn.type.startswith("JF02"):
        return None
    fn.type += "crop"

    folder = fn.folder
    folder = folder[len(source_folder):]
    fn.folder = target_folder + folder

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



