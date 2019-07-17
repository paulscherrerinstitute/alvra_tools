import h5py
import os


def save_JF_data_cropped(fn, img_roi1, img_roi2, pulse_ids, roi1=None, roi2=None, det_name="JF02T09V02"):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with h5py.File(fn, "w") as f:
        grp = f.create_group(det_name + "_crop")
        grp.create_dataset("pulse_ids", data=pulse_ids)
        ds_roi1 = grp.create_dataset("roi1", data=img_roi1)
        ds_roi2 = grp.create_dataset("roi2", data=img_roi2)
        if roi1 is not None:
            grp.create_dataset("roi1_coords", data=roi1)
        if roi2 is not None:
            grp.create_dataset("roi2_coords", data=roi2)


def save(fn, **kwargs):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with h5py.File(fn, "w") as f:
        for key, value in kwargs.items():
            f.create_dataset(key, data=value)



