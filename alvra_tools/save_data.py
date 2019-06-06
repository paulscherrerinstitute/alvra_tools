import h5py


def save_JF_data_cropped(fn, roi1, roi2, pulse_ids):
    with h5py.File(fn, "w") as f:
        grp = f.create_group("JF02T09V01_crop")
        grp.create_dataset("roi1",      data=roi1)
        grp.create_dataset("roi2",      data=roi2)
        grp.create_dataset("pulse_ids", data=pulse_ids)


def save(fn, **kwargs):
    with h5py.File(fn, "w") as f:
        grp = f.create_group("JF02T09V01_crop")
        grp.create_dataset("roi1",      data=roi1)
        grp.create_dataset("roi2",      data=roi2)
        grp.create_dataset("pulse_ids", data=pulse_ids)

        for key, value in kwargs.items():
            grp.create_dataset(key, data=value)




