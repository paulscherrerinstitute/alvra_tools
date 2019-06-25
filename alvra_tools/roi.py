#!/usr/bin/env python3

import numpy as np


class ROI(tuple):
    """
    Region of Interest (ROI)
    Holds four values (xmin, xmax, ymin, ymax),
    which can be applied to an array to select a ROI via apply()
    """

    def __new__(self, *args):
        data = np.array(args).ravel()
        if data.size != 4:
            raise ROIValueError("too many values in argument list (expected 4)")
        xmin, xmax, ymin, ymax = data
        x = slice(xmin, xmax)
        y = slice(ymin, ymax)

        self.xmin, self.xmax, self.ymin, self.ymax = self.data = data
        self.x, self.y = x, y
        return super().__new__(ROI, (x, y))

    def apply(self, arr):
        """Apply to an array"""
        return arr[..., self.x, self.y]

    def save(self, fname):
        """Store as numpy file"""
        if fname.endswith(".npy"):
            return np.save(fname, self.data)
        else:
            return np.savetxt(fname, self.data, fmt="%i")

    @classmethod
    def load(cls, fname):
        """Load data from numpy file and construct ROI"""
        if fname.endswith(".npy"):
            arr = np.load(fname)
        else:
            arr = np.loadtxt(fname, dtype=int)
        return cls(arr)

    def __str__(self):
        return "[({}, {}), ({}, {})]".format(*self.data)


class ROIValueError(Exception):
    pass





if __name__ == "__main__":
    print("Run some tests...")

    roi1 = ROI([[10, 20], [30, 40]])
    roi2 = ROI([10, 20, 30, 40])
    roi3 = ROI(10, 20, 30, 40)

    assert roi1 == roi2
    assert roi1 == roi3

    assert roi1.x == slice(10, 20)
    assert roi1.y == slice(30, 40)

    assert roi1[0] == slice(10, 20)
    assert roi1[1] == slice(30, 40)


    shape = (100, 200)
    length = np.product(np.array(shape))
    data = np.arange(length).reshape(shape)

    roi = ROI(10, 13, 30, 33)
    new = roi.apply(data)
    assert np.all(new == [[2030, 2031, 2032], [2230, 2231, 2232], [2430, 2431, 2432]])


    roi1 = ROI(10, 20, 30, 40)
    fn = "test.npy"
    roi1.save(fn)
    roi2 = ROI.load(fn)
    assert roi1 == roi2


    roi = (1, 2, 3, 4, 5)
    try:
        roi = ROI(roi)
    except ROIValueError as e:
#        print("got {}:".format(type(e).__name__), "\"{}\"".format(e), "for the input:", roi)
        pass
    else:
        raise AssertionError

    print("Done.")



