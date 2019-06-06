#!/usr/bin/env python3

from time import time


class Clock(object):
    """
    Simple timing clock

    prec sets precision of returned time deltas

    tick() returns time delta since last tick
    tock() returns time delta since the start of the clock
    """

    def __init__(self, prec=2):
        self.prec = prec
        self.start = self.last = time()

    def tick(self):
        """Time delta since last tick"""
        now = time()
        delta = now - self.last
        self.last = now
        return self._fmt_delta(delta)

    def tock(self):
        """Time delta since the start of the clock"""
        now = time()
        delta = now - self.start
        return self._fmt_delta(delta)

    def _fmt_delta(self, delta):
        """Format time deltas using the precision given to the constructor"""
        if self.prec is not None:
            delta = round(delta, self.prec)
        return delta





if __name__ == "__main__":
    from time import sleep

    c = Clock()
    for i in range(5):
        sleep(float(i) / 10)
        print(i, c.tick(), c.tock())



