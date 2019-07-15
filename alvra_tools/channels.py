import os
import logging
from .cfgfile import ConfigFile


DEFAULT_FNAME = "channels.ini"


#logging.basicConfig(level=logging.NOTSET)
log = logging.getLogger()


def update_channels(fname=DEFAULT_FNAME):
    cfg = ConfigFile(fname)
    globals().update(cfg)
    log.debug(f"Loaded channels from {fname}")
    return cfg



try:
    config = update_channels()
except FileNotFoundError:
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, DEFAULT_FNAME)
    config = update_channels(fname)
    log.warning(f"Fallback: loaded default channel list ({fname})")



#TODO handle channels that are deleted from config but are still in globals()
