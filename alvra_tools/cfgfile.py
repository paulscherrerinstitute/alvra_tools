#!/usr/bin/env python3

from configparser import DuplicateSectionError, MissingSectionHeaderError, ConfigParser
import os

MODES = list("rwai")


from ast import literal_eval


import re

RE_CLEAN = re.compile(r'\W|^(?=\d)') # non-word chars anywhere or digits at start of string

def clean_var_name(s):
    return re.sub(RE_CLEAN, '_', s)


def forgiving_eval(value):
    try:
        return literal_eval(value)
    except SyntaxError:
        return value



class ConfigFile(object):
    """
    Simplifying wrapper for ConfigParser
    Maps a section in a config file to an object with attributes
    """

    def __init__(self, filename=None, section='GENERAL', mode='i'):
        """
        mode can be:
            'r' (read-only), 'w' ((over)write), 'a' (append), 'i' (insert)
        """
        self._filename = filename
        self._section  = section
        self._mode     = mode

        if mode not in MODES:
            raise ValueError("mode string must be one of '{}' or '{}', not '{}'".format(", ".join(MODES[:-1]), MODES[-1], mode))

        self._parser = ConfigParser()
        self._parser.optionxform = str
        if filename:
            self.read()


    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def keys(self):
        return self._dict.keys()


    def read(self, filename=None, section=None, replace=True):
        filename = filename or self._filename
        section  = section  or self._section

        if self._mode == 'r' and not os.path.exists(filename):
            os.listdir(filename) # raise the correct exception

        try:
            self._parser.read(filename)
        except MissingSectionHeaderError:
            with open(filename) as f:
                content = "[{}]\n".format(section) + f.read()
                self._parser.read_string(content)

        if self._parser.sections():
            items = self._parser.items(section)
            for name, value in items:
                name = clean_var_name(name)
                value = forgiving_eval(value)
                setattr(self, name, value)

        if replace:
            self._filename = filename
            self._section  = section


    @property
    def _dict(self):
        return {name: value for name, value in sorted(self.__dict__.items())
                            if not name.startswith("_")}


    def write(self, filename=None, section=None, replace=False):
        filename = filename or self._filename
        section  = section  or self._section

        mode = self._mode
        if mode == 'r':
            return

        outcfg = ConfigFile()
        if mode == 'i':
            mode = 'w'
            outcfg.read(filename, section)

        outcfg = outcfg._parser

        try:
            outcfg.add_section(section)
        except DuplicateSectionError:
            pass

        items = self._dict.items()
        for name, value in items:
            value = str(value)
            outcfg.set(section, name, value)

        with open(filename, mode) as outfile:
            outcfg.write(outfile)

        if replace:
            self._parser   = outcfg
            self._filename = filename
            self._section  = section


    def __repr__(self):
        arg_strings = []
        for name, value in self._dict.items():
            arg_strings.append("{}={}".format(name, value))

        type_name = type(self).__name__
        return "{}({})".format(type_name, ', '.join(arg_strings))


    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.write()


    def print_table(self):
        length = max(len(i) for i in self)
        for i in self:
            print(i.ljust(length), cfg[i])




if __name__ == "__main__":
    cfg = ConfigFile("channels.ini")
    cfg.print_table()
    cfg.write("channels-new.ini")

    cfg = ConfigFile("channels-new.ini")
    cfg.write()





