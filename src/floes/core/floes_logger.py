"""
floes_logger.py - logging functionality for the floes package.

@author Dean Biskup
@email <dbiskup2@illinois.edu>
@org University of Illinois, Urbana-Champaign Audio Group
"""

import logging


class FloesLogger(object):
    """
    The `FloesLogger` class represents the common configuration of the logger
    used in `floes`. By default, this class encapsulates a Python `logging`
    logger with the name provided by `name`. Exposes a single function:
    `write`, which writes the given message to `stderr`.

    Args:
        name: str
            Name of the logger to create.
        level: int, default `logging.INFO`
            One of `logging.INFO`, `logging.WARNING`, etc.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        self.level = level
        self.name = name

        # set up the internal logger object
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        hdlr = logging.StreamHandler()
        hdlr.setLevel(level)
        fmtr = logging.Formatter("%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s")
        hdlr.setFormatter(fmtr)
        self.logger.addHandler(hdlr)

    def write(self, message: str, level: int = None):
        """
        Writes a message at the given level to the log.

        Args:
            message: str
                Message to write to the log.
            level: int, default `None`
                The level of the message to write. If not provided, or `None`,
                will default to the level of this `FloesLogger` object.
        """
        if level is None:
            level = self.level
        self.logger.log(level, message)


# these statements will create a reference to the same logger for everywhere
# that imports this file to the module namespace, e.g. floes_logger.logger
logger = FloesLogger('floes')
