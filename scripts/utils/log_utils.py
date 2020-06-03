# -*- coding:utf-8 -*-
import os
import time
import ctypes
import logging

from .config_utils import config
from .helper_utils import check_dir

FOREGROUND_WHITE = 0x0007
FOREGROUND_BLUE = 0x01  # text color contains blue.
FOREGROUND_GREEN = 0x02  # text color contains green.
FOREGROUND_RED = 0x04  # text color contains red.
FOREGROUND_YELLOW = FOREGROUND_RED | FOREGROUND_GREEN

STD_OUTPUT_HANDLE = -11
try:
    std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
except Exception as e:
    print("Exception:", e)
    std_out_handle = None

def set_color(color, handle=std_out_handle):
    try:
        bool_result = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
    except Exception as e:
        print("Exception:", e)
        bool_result = None

    return bool_result

def strftime(t=None):
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t or time.time()))


class Logger:
    def __init__(self, path, c_level=logging.INFO, f_level=logging.INFO):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)

        fmt = logging.Formatter("[ %(asctime)s][%(module)s.%(funcName)s][%(levelname)s] %(message)s")

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(c_level)

        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        fh.setLevel(f_level)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self, message, color=FOREGROUND_GREEN):
        set_color(color)
        self.logger.debug(message)
        set_color(FOREGROUND_WHITE)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message, color=FOREGROUND_YELLOW):
        set_color(color)
        self.logger.warning(message)
        set_color(FOREGROUND_WHITE)

    def error(self, message, color=FOREGROUND_RED):
        set_color(color)
        self.logger.error(message)
        set_color(FOREGROUND_WHITE)

    def critical(self, message):
        self.logger.critical(message)


# logger = Logger(os.path.join(config.log_root, strftime() + ".log"), logging.INFO, logging.INFO)
check_dir(config.log_root)
logger = Logger(os.path.join(config.log_root, strftime() + ".log"), logging.DEBUG, logging.DEBUG)

#################
# Logging
#################


# logging.basicConfig(format="[ %(asctime)s][%(module)s.%(funcName)s] %(message)s")
#
# DEFAULT_LEVEL = logging.INFO
# DEFAULT_LOGGING_DIR = config.log_root
# fh = None
#
# def init_fh():
#     global fh
#     if fh is not None:
#         return
#     if DEFAULT_LOGGING_DIR is None:
#         return
#     if not os.path.exists(DEFAULT_LOGGING_DIR): os.makedirs(DEFAULT_LOGGING_DIR)
#     logging_path = os.path.join(DEFAULT_LOGGING_DIR, strftime() + ".log")
#     fh = logging.FileHandler(logging_path)
#     fh.setFormatter(logging.Formatter("[ %(asctime)s][%(module)s.%(funcName)s] %(message)s"))
#
# def update_default_level(defalut_level):
#     global DEFAULT_LEVEL
#     DEFAULT_LEVEL = defalut_level
#
# def update_default_logging_dir(default_logging_dir):
#     global DEFAULT_LOGGING_DIR
#     DEFAULT_LOGGING_DIR = default_logging_dir
#
# def get_logger(name="FS", level=None):
#     level = level or DEFAULT_LEVEL
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     init_fh()
#     if fh is not None:
#         logger.addHandler(fh)
#     return logger
#
# logger = get_logger()
