# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import logging
import os
import getpass
import sys

try:
    import colorlog
    colorize = True
except ImportError:
    colorize = False
    print('Please (pip) install colorlog.')


def logger_setup():
    # Formats for colorlog.LevelFormatter
    log_level_formats = {'DEBUG': '%(log_color)s%(msg)s (%(module)s:%(lineno)d)',
                         'INFO': '%(log_color)s%(msg)s',
                         'WARNING': '%(log_color)sWARNING: %(msg)s (%(module)s:%(lineno)d)',
                         'ERROR': '%(log_color)sERROR: %(msg)s (%(module)s:%(lineno)d)',
                         'CRITICAL': '%(log_color)sCRITICAL: %(msg)s (%(module)s:%(lineno)d)',}

    log_colors = {'DEBUG': 'blue', 'INFO': 'cyan', 'WARNING': 'bold_yellow',
                  'ERROR': 'red', 'CRITICAL': 'red,bg_white'}

    log_colors_inspection = {'DEBUG': 'purple', 'INFO': 'purple',
                             'WARNING': 'yellow', 'ERROR': 'red',
                             'CRITICAL': 'red,bg_white'}

    # Initialize logger that can be used
    user_name = getpass.getuser()
    logger = logging.getLogger('elektronn3log')
    # Only set up the logger if it hasn't already been initialised before:
    if not len(logger.handlers) > 0:
        logger.setLevel(logging.DEBUG)

        lfile_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s]\t%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        lfile_path = os.path.abspath('/tmp/{}_elektronn3.log'.format(user_name))
        lfile_level = logging.DEBUG
        lfile_handler = logging.FileHandler(lfile_path)
        lfile_handler.setLevel(lfile_level)
        lfile_handler.setFormatter(lfile_formatter)
        logger.addHandler(lfile_handler)

        if colorize:
            lstream_handler = colorlog.StreamHandler(sys.stdout)
            lstream_handler.setFormatter(
                colorlog.LevelFormatter(fmt=log_level_formats,
                                        log_colors=log_colors))
        else:
            lstream_handler = logging.StreamHandler(sys.stdout)
        # set this to logging.DEBUG to enable output for logger.debug() calls
        lstream_level = logging.INFO
        lstream_handler.setLevel(lstream_level)
        logger.addHandler(lstream_handler)

        logger.propagate = False

        if False:  # Test log levels:
            logger.critical('== critical')
            logger.error('== error')
            logger.warning('== warn')
            logger.info('== info')
            logger.debug('== debug')

    inspection_logger = logging.getLogger('elektronn3log-inspection')
    # Only set up the logger if it hasn't already been initialised before:
    if not len(inspection_logger.handlers) > 0:
        inspection_logger.setLevel(logging.DEBUG)
        lfile_formatter = logging.Formatter('%(message)s')
        lfile_path = os.path.abspath(
            # os.path.expanduser('~/elektronn2-inspection.log')
            '/tmp/{}_elektronn3-inspection.log'.format(user_name)
        )
        lfile_level = logging.DEBUG
        lfile_handler = logging.FileHandler(lfile_path)
        lfile_handler.setLevel(lfile_level)
        lfile_handler.setFormatter(lfile_formatter)
        inspection_logger.addHandler(lfile_handler)

        # if config.inspection:
        #     if colorize:
        #         lstream_handler = colorlog.StreamHandler(sys.stdout)
        #         lstream_handler.setFormatter(
        #             colorlog.LevelFormatter(fmt=log_level_formats,
        #                                     log_colors=log_colors_inspection))
        #     else:
        #         lstream_handler = logging.StreamHandler(sys.stdout)
        #     # set this to logging.DEBUG to enable output for inspection_logger.debug() calls
        #     lstream_level = logging.INFO
        #     lstream_handler.setLevel(lstream_level)
        #     inspection_logger.addHandler(
        #         lstream_handler)  # comment out to suppress printing

        inspection_logger.propagate = False
