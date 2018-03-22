# -*- coding: utf-8 -*-
# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert, Marius Killinger

import logging
import os
import getpass
import sys

import colorlog


def logger_setup():
    # Formats for colorlog.LevelFormatter
    log_level_formats = {'DEBUG': '%(log_color)s%(msg)s (%(module)s:%(lineno)d)',
                         'INFO': '%(log_color)s%(msg)s',
                         'WARNING': '%(log_color)sWARNING: %(msg)s (%(module)s:%(lineno)d)',
                         'ERROR': '%(log_color)sERROR: %(msg)s (%(module)s:%(lineno)d)',
                         'CRITICAL': '%(log_color)sCRITICAL: %(msg)s (%(module)s:%(lineno)d)',}

    log_colors = {'DEBUG': 'blue', 'INFO': 'cyan', 'WARNING': 'bold_yellow',
                  'ERROR': 'red', 'CRITICAL': 'red,bg_white'}

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

        lstream_handler = colorlog.StreamHandler(sys.stdout)
        lstream_handler.setFormatter(
            colorlog.LevelFormatter(fmt=log_level_formats,
                                    log_colors=log_colors))
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
