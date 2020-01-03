__all__ = ["floatX"]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import numpy as np
from elektronn3.logger import logger_setup
import logging
logger = logging.getLogger('elektronn3log')

logger_setup()

floatX = np.float32  # TODO: Either hardcode float32 everywhere or add float16 support


def select_mpl_backend(mpl_backend):
    """ Set up matplotlib to use the specified backend.

    This needs to be run BEFORE the first import of matplotlib.pyplot!
    """
    import os
    from subprocess import check_call, CalledProcessError
    import matplotlib
    if mpl_backend.lower() == 'agg':
        matplotlib.use('AGG')
        logger.info('Using the AGG backend for matplotlib. No support for X11 windows.')
    else:
        if mpl_backend.startswith('force-'):
            matplotlib.use(mpl_backend.partition('force-')[-1])
        else:
            # Prevent setting of mpl qt-backend on machines without X-server before other modules import mpl.
            with open(os.devnull, 'w') as devnull:
                try:
                    # "xset q" will always succeed to run if an X server is currently running
                    check_call(['xset', 'q'], stdout=devnull, stderr=devnull)
                    if mpl_backend.lower() == 'auto':
                        pass  # Backend is silently set to system default.
                    else:
                        matplotlib.use(mpl_backend)
                    print('Using the {} backend for matplotlib.'.format(matplotlib.get_backend()))
                    # Don't set backend explicitly, use system default...
                # if "xset q" fails, conclude that X is not running
                except (OSError, CalledProcessError):
                    print('No X11 server found. Falling back to AGG backend for matplotlib.')
                    matplotlib.use('AGG')
