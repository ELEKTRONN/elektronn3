# elektronn3-dev
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

from elektronn3 import logger
logger.warning('elektronn3.training.loss is deprecated. Please use elektronn3.modules.loss '
               'instead.')
# required for old models to function properly
from ..modules.loss import *