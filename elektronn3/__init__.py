__all__ = ["cuda_enabled", "floatX"]
import torch
import numpy as np
from .logger import logger_setup
import logging
logger = logging.getLogger('elektronn3log')

logger_setup()
cuda_enabled = torch.cuda.is_available()
logger.info("Cuda %s." % "available" if cuda_enabled else "unavailable")
floatX = np.float32
