# -*- coding: utf-8 -*-
# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

""" Thin wrapper around the tensorboardX library.
It might be replaced with raw tensorboardX in the future."""

from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorboardX


class TensorBoardLogger:
    """Logging in tensorboard without tensorflow ops."""

    writer: tensorboardX.SummaryWriter
    log_dir: str
    always_flush: bool

    def __init__(self, log_dir: str, always_flush: bool = False) -> None:
        """Create a summary writer logging to log_dir."""
        self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        self.always_flush = always_flush

    def flush(self) -> None:
        self.writer.file_writer.flush()

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar variable.

        Parameters
        ----------
        tag
            Name of the scalar
        value
            Scalar value to be logged
        step
            training iteration
        """
        self.writer.add_scalar(tag, value, step)
        if self.always_flush:
            self.flush()

    def log_image(
            self,
            tag: str,
            image: np.ndarray,
            step: int,
            cmap=None,
            num_classes=None,
            colorbar=True,
    ) -> None:
        """Logs a an image to tensorboard.

        For gray-scale images, use ``cmap='gray'``.
        For label matrices (segmentation targets or class predictions),
        specify the global number of possible classes in ``num_classes``."""

        # Determine colormap and set discrete color values if needed.
        vmin, vmax = None, None
        ticks = None
        if cmap is None and num_classes is not None:
            # Assume label matrix with qualitative classes, no meaningful order
            # Using rainbow because IMHO all actually qualitative colormaps
            #  are incredibly ugly.
            cmap = plt.cm.get_cmap('viridis', num_classes)
            ticks = np.arange(num_classes)

        if num_classes is not None:  # For label matrices
            # Prevent colormap normalization. If vmax is not set, the colormap
            #  is dynamically rescaled to fit between the minimum and maximum
            #  values of the image to be plotted. This could lead to misleading
            #  visualizations if the maximum value of the array to be plotted
            #  is less than the global maximum of classes.
            vmin = 0
            vmax = num_classes

        fig, ax = plt.subplots()
        aximg = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        if colorbar:
            fig.colorbar(aximg, ticks=ticks)  # TODO: Centered tick labels

        # Create and write Summary
        self.writer.add_figure(tag, fig, step)
        if self.always_flush:
            self.flush()

    def log_histogram(
            self,
            tag: str,
            values: Union[Sequence[float], np.ndarray],
            step: int,
    ) -> None:
        """(Not yet tested!) Logs the histogram of a list/vector of values."""
        # Convert to a numpy array

        # Create and write Summary
        summary = self.writer.add_histogram(tag, values, step)
        if self.always_flush:
            self.flush()
