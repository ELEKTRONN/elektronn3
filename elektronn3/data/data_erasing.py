# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert, Marius Killinger

import numpy as np
import time
import h5py as h5
from scipy import ndimage
import os
from elektronn3.data.blob_generator import BlobGenerator


class IncorrectLimits(Exception):
    pass


class IncorrectThreshold(Exception):
    pass


class IncorrectType(Exception):
    pass


class FunctionCallsCounter():
    counter = 0


class ScheduledVariable(object):
    """ The class is responsible for variable scheduling along an iteration
    process according to the exponential law. The user specifies
    the initial value, the target one and the number of steps
    along which the variable will be gradually scaled.

    At each iteration the user has explicitly call update_variable to
    update and modify the variable

    The class overrides all comparison operators which make it easier
    to compare the current scheduled variable value with a float
    number

    If the user doesn't specify the target value or the interval the variable
    works as a constant

    """
    def __init__(self, value, max_value=None, interval=None, steps_per_report=None):
        """
        The constructor initializes all necessary variables and checks that
        the initial value is less than the target one
        Parameters
        ----------
        value - float
        max_value - float
        interval - int
        steps_per_report - int
        """
        if max_value and (value > max_value):
            raise IncorrectLimits("ERROR: threshold limits are wrong: "
                                  "initial_threshold_value = %f, "
                                  "max_threshold_value = %f" %
                                  (value, max_value))

        self.value = value
        self.max_value = max_value
        self.interval = interval

        if max_value and interval:
            self.base = np.power((self.max_value / self.value), 1.0 / self.interval)

        self.steps_per_report = steps_per_report
        self.counter = 0

    def update_value(self):
        """
        The function preforms update of the scheduled variable
        according to the exponential law
        Returns
        -------
        self.value - float
        """

        if not self.max_value or not self.interval:
            return self.value

        self.value *= self.base

        if self.value > self.max_value:
            self.value = self.max_value

        if self.steps_per_report:
            if (self.counter % self.steps_per_report) == 0:
                print("ScheduledVariable: value: %f; counter: %d"
                      % (self.value, self.counter))
            self.counter += 1

        return self.value

    def get_value(self):
        """
        The function returns the current value of the scheduled
        variable
        Returns
        -------
        self.value - float
        """
        return self.value

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other


def check_data_erasing_config(batch,
                              probability,
                              threshold,
                              lim_blob_depth,
                              lim_blob_width,
                              lim_blob_height,
                              lim_gauss_diffusion,
                              verbose=False,
                              save_path=None,
                              num_steps_save=None):
    """ The function checks data erasing parameters and ensures the user
    that all parameters won't cause problem during apply_data_erasing
    function calls. The function throws exceptions if a conflict is
    detected. Use this function before training procedure to be sure the config
    fulfills the requirements posed by the apply_data_erasing function

    batch - 4d float numpy array
        raw data batch with the format: [num_channels, depth, width, height]
    probability - float
        probability of applying the data erasing algorithm
    threshold - ScheduledVariable
        The variable controls the level of data erasing with respect to
        raw data batch volume
    lim_blob_depth - 2d int numpy array
        min and max values of the blob depth
    lim_blob_width - 2d int numpy array
        min and max values of the blob width
    lim_blob_height - 2d int numpy array
        min and max values of the blob height
    lim_gauss_diffusion - 2d int numpy array
        min and max values of gaussian blurring level
    verbose - boolean
        mode that controls text information on the screen
    save_path - str
        path to the files that will contain modified
        raw data batch in the "h5" format
    num_steps_save - int
        number of steps between writing modified
        raw data batch in the "h5" format

    Returns
    -------
    None
    """

    channels, batch_depth, batch_width, batch_height = batch.shape

    # Check the user's specified blob size.
    # First entry of each list must be less than the second one
    if lim_blob_depth[0] >= lim_blob_depth[1]:
        raise IncorrectLimits("ERROR: blob depth limits are inconsistent: "
                              "min depth = %i, max depth = %i" %
                              (lim_blob_depth[0],
                               lim_blob_depth[1]))

    if lim_blob_width[0] >= lim_blob_width[1]:
        raise IncorrectLimits("ERROR: blob width limits are inconsistent: "
                              "min width = %i, max width = %i" %
                              (lim_blob_width[0],
                               lim_blob_width[1]))

    if lim_blob_height[0] >= lim_blob_height[1]:
        raise IncorrectLimits("ERROR: blob height limits are inconsistent: "
                              "min height = %i, max height = %i" %
                              (lim_blob_height[0],
                               lim_blob_height[1]))

    # Сheck whether the blob size exceeds the domain (batch) size
    # If so, raise the corresponfing exception
    if lim_blob_depth[1] >= batch_depth:
        raise IncorrectLimits("ERROR: blob depth exceeds domain depth: "
                              "blob depth = %i, domain depth = %i" %
                              (lim_blob_depth[1],
                               batch_depth))

    elif lim_blob_height[1] >= batch_width:
        raise IncorrectLimits("ERROR: blob width exceeds domain width: "
                              "blob width = %i, domain width = %i" %
                              (lim_blob_width[1],
                               batch_width))

    elif lim_blob_height[1] >= batch_height:
        raise IncorrectLimits("ERROR: blob height exceeds domain height: "
                              "blob height = %i, domain height = %i" %
                              (lim_blob_height[1],
                               batch_height))

    # Check the data type of  the threshold variable
    # The threshold must have its type of ScheduledVariable
    if not isinstance(threshold, ScheduledVariable):
        raise IncorrectType("ERROR: threshold variable is not type of ScheduledVariable")

    # Check whether the theshold value specified by the user
    # is within the range (0.0, 1.0). If it's not, raise the
    # corresponding exception
    if threshold < 0.0 or threshold > 1.0:
        raise IncorrectLimits("ERROR: threshold of data erasing is out "
                              "of the limits [0.0,1.0]: threshold = %f" %
                              (threshold))

    # Check the user's specified level of Gaussing blurring.
    # First entry of the list must be less than the second one
    if lim_gauss_diffusion[0] >= lim_gauss_diffusion[1]:
        raise IncorrectLimits("ERROR: blob diffusion limits are inconsistent: "
                              "min diff = %i, max diff = %i" %
                              (lim_gauss_diffusion[0],
                               lim_gauss_diffusion[1]))

    # Check whether the probability value specified by the user
    # is within the range (0.0, 1.0). If it's not, raise the
    # corresponding exception
    if probability < 0.0 or probability > 1.0:
        raise IncorrectLimits("ERROR: probability of data erasing is out "
                              "of the limits [0.0,1.0]: probability = %f" %
                              (probability))

    # Check whether the directory specified by the user exists.
    # If no, try to create the dirrectory using the path
    # passed by the user to the function
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)


def apply_data_erasing(batch,
                       probability,
                       threshold,
                       lim_blob_depth,
                       lim_blob_width,
                       lim_blob_height,
                       lim_gauss_diffusion,
                       verbose=False,
                       save_path=None,
                       num_steps_save=None):
    """ The function takes a batch and applies data erasing.
    At the beginning the function generates a random number within
    the range [0,1) and compares it with the probability value passed
    by the user. If the random number exceeds the probability value
    the function terminates and returns the controls to the caller.
    Otherwise, the function generates random blobs within a raw data batch
    volume until the total accumulated blob volume exceeds that value
    specified by means of the threshold variable. The threshold denotes
    the percentage of raw data batch volume that has to be filled in
    by blobs. Blobs have different spatial size which is randomly
    generated within the ranges specified by the user.
    Moreover, the volume within a blob is blurred by the Gaussian filter.
    The user is responsible to pass range of the Gaussian
    blurring filter levels

    Parameters
    ----------
    batch - 4d float numpy array
        raw data batch with the format: [num_channels, depth, width, height]
    probability - float
        probability of applying the data erasing algorithm
    threshold - ScheduledVariable
        The variable controls the level of data erasing with respect to
        raw data batch volume
    lim_blob_depth - 2d int numpy array
        min and max values of the blob depth
    lim_blob_width - 2d int numpy array
        min and max values of the blob width
    lim_blob_height - 2d int numpy array
        min and max values of the blob height
    lim_gauss_diffusion - 2d int numpy array
        min and max values of gaussian blurring level
    verbose - boolean
        mode that controls text information on the screen
    save_path - str
        path to the files that will contain modified
        raw data batch in the "h5" format
    num_steps_save - int
        number of steps between writing modified
        raw data batch in the "h5" format

    Returns
    -------
    None
    """

    if np.random.rand() > probability:
        return

    channels, batch_depth, batch_width, batch_height = batch.shape
    batch_volume = batch_depth * batch_width * batch_height

    generator = BlobGenerator(domain=[batch_depth, batch_width, batch_height],
                              lim_depth=lim_blob_depth,
                              lim_width=lim_blob_width,
                              lim_height=lim_blob_height)

    start_time = time.time()
    threshold.update_value()
    for batch_indx in range(channels):
        erasing_percentage = 0.0
        intersection = set()

        while erasing_percentage < threshold:

            blob = generator.create_blob()

            for k in range(blob.z_min, blob.z_max + 1):
                for i in range(blob.x_min, blob.x_max + 1):
                    for j in range(blob.y_min, blob.y_max + 1):
                        intersection.add((k, i, j))

            snippet = batch[batch_indx,
                            blob.z_min:blob.z_max + 1,
                            blob.x_min:blob.x_max + 1,
                            blob.y_min:blob.y_max + 1]

            diffuseness = np.random.randint(low=lim_gauss_diffusion[0],
                                            high=lim_gauss_diffusion[1],
                                            dtype=np.int16)

            snippet = ndimage.gaussian_filter(snippet, diffuseness)

            batch[batch_indx,
                  blob.z_min:blob.z_max + 1,
                  blob.x_min:blob.x_max + 1,
                  blob.y_min:blob.y_max + 1] = snippet

            erased_volume = len(intersection)
            erasing_percentage = erased_volume / batch_volume

        if verbose:
            print("erased percentage for channel (%i): %f" %
                  (batch_indx, erasing_percentage))

        if save_path and num_steps_save:
            if (FunctionCallsCounter.counter % num_steps_save) == 0:
                file = h5.File('%sraw_data_batch(channel-%i).h5'
                               % (save_path, batch_indx), 'w')
                file.create_dataset("raw_data",
                                    data=batch[batch_indx, :, :, :],
                                    dtype=np.float32)
                file.close()

    end_time = time.time() - start_time

    if verbose:
        print("spent cpu time for data erasing: %f, s" % end_time)

    FunctionCallsCounter.counter += 1