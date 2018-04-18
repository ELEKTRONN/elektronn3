# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert, Marius Killinger

import numpy as np
import h5py as h5
import random
from scipy import ndimage
import os
from elektronn3.data.region_generator import RegionGenerator
import logging


class IncorrectLimits(Exception):
    pass


class IncorrectThreshold(Exception):
    pass


class IncorrectValue(Exception):
    pass


class IncorrectType(Exception):
    pass


class FunctionCallsCounter():
    counter = 0


class ScheduledParameter(object):
    """ The class is responsible for a parameter scheduling along an iterative
    process according to either the linear or exponential growth. The user
    specifies the initial value, the target one, growth type and the number
    of steps along which the parameter has to be gradually scaled.

    At each iteration the user has to explicitly call step() to
    update and modify the parameter

    If the user doesn't specify the target value or the interval,
    the parameter works as a constant

    """

    logger = logging.getLogger('elektronn3log')

    def __init__(self,
                 value,
                 max_value=None,
                 growth_type=None,
                 interval=None,
                 steps_per_report=None):
        """
        Initializes all necessary variables and checks that
        the initial value is less than the target one and
        growth type is chosen correctly
        Parameters
        ----------
        value - float
            the parameter value at the beginning of an scheduled process
        max_value - float
            the parameter value at the end of an scheduled process
        growth_type - str
            type of growth: "lin" - linear; "exp" - exponential
        interval - int
            number of steps along which the parameter value has to be
            increased from the initial value to the maximal one
        steps_per_report - int
            number of step between information update on the screen
        """

        if max_value and (value > max_value):
            raise IncorrectLimits(f'ERROR: threshold limits are wrong: '
                                  f'initial_threshold_value = {value}, '
                                  f'max_threshold_value = {max_value}')

        self.value = value

        if max_value and interval:
            self.max_value = max_value
            self.interval = interval

            if growth_type == "lin":
                self.update_function = self.lin_update
                self.base = float(max_value - value) / self.interval
            elif growth_type == "exp":
                self.update_function = self.exp_update
                self.base = np.power((self.max_value / self.value), 1.0 / self.interval)
            else:
                raise IncorrectValue(f'ERROR: ScheduledParameter class can only '
                                     f'take \"growth_type\" parameter with values '
                                     f'either \"lin\" or \"exp\". Value \"{growth_type}\" '
                                     'has been passed instead')

        else:
            self.update_function = self.idle_update


        self.steps_per_report = steps_per_report
        self.counter = 0


    def step(self):
        """ Preforms an update of the scheduled parameter
        according to the growth type chosen by the user
        Returns
        -------
        self.value - float
        """

        self.update_function()
        self._print_report()

        return self.value


    def lin_update(self):
        """ Preforms an update of the scheduled parameter
        according to the linear growth
        Returns
        -------
        None
        """

        self.value += self.base
        self.check_max_limit()


    def exp_update(self):
        """ Preforms an update of the scheduled parameter
        according to the exponential growth
        Returns
        -------
        None
        """
        self.value *= self.base
        self.check_max_limit()


    def idle_update(self):
        """ Was designed to keep step() function uniform across different types
        of grows. The function is called if a class instance is used as
        a constant variable within an iterative process
        Returns
        -------
        None
        """
        pass


    def check_max_limit(self):
        """ Checks whether the current parameter value is less
        than the maximum value specified by the user. If the value exceeds
        it the parameter variable will be assigned to the maximum one.
        Returns
        -------
        None
        """
        if self.value > self.max_value:
            self.value = self.max_value


    def _print_report(self):
        """ Prints the current value of the parameter on the screen
        during an iterative process. The function counts number of
        step() calls and prints information each time when the number
        of the calls is even with respect to steps_per_report

        If the used doesn't pass the number of steps_per_report the function
        doesn't print the information
        Returns
        -------
        None
        """
        if self.steps_per_report:

            if (self.counter % self.steps_per_report) == 0:
                ScheduledParameter.logger.info(f'ScheduledVariable: '
                                               f'value: {self.value}, '
                                               f'counter: {self.counter}')

            self.counter += 1



def check_random_data_blurring_config(patch_shape,
                                      probability,
                                      threshold,
                                      lower_lim_region_size,
                                      upper_lim_region_size,
                                      verbose=False,
                                      save_path=None,
                                      num_steps_save=None):
    """ Checks random data blurring parameters and ensures the user
    that all parameters won't cause problems during apply_random_blurring
    function calls. The function throws exceptions if a conflict is
    detected. Use this function before training procedure to be sure the config
    fulfills the requirements posed by the apply_random_blurring function

    patch_shape - list of int
        the sizes of input sample along each axis:
        [depth, width, height]
    probability - float
        probability of applying the data random blurring algorithm
    threshold - ScheduledParameter
        The Parameter controls the level of data random blurring with respect to
        raw data input sample volume
    lower_lim_region_size - np.array of int
        min values of regions size along each axis
    upper_lim_region_size - np.array of int
        max values of regions size along each axis
    verbose - boolean
        mode that controls text information on the screen
    save_path - str
        path to the files that will contain modified
        raw data input sample in the "h5" format
    num_steps_save - int
        number of steps between writing modified
        raw data input sample in the "h5" format

    Returns
    -------
    None
    """

    # Check the user's specified dimensionality
    if (len(lower_lim_region_size) != len(upper_lim_region_size) or
        len(patch_shape) != len(lower_lim_region_size) or
        len(patch_shape) != len(upper_lim_region_size)):
        raise IncorrectLimits(f'ERROR: the region limits or/and input sample '
                              f'have different dimensionality:\n'
                              f'dimension of lower region limits: {len(lower_lim_region_size)}\n'
                              f'dimension of upper region limits: {len(upper_lim_region_size)}\n'
                              f'dimension of sample: {len(patch_shape)}')

    # Check the user's specified region size
    dim = len(patch_shape)
    for i in range(dim):
        if lower_lim_region_size[i] >= upper_lim_region_size[i]:
            raise IncorrectLimits(f'ERROR: region limits are inconsistent at axis={i}:\n'
                                  f'min = {lower_lim_region_size[i]}\n'
                                  f'max = {upper_lim_region_size[i]}\n')

    # Сheck whether the region size exceeds the input sample size
    for i in range(dim):
        if upper_lim_region_size[i] >= patch_shape[i]:
            raise IncorrectLimits(f'ERROR: region size exceeds input sample at axis={i}:\n'
                                  f'region size = {upper_lim_region_size[i]}\n'
                                  f'sample size = {patch_shape[i]}\n')


    # Check the data type of  the threshold parameter
    # The threshold must have its type of ScheduledParameter
    if not isinstance(threshold, ScheduledParameter):
        raise IncorrectType(f'ERROR: threshold type is not type of ScheduledParameter\n'
                            f'instead, it has its type of: {type(threshold)}')

    # Check whether the theshold value specified by the user
    # is within the range (0.0, 1.0)
    if threshold.value < 0.0 or threshold.value > 1.0:
        raise IncorrectLimits(f'ERROR: threshold of random data blurring is out '
                              f'of the range [0.0,1.0]: threshold = {threshold.value}')

    # Check whether the probability value specified by the user
    # is within the range (0.0, 1.0)
    if probability < 0.0 or probability > 1.0:
        raise IncorrectLimits(f'ERROR: probability of random data blurring is out '
                              f'of the limits [0.0,1.0]: probability = {probability}')

    # Check whether the directory specified by the user exists.
    # If no, try to create the directory using the path
    # passed by the user to the function
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)


def apply_random_blurring(inp_sample,
                          probability,
                          threshold,
                          lower_lim_region_size,
                          upper_lim_region_size,
                          verbose=False,
                          save_path=None,
                          num_steps_save=None):
    """ Takes an input sample and applies data random blurring.
    At the beginning the function generates a random number within
    the range [0,1) and compares it with the probability value passed
    by the user. If the random number exceeds the probability value
    the function terminates and returns the controls to the caller.
    Otherwise, the function generates random regions within a raw data sample
    volume until the total accumulated region volume exceeds that value
    specified by means of the threshold parameter. The threshold denotes
    the percentage of raw data input sample volume that has to be filled in
    by regions. Regions have different spatial size which is randomly
    generated within the ranges specified by the user.
    Moreover, the volume within a region is blurred by the Gaussian filter.

    Parameters
    ----------
    inp_sample - np.ndarray of int
        raw data input sample with the format:
        [num_channels, depth, width, height]
    probability - float
        probability of applying the data random blurring algorithm
    threshold - ScheduledParameter
        The Parameter controls the level of data random blurring with respect to
        raw data input sample volume
    lower_lim_region_size - np.array of int
        min values of regions sizes along each axis
    upper_lim_region_size - np.array of int
        max values of regions sizes along each axis
    verbose - boolean
        mode that controls text information on the screen
    save_path - str
        path to the files that will contain modified
        raw data input sample in the "h5" format
    num_steps_save - int
        number of steps between writing modified
        raw data input sample in the "h5" format

    Returns
    -------
    None
    """

    if np.random.rand() > probability:
        return

    channels, sample_depth, sample_width, sample_height = inp_sample.shape
    sample_volume = np.prod(inp_sample.shape[1:])

    coord_bounds = [sample_depth, sample_width, sample_height]

    generator = RegionGenerator(coord_bounds,
                                lower_lim_region_size,
                                upper_lim_region_size)

    threshold.step()
    for sample_indx in range(channels):
        blurring_percentage = 0.0
        intersection = set()

        while blurring_percentage < threshold.value:

            region = generator.create_region()

            for k in range(region.coords_lo[0], region.coords_hi[0] + 1):
                for i in range(region.coords_lo[1], region.coords_hi[1] + 1):
                    for j in range(region.coords_lo[2], region.coords_hi[2] + 1):
                        intersection.add((k, i, j))

            snippet = inp_sample[sample_indx,
                                 region.coords_lo[0]:region.coords_hi[0] + 1,
                                 region.coords_lo[1]:region.coords_hi[1] + 1,
                                 region.coords_lo[2]:region.coords_hi[2] + 1]

            gaussian_std = [np.random.randn() * size for size in region.size]
            snippet = ndimage.gaussian_filter(snippet, gaussian_std)

            inp_sample[sample_indx,
                       region.coords_lo[0]:region.coords_hi[0] + 1,
                       region.coords_lo[1]:region.coords_hi[1] + 1,
                       region.coords_lo[2]:region.coords_hi[2] + 1] = snippet

            erased_volume = len(intersection)
            blurring_percentage = erased_volume / sample_volume

        if verbose:
            logger = logging.getLogger('elektronn3log')
            logger.info(f'erased percentage for channel {sample_indx}: {blurring_percentage}')

        if save_path and num_steps_save:
            if (FunctionCallsCounter.counter % num_steps_save) == 0:
                file = h5.File('%sraw_data_input_sample(channel-%i).h5'
                               % (save_path, sample_indx), 'w')
                file.create_dataset("raw_data",
                                    data=inp_sample[sample_indx, :, :, :],
                                    dtype=np.float32)
                file.close()

    FunctionCallsCounter.counter += 1
