# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Marius Killinger

import torch.nn as nn
import numpy as np
import tqdm
import time
from elektronn3 import floatX
from elektronn3.data.utils import as_floatX
from elektronn3.training.train_utils import pretty_string_time


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def predict_dense(self, raw):
        predict_dense(raw, self.__call__)


# TODO: This needs some work
def predict_dense(raw_img, pred_func, as_uint8=False, pad_raw=True,
                  offset=(0, 0, 0), strides=(1, 1, 1), n_lab=2, patch_size=(16, 128, 128)):
    # TODO: Infer offset, strides, n_lab, patch_shape from neural class. best: add functionality to BaseModule and
    # pass BaseModule inherited models to this function
    """
    Core function that performs the inference

    Parameters
    ----------
    raw_img : np.ndarray
        raw data in the format (ch, (z,) y, x)
    as_uint8: Bool
        Return class probabilites as uint8 image (scaled between 0 and 255!)
    pad_raw: Bool
        Whether to apply padding (by mirroring) to the raw input image
        in order to get predictions on the full image domain.

    Returns
    -------
    np.ndarray
        Predictions.
    """
    # if self.shape.ndim < 2:
    #     print("'predict_dense' works only for nodes with 2 or 3 "
    #                  "spatial axes, this node has shape %s." % (self.shape))
    #     return None
    #
    # if self.shape.ndim == 2 and raw_img.ndim == 3 and \
    #                 raw_img.shape[0] != self.input_nodes[0].shape['f']:
    #     print("If 3d input is given to 2d CNNs, the first axis must"
    #                  "contain the features/channel and have size %i. Note"
    #                  "that also 4d input can be given to 2d CNNs in the "
    #                  "axis order (ch, z, y, x), where for each z-slice a"
    #                  "2d prediction image is created"
    #                  % self.input_nodes[0].shape['f'])
    #     return None

    if np.any(np.less(offset, 0)):
        # TODO: This shouldn't apply to elektronn3 anymore.
        raise ValueError("Cannot predict dense because the CNN contains "
                         "UpConvs which cause unknown FOVs. If you use "
                         "UpConvs you should not need predict dense anyway!")

    if raw_img.dtype in [np.int, np.int8, np.int16, np.int32, np.uint32, np.uint,
                         np.uint8, np.uint16, np.uint32, np.uint32]:
        m = 255
    else:
        m = 1

    raw_img = as_floatX(raw_img) / m

    time_start = time.time()
    strip_z = False
    if raw_img.ndim == 3:
        strip_z = True
        raw_img = raw_img[:, None]  # add singleton z-channel  # TODO: Correct order?
    #
    # if self.shape.ndim == 2:
    #     raise NotImplementedError

    if pad_raw:
        raw_img = np.pad(
            raw_img,
            [
                (0, 0),
                (offset[0], offset[0]),
                (offset[1], offset[1]),
                (offset[2], offset[2])
            ],
            mode='symmetric'
        )

    raw_sh = raw_img.shape[1:]  # only spatial, not channels
    tile_sh = np.add(patch_size, strides) - 1
    prob_sh = np.multiply(patch_size, strides)
    prob_arr = np.zeros(np.concatenate([[n_lab, ], prob_sh]), dtype=floatX)

    pred_sh = np.array([raw_sh[0] - 2 * offset[0], raw_sh[1] - 2 * offset[1], raw_sh[2] - 2 * offset[2]])
    if as_uint8:
        predictions = np.zeros(np.concatenate(([n_lab, ], pred_sh)), dtype=np.uint8)
    else:
        predictions = np.zeros(np.concatenate(([n_lab, ], pred_sh)), dtype=floatX)

    # Calculate number of tiles (in 3d: blocks) that need to be performed
    z_tiles = int(np.ceil(float(pred_sh[0]) / prob_sh[0]))
    x_tiles = int(np.ceil(float(pred_sh[1]) / prob_sh[1]))
    y_tiles = int(np.ceil(float(pred_sh[2]) / prob_sh[2]))
    total_nb_tiles = int(np.product([x_tiles, y_tiles, z_tiles]))

    print("Predicting img %s in %i Blocks: (%i, %i, %i)" \
                % (raw_img.shape, total_nb_tiles, z_tiles, x_tiles, y_tiles))

    pbar = tqdm.tqdm(total=np.prod(pred_sh), ncols=80, leave=False, unit='Vx',
                     unit_scale=True, dynamic_ncols=False)
    for z_t in range(z_tiles):
        for x_t in range(x_tiles):
            for y_t in range(y_tiles):
                # For every z_tile a slice of thickness cnn_out_sh[2] is
                # collected and then collectively written to the output_data
                raw_tile = raw_img[
                    :,
                    z_t * prob_sh[0]:z_t * prob_sh[0] + tile_sh[0],
                    x_t * prob_sh[1]:x_t * prob_sh[1] + tile_sh[1],
                    y_t * prob_sh[2]:y_t * prob_sh[2] + tile_sh[2]
                ]

                this_is_end_tile = False if np.all(np.equal(raw_tile.shape[1:], tile_sh)) else True

                if this_is_end_tile:  # requires 0-padding
                    right_pad = np.subtract(tile_sh, raw_tile.shape[1:])  # (ch,z,x,y)
                    right_pad = np.concatenate(([0, ], right_pad))  # for channel dimension
                    left_pad = np.zeros(raw_tile.ndim, dtype=np.int)
                    pad_with = list(zip(left_pad, right_pad))
                    raw_tile = np.pad(raw_tile, pad_with, mode='constant')

                # if self.shape.ndim == 2:
                #     # slice from raw_tile(ch,z,x,y) --> (ch,x,y)
                #     prob_arr = pred_func(raw_tile[:, 0], prob_arr)  # returns (ch,z=1,x,y)
                #     prob = prob_arr
                if 1:
                    prob_arr = pred_func(raw_tile, prob_arr)
                    prob = prob_arr

                if this_is_end_tile:  # cut away padded range
                    prob = prob[:, :prob_sh[0] - right_pad[1], :prob_sh[1] - right_pad[2], :prob_sh[2] - right_pad[3]]

                if as_uint8:
                    prob *= 255

                predictions[
                    :,
                    z_t * prob_sh[0]:(z_t + 1) * prob_sh[0],
                    x_t * prob_sh[1]:(x_t + 1) * prob_sh[1],
                    y_t * prob_sh[2]:(y_t + 1) * prob_sh[2]
                ] = prob

                # print ("shape", prob.shape)
                current_buffer = np.prod(prob_sh)
                pbar.update(current_buffer)

    pbar.close()
    dtime = time.time() - time_start
    speed = np.product(predictions.shape[1:]) * 1.0 / 1000000 / dtime
    dtime = pretty_string_time(dtime)
    print(" Inference speed: %.3f MB or MPix /s, time %s" % (speed, dtime))

    if strip_z:
        predictions = predictions[:, 0, :, :]

    return predictions
