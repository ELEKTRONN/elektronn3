# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany

import torch
import torch.nn as nn
import glob
from collections import OrderedDict
import numpy as np
import time
from typing import Union
from elektronn3.training.train_utils import pretty_string_time


class InferenceModel(object):
    """Class to perform inference using a trained elektronn3 model or nn.Module object.

    Args:
        src: Path to training folder of e3 model or already loaded/initialized nn.Module defining the model.
        disable_cuda: use cpu only
        multi_gpu: enable multi-gpu support of pytorch
    Examples:
        >>> cnn = nn.Sequential(
        ... nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
        ... nn.Conv2d(32, 2, 1)).to('cpu')
        >>> inp = np.random.randn(2, 5, 10, 10)
        >>> model = InferenceModel(cnn)
        >>> out = model.predict_proba(inp)
        >>> assert np.all(np.array(out.shape) == np.array([2, 2, 10, 10]))
    """
    def __init__(self, src: Union[str, nn.Module], disable_cuda: bool = False,
                 multi_gpu: bool = True):
        if not disable_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device
        if type(src) is str:
            print('Initializing model from {}.'.format(src))
            self.model = load_model(src)
            self.model_p = src
        else:
            self.model = src
            self.model_p = None
        self.model.eval()
        if multi_gpu:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def predict_proba(self, inp: np.ndarray, bs: int = 10,
                      verbose: bool = False):
        """

        Args:
            inp: Input data, e.g. of shape [N, C, H, W]
            bs: batch size
            verbose: report inference speed

        Returns:

        """
        if verbose:
            start = time.time()
        if type(inp) is np.ndarray:
            inp = torch.Tensor(inp)
        with torch.no_grad():
            # get output shape shape
            if type(inp) is tuple:
                out = self.model(*(torch.Tensor(ii[:2]).to(torch.float32).to(self.device) for ii in inp))
                n_samples = len(inp[0])
            else:
                out = self.model(inp[:2].to(torch.float32).to(self.device))
                n_samples = len(inp)
            # change sample number according to input
            if type(out) is tuple:
                out = tuple(np.zeros([n_samples] + list(out[ii].shape)[1:],
                               dtype=np.float32) for ii in range(len(out)))
            else:
                out = np.zeros([n_samples] + list(out.shape)[1:], dtype=np.float32)
            for ii in range(0, int(np.ceil(n_samples / bs))):
                low = bs * ii
                high = bs * (ii + 1)
                if type(inp) is tuple:
                    inp_stride = tuple(torch.Tensor(ii[low:high]).to(torch.float32).to(self.device) for ii in inp)
                    res = self.model(*inp_stride)
                else:
                    inp_stride = inp[low:high].to(torch.float32).to(self.device)
                    res = self.model(inp_stride)
                if type(res) is tuple:
                    for ii in range(len(res)):
                        out[ii][low:high] = res[ii].cpu()
                else:
                    out[low:high] = res.cpu()
                del inp_stride
                torch.cuda.empty_cache()
            assert high >= n_samples, "Prediction less samples then given" \
                                     " in input."
        if verbose:
            dtime = time.time() - start
            if type(inp) is tuple:
                inp_el = np.sum([float(np.prod(inp[kk].shape)) for kk in range(len(inp))])
            else:
                inp_el = float(np.prod(inp.shape))
            speed = inp_el / dtime / 1e6
            dtime = pretty_string_time(dtime)
            print(f'Inference speed: {speed:.2f} MB or MPix /s, time: {dtime}.')
        return out


def load_model(src: str) -> nn.Module:
    """
    Load trained elektronn3 model.

    Args:
        src: Source path to model directory. Directory must contain training
        script and model-checkpoint.pth.

    Returns:
        Trained model
    """
    train_script = glob.glob(f'{src}/*.py')
    assert len(train_script) == 1, "Multiple/None trainer file(s). " \
                                   "Ill-defined trainer script."
    exec(open(train_script[0]).read(), globals())
    assert "get_model" in globals(), "'get_model' not defiend in trainer script."
    model = get_model()
    state_dict_p = glob.glob(f'{src}/*.pth')
    if len(state_dict_p) > 1:
        final_p = ["final" in sp for sp in state_dict_p]
        if np.sum(final_p) == 1:
            state_dict_p = [state_dict_p[np.argmax(final_p)]]
    assert len(state_dict_p) == 1, "Multiple/None state dict file(s). " \
                                   "Ill-defined state dict file."
    state_dict = torch.load(state_dict_p[0])
    try:
        model.load_state_dict(state_dict)
    # if model was saved as nn.DataParallel then remove 'module.'
    #  prefix in every key
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        try:  # HACK to ensure backwards compat. with example cube models TODO: remove ASAP
            model.load_state_dict(new_state_dict)
        except RuntimeError:
            newnew_state_dict = OrderedDict()
            for k, v in new_state_dict.items():
                newnew_state_dict[k.replace('pretrained_net.', 'base_net.')] = v
            model.load_state_dict(newnew_state_dict)
    return model.eval()
