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
from typing import Union, Callable, Optional
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
                 multi_gpu: bool = True, normalize_func: Optional[Callable] = None,
                 bs: int = 10):
        self.normalize_func = normalize_func
        self.bs = bs
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

    def predict_proba(self, inp: np.ndarray, bs: Optional[int] = None,
                        verbose: bool = False):
        """

        Args:
            inp: Input data, e.g. of shape [N, C, H, W]
            bs: batch size, ``self.bs`` is used per default.
            verbose: report inference speed

        Returns:

        """
        if bs is None:
            bs = self.bs
        if verbose:
            start = time.time()
        if self.normalize_func is not None:
            inp = self.normalize_func(inp)
        # get output shape shape
        if type(inp) is tuple:
            with torch.no_grad():
                out = self.model(*(torch.Tensor(ii[:2]).to(torch.float32).to(self.device) for ii in inp))
            n_samples = len(inp[0])
        else:
            with torch.no_grad():
                out = self.model(torch.Tensor(inp[:2]).to(torch.float32).to(self.device))
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
                with torch.no_grad():
                    res = self.model(*inp_stride)
            else:
                inp_stride = torch.Tensor(inp[low:high]).to(torch.float32).to(self.device)
                with torch.no_grad():
                    res = self.model(inp_stride)
            if type(res) is tuple:
                for ii in range(len(res)):
                    out[ii][low:high] = res[ii].detach().cpu()
            else:
                out[low:high] = res.detach().cpu()
            if type(inp_stride) is tuple:
                for el in inp_stride:
                    el.detach_()
            else:
                inp_stride.detach_()
            del inp_stride
            del res
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


def load_model(src: str, network_str='') -> nn.Module:
    """
    Load trained elektronn3 model.

    Args:
        src: Source path to model directory. Directory must contain training
        script and model-checkpoint.pth.
        network_str: Specifically for choosing different architecture of U-Net.
                     Choose from ['unet', 'unet++', 'attention-unet']

    Returns:
        Trained model
    """
    train_script = glob.glob(f'{src}/*.py')
    assert len(train_script) == 1, "Multiple/None trainer file(s). " \
                                   "Ill-defined trainer script."
    exec(open(train_script[0]).read(), globals())
    assert "get_model" in globals(), "'get_model' not defiend in trainer script."
    model = get_model(network=network_str)
    state_dict_p = glob.glob(f'{src}/*.pth')
    if len(state_dict_p) > 1:
        last_p = ["state_dict.pth" in sp for sp in state_dict_p]
        if np.sum(last_p) == 1:
            state_dict_p = [state_dict_p[np.argmax(last_p)]]
    assert len(state_dict_p) == 1, "Multiple/None state dict file(s). " \
                                   "Ill-defined state dict file."
    state_dict = torch.load(state_dict_p[0])
    # new trainer class stores more state dicts, we are only interested in the model here
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
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
