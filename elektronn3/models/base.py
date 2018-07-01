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
from elektronn3.training.train_utils import pretty_string_time


class InferenceModel(object):
    def __init__(self, src, disable_cuda=False, multi_gpu=True):
        if not disable_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device
        if type(src) is str:
            self.model = load_model(src)
        else:
            self.model = src
            self.model.eval()
        if multi_gpu:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def predict_proba(self, inp, bs=10, verbose=False):
        if verbose:
            start = time.time()
        if type(inp) is np.ndarray:
            inp = torch.Tensor(inp)
        inp = inp.to(torch.float32).to(self.device)
        with torch.no_grad():
            # get output shape shape
            out = self.model(inp[:1])
            # change sample number according to input
            out = np.zeros([len(inp)] + list(out.shape)[1:], dtype=np.float32)
            for ii in range(0, int(np.ceil(len(inp) / bs))):
                low = bs * ii
                high = bs * (ii + 1)
                inp_stride = inp[low:high]
                out[low:high] = self.model(inp_stride)
            assert high >= len(inp), "Prediction less samples then given" \
                                     " in input."
        if verbose:
            dtime = time.time() - start
            speed = float(np.prod(inp.shape)) / dtime / 1e6
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
        model.load_state_dict(new_state_dict)
    return model.eval()
