# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Martin Drawitsch

import torch
import torch.nn as nn
import glob
from collections import OrderedDict
import numpy as np
import time
from typing import Union, Optional


class InferenceModel:
    """Class to perform inference using a nn.Module object either passed
    directly or loaded from an elektronn3 training folder.

    Args:
        model: Path to training folder of e3 model or already loaded/initialized
             nn.Module defining the model.
        state_dict_src: Path to ``state_dict`` file (.pth) or loaded
            ``state_dict`` or ``None``. If not ``None``, the ``state_dict`` of
            the ``model`` is replaced with it.
        disable_cuda: Use cpu only
        multi_gpu: Enable multi-GPU support of PyTorch
    Examples:
        >>> cnn = nn.Sequential(
        ...     nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
        ...     nn.Conv2d(32, 2, 1))
        >>> inp = np.random.randn(2, 5, 10, 10)
        >>> model = InferenceModel(cnn)
        >>> out = model.predict_proba(inp)
        >>> assert np.all(np.array(out.shape) == np.array([2, 2, 10, 10]))
    """
    def __init__(
            self,
            model: nn.Module,
            state_dict_src: Optional[Union[str, dict]] = None,
            disable_cuda: Optional[bool] = False,
            multi_gpu: Optional[bool] = True
    ):
        if not disable_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device
        self.model = model
        if type(state_dict_src) is str:
            state_dict = torch.load(state_dict_src[0])
        elif type(state_dict_src) is dict or state_dict_src is None:
            state_dict = state_dict_src
        else:
            raise ValueError('"state_dict_src" has to be path to .pth file (str'
                             ') or state dict object (dict) or None.')
        if state_dict is not None:
            try:
                model.load_state_dict(state_dict)
            # if model was saved as nn.DataParallel then remove 'module.'
            #  prefix in every key
            except RuntimeError:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k.replace('module.', '')] = v
                model.load_state_dict(new_state_dict)
        self.model.eval()
        if multi_gpu:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def predict_proba(
            self,
            inp: Union[np.ndarray, torch.Tensor],
            batch_size: int = 10,
            verbose: Optional[bool] = False,
            out_shape: Optional[tuple] = None
    ):
        """

        Args:
            inp: Input data, e.g. of shape [N, C, H, W]
            batch_size: batch size
            verbose: report inference speed
            out_shape: output shape, will be inferred if None

        Returns:
            Model output
        """
        if verbose:
            start = time.time()
        inp = torch.as_tensor(inp)
        n_samples = inp.shape[0]
        with torch.no_grad():
            # change sample number according to input
            if out_shape is None:
                out = self.model(inp[:1].to(self.device, torch.float32))
                out_shape = tuple([n_samples] + list(out.shape)[1:])
            out = np.zeros(out_shape, dtype=np.float32)
            for batch_ix in range(0, int(np.ceil(n_samples / batch_size))):
                low = batch_size * batch_ix
                high = batch_size * (batch_ix + 1)
                inp_stride = inp[low:high].to(self.device, torch.float32)
                out[low:high] = self.model(inp_stride)
                del inp_stride
            assert high >=n_samples, "Prediction less samples then given" \
                                     " in input."
        if verbose:
            dtime = time.time() - start
            speed = inp.numel() / dtime / 1e6
            print(f'Inference speed: {speed:.2f} MPix /s, time: {dtime:.2f}.')
        return out


def load_trained_model(src: str) -> InferenceModel:
    """
    Load trained elektronn3 model.

    Args:
        src: Source path to model directory. Directory must contain training
            script and model-checkpoint.pth.

    Returns:
        Trained model
    """
    # get architecture definition
    train_script = glob.glob(f'{src}/*.py')
    assert len(train_script) == 1, "Multiple/None trainer file(s). " \
                                   "Ill-defined trainer script."
    exec(open(train_script[0]).read(), globals())
    assert "get_model" in globals(), "'get_model' not defiend in trainer script."
    # get state dict path
    state_dict_p = glob.glob(f'{src}/*.pth')
    if len(state_dict_p) > 1:
        final_p = ["final" in sp for sp in state_dict_p]
        if np.sum(final_p) == 1:
            state_dict_p = [state_dict_p[np.argmax(final_p)]]
    assert len(state_dict_p) == 1, "Multiple/None state dict file(s). " \
                                   "Ill-defined state dict file."
    model = InferenceModel(get_model(), state_dict_p[0])
    return model
