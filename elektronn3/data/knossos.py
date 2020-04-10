from typing import Callable, Sequence

import knossos_utils
import numpy as np
import torch
import torch.utils.data

from elektronn3.data import transforms


class KnossosRawData(torch.utils.data.Dataset):
    """Delivers raw patches that are randomly sampled from a KNOSSOS dataset.
    Supports 2D and 3D (choice is determined from the length of the
    ``patch_shape`` param."""
    def __init__(
            self,
            conf_path: str,
            patch_shape: Sequence[int],
            transform: Callable = transforms.Identity(),
            epoch_size=100
    ):
        self.conf_path = conf_path
        self.patch_shape = np.array(patch_shape)
        self.transform = transform
        self.epoch_size = epoch_size
        self.kd = knossos_utils.KnossosDataset(self.conf_path, show_progress=False)
        self.dim = len(self.patch_shape)

    def __getitem__(self, index):
        size = self.patch_shape[::-1]  # zyx -> xyz
        if self.dim == 2:
            size = np.array([*size, 1])  # z=1 for 2D
        max_offset = self.kd.boundary - size

        offset = np.random.randint((0, 0, 0), max_offset + 1, (3,))
        inp = self.kd.load_raw(
            offset=offset,
            size=size,
            mag=1
        )  # zyx (D, H, W)
        if self.dim == 2:
            inp = inp[0]  # squeeze z=1 dim -> yx

        inp = inp.astype(np.float32)[None]  # Prepend C dim -> (C, [D,] H, W)
        inp, _ = self.transform(inp, None)
        sample = {
            'inp': torch.as_tensor(inp)
        }
        return sample

    def __len__(self):
        return self.epoch_size
