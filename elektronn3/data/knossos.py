import os
import random
from typing import Callable, Dict, Optional, Sequence

import knossos_utils
import numpy as np
import torch
import torch.utils.data

from elektronn3.data import transforms


class KnossosRawData(torch.utils.data.Dataset):
    """Delivers raw patches that are randomly sampled from a KNOSSOS dataset.
    Supports 2D and 3D (choice is determined from the length of the
    ``patch_shape`` param).

    Args:
        conf_path: Path to KNOSSOS .conf file
        patch_shape: Shape (zyx) of patches that should be sampled from the
            dataset.
        transform: Transformation to be applied to the loaded data.
            See :py:mod:`elektronn3.data.transforms`.
        bounds: Tuple of boundary coordinates (xyz) that constrain the region
            of where patches should be sampled from within the KNOSSOS dataset.
            E.g. ``bounds=((256, 256, 128), (512, 512, 512))`` means that only
            the region between the low corner (x=256, y=256, z=128) and the
            high corner (x=512, y=512, z=512) of the dataset is considered. If
            ``None``, the whole dataset is used.
        mag: KNOSSOS magnification number
        in_memory: If ``True`` (default), the dataset (or the subregion that
            is constrained by ``bounds``) is pre-loaded into memory on
            initialization.
        epoch_size: Determines the length (``__len__``) of the ``Dataset``
            iterator. ``epoch_size`` can be set to an arbitrary value and
            doesn't have any effect on the content of produced training
            samples.
        disable_memory_check: If ``False`` (default), the amount of required
            memory is compared to the amount of free memory. If a planned
            allocation that exceeds 90% of free memory is detected, an error
            is raised. If ``True``, this check is disabled.
        verbose: If ``True``, be verbose about disk I/O.
        caching: If ``True`` and ``in_memory=False``, cache data from disk and reuse it.
        cache_size: How many samples to hold in cache.
        cache_reusages: How often to reuse a sample in cache before loading a new one from disk.
    """

    def __init__(
            self,
            conf_path: str,
            patch_shape: Sequence[int],  # [z]yx
            transform: Callable = transforms.Identity(),
            bounds: Optional[Sequence[Sequence[int]]] = None,  # xyz
            mag: int = 1,
            in_memory: bool = True,
            epoch_size: int = 100,
            disable_memory_check: bool = False,
            verbose: bool = False,
            caching: bool = False,
            cache_size: int = 50,
            cache_reusages: int = 10
    ):
        self.conf_path = conf_path
        self.patch_shape = np.array(patch_shape)
        self.transform = transform
        self.mag = mag
        self.in_memory = in_memory
        self.epoch_size = epoch_size
        self.disable_memory_check = disable_memory_check
        self.verbose = verbose
        self.caching = caching
        self.cache_size = cache_size
        self.cache_reusages = cache_reusages

        self.kd = knossos_utils.KnossosDataset(self.conf_path, show_progress=self.verbose)
        self.dim = len(self.patch_shape)
        patch_shape_xyz = self.patch_shape[::-1]  # zyx -> xyz
        if self.dim == 2:
            patch_shape_xyz = np.array([*patch_shape_xyz, 1])  # z=1 for 2D
        self.patch_shape_xyz = patch_shape_xyz
        if bounds is None:
            bounds = [[0, 0, 0], self.kd.boundary]
        self.bounds = np.array(bounds)
        self.shape = self.bounds[1] - self.bounds[0]
        self.raw = None  # Will be filled with raw data if in_memory is True
        if self.in_memory:
            self._load_into_memory()
        elif self.caching:
            self._fill_cache()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        inp = self._load_from_memory() if self.in_memory else \
            (self._get_from_cache() if self.caching else self._load_from_disk())
        if self.dim == 2:
            inp = inp[0]  # squeeze z=1 dim -> yx
        inp = inp.astype(np.float32)[None]  # Prepend C dim -> (C, [D,] H, W)
        inp, _ = self.transform(inp, None)
        sample = {
            'inp': torch.as_tensor(inp)
        }
        return sample

    def _load_from_disk(self) -> np.ndarray:
        min_offset = self.bounds[0]  # xyz
        max_offset = self.bounds[1] - self.patch_shape_xyz  # xyz
        offset = np.random.randint(min_offset, max_offset + 1)  # xyz
        inp = self.kd.load_raw(
            offset=offset, size=self.patch_shape_xyz, mag=self.mag
        )  # zyx (D, H, W)
        return inp

    def _load_from_memory(self) -> np.ndarray:
        min_offset = (0, 0, 0)  # 0 because self.raw already accounts for min offset
        max_offset = self.shape - self.patch_shape_xyz
        offset = np.random.randint(min_offset, max_offset + 1)  # xyz
        inp = self.raw[  # self.raw has zyx dim order
              offset[2]:offset[2] + self.patch_shape_xyz[2],
              offset[1]:offset[1] + self.patch_shape_xyz[1],
              offset[0]:offset[0] + self.patch_shape_xyz[0],
              ]  # zyx (D, H, W)
        return inp

    def __len__(self) -> int:
        return self.epoch_size

    def _load_into_memory(self) -> None:
        if not self.disable_memory_check:
            self.memory_check()
        if self.verbose:
            print('Loading dataset into memory...')
        self.raw = self.kd.load_raw(
            offset=self.bounds[0], size=self.shape, mag=self.mag
        )  # zyx (D, H, W)

    def memory_check(self) -> None:
        # Memory cost in GiB: Number of gibivoxels * 4 because each uint8 voxel needs 1 byte.
        required_mem_gib = np.prod(self.bounds[1] - self.bounds[0]) / 1024 ** 3 * 1
        # Total RAM on the system in GiB
        free_mem_gib = float(os.popen('free -tm').readlines()[-1].split()[3]) / 1024
        mem_frac = required_mem_gib / free_mem_gib
        if mem_frac > 0.9:
            raise RuntimeError(
                f'Using in_memory and bounds of {self.bounds.tolist()} would require at least '
                f'{required_mem_gib:.2f} GiB of memory for each dataset instance.\n'
                'Please specifiy smaller bounds, or if you are sure about what you are '
                'doing, you can disable this check with the disable_memory_check flag.'
            )

    def _fill_cache(self):
        self.cache = [self._load_from_disk() for _ in range(self.cache_size)] 
        self.remaining_cache_reusages = [self.cache_reusages] * self.cache_size

    def _get_from_cache(self):
        idx = random.randrange(self.cache_size)
        inp = self.cache[idx]
        self.remaining_cache_reusages[idx] -= 1
        if self.remaining_cache_reusages[idx] < 1:
            self.cache[idx] = self._load_from_disk()
            self.remaining_cache_reusages[idx] = self.cache_reusages
        return inp
