import collections
import os
import random
from typing import Callable, Dict, Optional, Sequence

import knossos_utils
import numpy as np
import torch
import torch.utils.data
from elektronn3.data import transforms
from elektronn3.data.knossos import KnossosRawData


class KnossosLabelsNozip(torch.utils.data.Dataset):
    """Delivers label and raw data as patches that are randomly
        sampled from a KNOSSOS dataset. The labels are extracted from a kzip
        file and the corresponding raw data associated with them is returned.
        Supports 2D and 3D (choice is determined from the length of the
        ``patch_shape`` param).

        Args:
            conf_path_label: Path to KNOSSOS .conf file corresponding to the labels
            conf_path_raw_data: Path to KNOSSOS .conf file corresponding to the raw data
            #dir_path_label: Directory containing label kzip files
            patch_shape: Shape (zyx) of patches that should be sampled from the
                dataset.
            transform: Transformation to be applied to the loaded data.
                See :py:mod:`elektronn3.data.transforms`.
            mag: KNOSSOS magnification number
            epoch_size: Determines the length (``__len__``) of the ``Dataset``
                iterator. ``epoch_size`` can be set to an arbitrary value and
                doesn't have any effect on the content of produced training
                samples.
            label_names: A sequence of labels to be extracted. Extraction of label data is
                only done on files of the form ``label_names.k.zip``. If``None``, label
                data from all the kzip files is extracted. Useful when each kzip contains data
                for only one label.
            knossos_bounds: List of Tuple of boundary coordinates (xyz) that constrain the region
            of where patches should be sampled from within the KNOSSOS dataset.
            E.g. ``knossos_bounds=[((256, 256, 128), (512, 512, 512))]`` means that only
            the region between the low corner (x=256, y=256, z=128) and the high corner
            (x=512, y=512, z=512) of the dataset is considered. The bounds should be one of the
            bounds of different patches in the knossos dataset. If ``None``, then whole dataset is returned.
            label_offset: offset of the indices in the knossos labels, default is 0.
            label_order: change the order of the classes in the labels
            raw_mode: Dataloading mode for raw data. One of ``in_memory``, ``caching`` or ``disk``. If ``in_memory`` (default),
            the dataset (or the subregion
            that is constrained by ``bounds``) is pre-loaded into memory on initialization. If ``caching``, cache data
            from the disk and reuse it. If ``disk``, load data from disk on demand.
           """

    def __init__(
            self,
            conf_path_label: str,
            conf_path_raw_data: str,
            #dir_path_label: str,
            patch_shape: Sequence[int],  # [z]yx
            transform: Callable = transforms.Identity(),
            mag: int = 1,
            epoch_size: int = 100,
            #label_names: Optional[Sequence[str]] = None,
            knossos_bounds: Optional[Sequence[Sequence[Sequence[int]]]] = None,  # xyz
            label_offset: int = 0,
            label_order: Optional[Sequence[int]] = None,
            raw_mode : str = "disk",
            raw_disable_memory_check: bool = False,
            raw_verbose: bool = False,
            raw_cache_size: int = 50,
            raw_cache_reuses: int = 10
    ):
        self.conf_path_label = conf_path_label
        self.conf_path_raw_data = conf_path_raw_data
        self.patch_shape = np.array(patch_shape)
        self.dim = len(self.patch_shape)
        patch_shape_xyz = self.patch_shape[::-1]  # zyx -> xyz
        if self.dim == 2:
            patch_shape_xyz = np.array([*patch_shape_xyz, 1])  # z=1 for 2D
        self.patch_shape_xyz = patch_shape_xyz
        self.transform = transform
        self.mag = mag
        self.epoch_size = epoch_size
        self.inp_targets = []
        if knossos_bounds is not None:
            self.knossos_bounds = np.array(knossos_bounds)
        self.knossos_bounds = knossos_bounds
        self.label_offset = label_offset  # todo: verify correct handling of this offset
        self.label_order = label_order
        self.raw_mode = raw_mode
        self.raw_disable_memory_check = raw_disable_memory_check
        self.raw_verbose = raw_verbose
        self.raw_cache_reuses = raw_cache_reuses
        self.raw_cache_size = raw_cache_size
        self.offset_history = []


        #raw data as input
        #specify no elektronn3 transformation in the raw data loader because transform is applied
        #after loading the data from KnossosRawData, outside of it's scope
        self.inp_raw_data_loader = KnossosRawData(conf_path=self.conf_path_raw_data,
                                      patch_shape=self.patch_shape,
                                      bounds=self.knossos_bounds,
                                      mag=self.mag, mode=self.raw_mode, epoch_size=self.epoch_size,
                                      disable_memory_check=self.raw_disable_memory_check,
                                      verbose=self.raw_verbose, cache_size=self.raw_cache_size,
                                      cache_reuses = self.raw_cache_reuses)  # xyz form

        #labels as target
        self.label_target_loader = knossos_utils.KnossosDataset(self.conf_path_label, show_progress=False)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        
        #generate raw-data sample using the __getitem__() method of the KnossosRawData class
        #and store the position the sample is taken from as offset
        #Note 1: index is always taken to be 0 because the __getitem__() method disregards the index
        #anyway and randomly samples a patch within the given bounds (+label offset)
        #Note 2: the KnossosRawData loader outputs torch.tensor types, so in order to use the data
        #retrieved from KnossosRawData for the elektronn3 trafo one must cast it to numpy array
        input_dict = self.inp_raw_data_loader[0]
        inp = input_dict["inp"].numpy() #czyx
        offset_from_raw = input_dict["offset"]
        self.offset_history.append(offset_from_raw)

        #use the offset retrieved from calling the __getitem__() method to load the corresponding label patch from the
        #label data
        label = self.label_target_loader.load_seg(offset= offset_from_raw + self.label_offset, size = self.patch_shape,
                                                    mag = self.mag, datatype = np.int64)

        if self.dim == 2:
            label = label[0]

        #apply elektronn3 transforms
        inp, label = self.transform(inp, label)

        sample = {
            'inp': torch.as_tensor(inp),
            'target': torch.as_tensor(label).long(),
            'fname': self.conf_path_label
        }
        if self.label_order is not None:
            sample_target = sample['target'].detach().clone()
            for i, label in enumerate(self.label_order):
                sample['target'][sample_target == i] = label
        return sample

    def __len__(self) -> int:
        return self.epoch_size
