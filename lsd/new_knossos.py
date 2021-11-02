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
        sampled from a KNOSSOS dataset. The labels are not extracted from a kzip
        file, instead a .conf or pyk.conf file is expected, and the corresponding raw data associated with them is returned.
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
            threshold_background_fraction: float [0.,1.] or None. If a float, this denotes the percentage of background that is tolerated in each sample.
            The "empty" voxels, i.e. voxels with value of 0, are counted, and if the fraction of empty voxels exceeds the threshold
            the sample is skipped and a new one is generated. Default None.
            save_interval: during the training this object will record all the coordinates of the samples it encountered. these and other
            meta-data are saved in a dict which is stored in the training directory
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
            raw_cache_reuses: int = 10,
            threshold_background_fraction: float = 0.05, #maximum fraction of background (segmentation==0) in a sample
            save_interval: int = None

    ):

        self.conf_path_label = conf_path_label
        self.conf_path_raw_data = conf_path_raw_data
        self.patch_shape = np.array(patch_shape)
        self.dim = len(self.patch_shape)
        patch_shape_xyz = self.patch_shape[::-1]  # zyx -> xyz
        if self.dim == 2:
            patch_shape_xyz = np.array([*patch_shape_xyz, 1])  # z=1 for 2D
        self.patch_shape_xyz = patch_shape_xyz
        self.patch_volume = np.prod(self.patch_shape_xyz)
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
        self.coordinate_history = []
        self.threshold_background_fraction = threshold_background_fraction
        self.save_interval = save_interval
        self.epoch = 0

        #set up the saving dictionary for saving in the training folder
        self.samples_history_dictionary = {}
        self.samples_history_dictionary["conf_path_label"] = self.conf_path_label
        self.samples_history_dictionary["conf_path_raw_data"] = self.conf_path_raw_data
        self.samples_history_dictionary["patch_shape_zyx"] = self.patch_shape
        self.samples_history_dictionary["save_interval"] = self.save_interval
        self.samples_history_dictionary["threshold_background_fraction"] = self.threshold_background_fraction        
        self.samples_history_dictionary["label_offset"] = self.label_offset
        self.samples_history_dictionary["label_order"] = self.label_order
        self.samples_history_dictionary["raw_mode"] = self.raw_mode
        
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
        #and store the position the sample is taken from as coordinate_from_raw (not adjusted for any)
        #Note 1: index is always taken to be 0 because the __getitem__(self, index) method disregards the index
        #anyway and randomly samples a patch within the given bounds (+label_offset)
        #Note 2: the KnossosRawData loader outputs torch.tensor types, so in order to use the data
        #retrieved from KnossosRawData for the elektronn3 transformation one must cast it to numpy array

        input_dict = self.inp_raw_data_loader[0]
        coordinate_from_raw = input_dict["offset"]#xyz

        #use the position_from_raw retrieved from calling the __getitem__() method to load the corresponding label patch from the
        #label data
        #Note: KnossosDataset.load_seg() requires the patch_shape in the xyz-format, while KnossosRawData takes care of
        #the zyx->xyz conversion inside it's scope
        label = self.label_target_loader.load_seg(offset= coordinate_from_raw + self.label_offset, size = self.patch_shape_xyz,
                                                    mag = self.mag, datatype = np.int64)
        
        

        #count calculate the fraction of empty (value 0 in segmentation) voxels in the sample (segmentation). The loader generates new samples so long until it finds
        #one with a fraction of background lower then the threshold background fraction
        while np.count_nonzero(label==0) / self.patch_volume > self.threshold_background_fraction:
            input_dict = self.inp_raw_data_loader[0]
            coordinate_from_raw = input_dict["offset"]#xyz
            label = self.label_target_loader.load_seg(offset= coordinate_from_raw + self.label_offset, size = self.patch_shape_xyz,
                                                        mag = self.mag, datatype = np.int64)
   
        inp = input_dict["inp"].numpy() #czyx
        coordinate_from_raw = input_dict["offset"]
        
        self.samples_history_dictionary["epoch {}".format(self.epoch)] =  {"coordinate_raw_xyz": coordinate_from_raw}

        if self.save_interval is not None:
            if self.epoch % self.save_interval == 0:
                with open(self.savepath + "samples_history.json", "w") as f:
                    json.dump(self.samples_history_dictionary, f)

        if self.dim == 2:
            label = label[0]

        #apply elektronn3 transforms
        trafo_inp, target = self.transform(inp, label)

        sample = {
            'inp': torch.as_tensor(trafo_inp),#czyx
            'target': torch.as_tensor(target),#.long(), zyx
            'fname': self.conf_path_label,
            'coordinate_raw': coordinate_from_raw,#xyz
            'segmentation': torch.as_tensor(inp)
        }
        #if self.label_order is not None:
        #    sample_target = sample['target'].detach().clone()
        #    for i, label in enumerate(self.label_order):
        #        sample['target'][sample_target == i] = label
        self.epoch += 1
        return sample

    def __len__(self) -> int:
        return self.epoch_size
