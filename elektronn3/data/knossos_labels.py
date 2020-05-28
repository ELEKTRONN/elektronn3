import os
import random
from typing import Callable, Dict, Optional, Sequence

import knossos_utils
import numpy as np
import torch
import torch.utils.data

from elektronn3.data import transforms
from elektronn3.data.knossos import KnossosRawData


class KnossosLabels(torch.utils.data.Dataset):
    """Delivers label and raw data as patches that are randomly
        sampled from a KNOSSOS dataset. The labels are extracted from a kzip
        file and the corresponding raw data associated with them is returned.
        Supports 2D and 3D (choice is determined from the length of the
        ``patch_shape`` param).

        Args:
            conf_path_label: Path to KNOSSOS .conf file corresponding to the labels
            conf_path_raw_data: Path to KNOSSOS .conf file corresponding to the raw data
            dir_path_label: Directory containing label kzip files
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
            bounds of different patches in the knossos dataset (check get_label_info function to
            obtain information on different bounds and the data associated with them in the label
            dataset). If ``None``, then whole dataset is returned.
           """

    def __init__(
            self,
            conf_path_label: str,
            conf_path_raw_data: str,
            dir_path_label: str,
            patch_shape: Sequence[int],  # [z]yx
            transform: Callable = transforms.Identity(),
            mag: int = 1,
            epoch_size: int = 100,
            label_names: Optional[Sequence[str]] = None,
            knossos_bounds: Optional[Sequence[Sequence[Sequence[int]]]] = None  # xyz

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
        self.kd = knossos_utils.KnossosDataset(self.conf_path_label, show_progress=False)
        self.targets = []
        self.inp_raw_data = []
        self.file_bounds = {}
        self.kzip_files_path = []
        self.dir_path = dir_path_label
        self.knossos_bounds = knossos_bounds

        self._get_file_bounds(label_names)
        self._get_data()

    def _get_file_bounds(self, label_names):
        if self.dir_path:
            if label_names is None:
                self.kzip_files_path = [f for f in os.listdir(self.dir_path) if f.endswith('.k.zip')]
            else:
                for name in label_names:
                    self.kzip_files_path += [f for f in os.listdir(self.dir_path) if f.endswith(name + '.k.zip')]

            for file in self.kzip_files_path:
                zip_path = self.dir_path + "/" + file
                bounds = self.kd.get_movement_area(zip_path)
                offset = tuple(bounds[0])
                max_values = tuple(bounds[1])
                if (offset, max_values) not in self.file_bounds.keys():
                    self.file_bounds[(offset, max_values)] = [zip_path]
                else:
                    self.file_bounds[(offset, max_values)].append(zip_path)
        else:
            raise OSError(f"Path to conf directory is empty")

    def _get_data(self):
        for bounds, paths in self.file_bounds.items():
            if self.knossos_bounds is None or bounds in self.knossos_bounds:
                size = (bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1], bounds[1][2] - bounds[0][2])
                labels_patch = np.zeros(size[::-1])
                location_per_label = []
                for zip_path in paths:
                    data = self.kd._load_kzip_seg(zip_path, bounds[0], size,
                                                  self.mag, datatype=np.int32,
                                                  padding=0,
                                                  apply_mergelist=False,
                                                  return_dataset_cube_if_nonexistent=False,
                                                  expand_area_to_mag=False)

                    labels_patch += data
                    location_per_label.append(np.transpose(np.nonzero(data)))

                overlapping_indices = self._get_overlapping_indices(location_per_label)

                for i in overlapping_indices:
                    labels_patch[i] = 0  # setting voxels with overlapping labels to background

                self.targets.append({'data': labels_patch, 'min_bound': bounds[0],
                                     'max_bound': bounds[1]})  # zyx form

                self.inp_raw_data.append(KnossosRawData(conf_path=self.conf_path_raw_data,
                                                        patch_shape=self.patch_shape,
                                                        transform=None,
                                                        bounds=bounds,
                                                        mag=self.mag, mode='disk', epoch_size=self.epoch_size,
                                                        disable_memory_check=False, verbose=False))  # xyz form

    @staticmethod
    def _get_overlapping_indices(location_per_label):

        indices_per_label = [{tuple(elem) for elem in location_per_label[i]} for i in range(len(location_per_label))]

        overlapping_indices = []
        for i in range(len(indices_per_label) - 1):
            for j in range(i + 1, len(indices_per_label)):
                overlapping_indices += (indices_per_label[i].intersection(indices_per_label[j]))

        return overlapping_indices

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:

        random_idx = random.randrange(len(self.inp_raw_data))

        raw_dict = self.inp_raw_data[random_idx].__getitem__(0)
        inp = raw_dict['inp']  # zyx
        offset = raw_dict["offset"]  # xyz

        min_indices = offset[::-1] - self.targets[random_idx]['min_bound'][::-1]
        max_indices = min_indices + self.patch_shape_xyz[::-1]
        label = self.targets[random_idx]['data'][
            tuple(slice(*i) for i in zip(min_indices, max_indices))]

        if self.dim == 2:
            label = label[0]

        inp, label = self.transform(inp.data.numpy(), label)
        label = label.astype('int32')
        sample = {
            'inp': torch.as_tensor(inp),
            'target': torch.as_tensor(label)
        }
        return sample

    def __len__(self) -> int:
        return self.epoch_size

    def get_label_info(self):
        return self.targets
