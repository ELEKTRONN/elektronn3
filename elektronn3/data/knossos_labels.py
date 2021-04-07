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
            bounds of different patches in the knossos dataset. If ``None``, then whole dataset is returned.
            label_offset: offset of the indices in the knossos labels, default is 0.
            label_order: change the order of the classes in the labels
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
            knossos_bounds: Optional[Sequence[Sequence[Sequence[int]]]] = None,  # xyz
            label_offset: int = 0,
            label_order: Optional[Sequence[int]] = None
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
        self.inp_targets = []
        self.file_bounds = {}
        self.kzip_files_path = []
        self.dir_path = dir_path_label
        self.knossos_bounds = knossos_bounds
        self.label_offset = label_offset  # todo: verify correct handling of this offset
        self.label_order = label_order

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

            for offset, max_values in self.file_bounds.keys():
                if len(self.file_bounds[offset, max_values]) != len(label_names):
                    print(f"Warning: Found {len(self.file_bounds[offset, max_values])} label files for "
                          f"offset {offset} and max_values {max_values}, but expected {len(label_names)}")
        else:
            raise OSError(f"Path to conf directory is None!")

    def _get_data(self):
        voxels_per_patch = []
        for bounds, paths in self.file_bounds.items():
            if self.knossos_bounds is None or bounds in self.knossos_bounds:
                size = tuple(np.array(bounds[1]) - np.array(bounds[0]))
                labels_patch = np.zeros(size[::-1])  # the labels of the current patch
                non_background_label_locations = np.zeros_like(labels_patch)
                for zip_path in paths:
                    data = self.kd._load_kzip_seg(zip_path, bounds[0], size,
                                                  self.mag,
                                                  padding=0,
                                                  apply_mergelist=False,
                                                  return_dataset_cube_if_nonexistent=False,
                                                  expand_area_to_mag=False)
                    labels_patch += data
                    non_background_label_locations += data != 0

                overlapping_locations = non_background_label_locations > 1

                if overlapping_locations.sum() != 0:
                    print(
                        f"Detected {overlapping_locations.sum()} overlapping/contradicting labels for "
                        f"labels with bounds {bounds}")
                    # todo: better assign one of the labels instead of background?
                    labels_patch[overlapping_locations] = 0

                # todo: maybe just create one KnossosRawData instance with no bounds, then sample directly?
                inp_raw_data = KnossosRawData(conf_path=self.conf_path_raw_data,
                                              patch_shape=self.patch_shape,
                                              bounds=np.array(bounds) + self.label_offset,
                                              mag=self.mag, mode='disk', epoch_size=self.epoch_size,
                                              disable_memory_check=False, verbose=False)  # xyz form
                target = {'data': labels_patch, 'min_bound': bounds[0],
                          'max_bound': bounds[1], 'fname': paths}  # zyx form
                self.inp_targets.append((inp_raw_data, target))
                voxels_per_patch.append(labels_patch.size)

        self.patch_weights = np.array(voxels_per_patch) / sum(voxels_per_patch)
        print("patch weights:", self.patch_weights)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        idx = np.random.choice(np.arange(start=0, stop=len(self.patch_weights)), p=self.patch_weights)
        inp_raw_data, target = self.inp_targets[idx]
        raw_dict = inp_raw_data.__getitem__(0)
        inp = raw_dict['inp']  # zyx
        random_offset = raw_dict["offset"] - self.label_offset  # xyz

        min_indices = random_offset[::-1] - target['min_bound'][::-1]  # computes indices in labeled region
        max_indices = min_indices + self.patch_shape_xyz[::-1]
        label = target['data'][tuple(slice(*i) for i in zip(min_indices,
                                                            max_indices))]
        if self.dim == 2:
            label = label[0]

        inp, label = self.transform(inp.data.numpy(), label)

        sample = {
            'inp': torch.as_tensor(inp),
            'target': torch.as_tensor(label).long(),
            'fname': target['fname']
        }
        if self.label_order is not None:
            sample_target = sample['target'].detach().clone()
            for i, label in enumerate(self.label_order):
                sample['target'][sample_target == i] = label
        return sample

    def __len__(self) -> int:
        return self.epoch_size
