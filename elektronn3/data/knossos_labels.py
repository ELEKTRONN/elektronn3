import os
import random
from typing import Callable, Dict, Optional, Sequence

import knossos_utils
import numpy as np
import torch
import torch.utils.data

from elektronn3.data import transforms
from elektronn3.data.knossos import KnossosRawData


# todo: write a header + class definition
class KnossosLabels(torch.utils.data.Dataset):
    def __init__(
            self,
            conf_path_label: str,
            conf_path_raw_data: str,
            current_dir_path: str,
            patch_shape: Sequence[int],  # [z]yx
            transform: Callable = transforms.Identity(),
            mag: int = 1,
            epoch_size: int = 100,
    ):
        self.conf_path_label = conf_path_label
        self.conf_path_raw_data = conf_path_raw_data
        self.patch_shape = np.array(patch_shape)
        patch_shape_xyz = self.patch_shape[::-1]  # zyx -> xyz
        # if self.dim == 2: # todo: extend of patch shape with dim = 2
        #    patch_shape_xyz = np.array([*patch_shape_xyz, 1])  # z=1 for 2D
        self.patch_shape_xyz = patch_shape_xyz
        self.transform = transform
        self.mag = mag
        self.epoch_size = epoch_size
        self.kd = knossos_utils.KnossosDataset(self.conf_path_label, show_progress=False)

        self.dir_path = current_dir_path
        if self.dir_path:
            self.kzip_files_path = [f for f in os.listdir(self.dir_path) if f.endswith('.k.zip')]

            self.labelled_files = []
            for file in self.kzip_files_path:
                zip_path = self.dir_path + "/" + file
                file_bounds = self.kd.get_movement_area(zip_path)
                self.labelled_files.append({'file_path': zip_path, 'min': file_bounds[0], 'max': file_bounds[1]})
        else:
            raise OSError(f"Path to conf directory is empty")

        self.targets = []
        self.inp_raw_data = []

        for file in self.labelled_files:
            size = (file['max'][0] - file['min'][0], file['max'][1] - file['min'][1],
                    file['max'][2] - file['min'][2])
            offset = tuple(file['min'])
            data = self.kd._load_kzip_seg(file['file_path'], offset, size,
                                          self.mag, datatype=np.int32,
                                          padding=0,
                                          apply_mergelist=False,
                                          return_dataset_cube_if_nonexistent=False,
                                          expand_area_to_mag=False)
            if data.max() < 4:  # todo: only select files with valid label range
                self.targets.append({'data': data, 'lower_bound': file['min'],
                                     'upper_bound': file['max']})  # zyx form

                self.inp_raw_data.append(KnossosRawData(conf_path=self.conf_path_raw_data,
                                                        patch_shape=self.patch_shape
                                                        , transform=None,
                                                        bounds=(tuple(file['min']), tuple(file['max'])),
                                                        mag=self.mag, mode='disk', epoch_size=self.epoch_size,
                                                        disable_memory_check=False, verbose=False))  # xyz form

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:

        random_idx = random.randrange(len(self.inp_raw_data))

        raw_dict = self.inp_raw_data[random_idx].__getitem__(0)
        inp = raw_dict['inp']  # zyx
        offset = raw_dict["offset"]  # xyz
        label = self.targets[random_idx]['data'][
                offset[2] - self.targets[random_idx]['lower_bound'][2]: offset[2] -
                                                                        self.targets[random_idx]['lower_bound'][2] +
                                                                        self.patch_shape_xyz[2],
                offset[1] - self.targets[random_idx]['lower_bound'][1]: offset[1] -
                                                                        self.targets[random_idx]['lower_bound'][1] +
                                                                        self.patch_shape_xyz[1],
                offset[0] - self.targets[random_idx]['lower_bound'][0]: offset[0] -
                                                                        self.targets[random_idx]['lower_bound'][0] +
                                                                        self.patch_shape_xyz[0]]  # zyx (D, H, W)
        inp, label = self.transform(inp.data.numpy(), label)
        label = label.astype('int32')
        sample = {
            'inp': torch.as_tensor(inp),
            'target': torch.as_tensor(label)
        }
        return sample

    def __len__(self) -> int:
        return self.epoch_size
