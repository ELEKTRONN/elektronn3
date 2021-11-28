from visualizer import Visualizer
from new_knossos import KnossosLabelsNozip
from lsd import LSDGaussVdtCom 
from elektronn3.data import transforms
from visualizer import Visualizer
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Set density of quiver plot')
parser.add_argument('-s', '--skip',type = int, default = 1, help = 'number of datapoints skipped after each arrow')
args = parser.parse_args()
skip = args.skip

conf_path_raw = "/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf"

conf_path_labels = "/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/knossos.pyk.conf"

local_shape_descriptor = LSDGaussVdtCom()
common_transforms = [
    transforms.Normalize(mean=0, std=255.),
    local_shape_descriptor

]
transform = transforms.Compose(common_transforms + [
])

np.random.seed(1)
loader = KnossosLabelsNozip(conf_path_label = conf_path_labels, conf_path_raw_data = conf_path_raw, patch_shape=(10,300,310),transform=transform, raw_mode="caching")

data = loader[0]
