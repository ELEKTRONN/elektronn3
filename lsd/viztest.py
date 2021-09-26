from visualizer import Visualizer
from new_knossos import KnossosLabelsNozip
from lsd import LSDGaussVdtCom 
from elektronn3.data import transforms
from visualizer import Visualizer
conf_path_raw = "/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf"

conf_path_labels = "/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/j0251_72_seg_20210127_agglo2.pyk.conf"

local_shape_descriptor = LSDGaussVdtCom()
common_transforms = [
    transforms.Normalize(mean=0, std=255.),
    local_shape_descriptor

]
transform = transforms.Compose(common_transforms + [
])
loader = KnossosLabelsNozip(conf_path_label = conf_path_labels, conf_path_raw_data = conf_path_raw, patch_shape=(44,88,88),transform=transform, raw_mode="caching")

data = loader[0]
model_path = "/wholebrain/scratch/fkies/e3training/lsd/L1_seed0_SGD/model_best.pt"
viz = Visualizer(conf_path_raw, conf_path_raw, model_path, patch_shape = (70, 150, 200), transform=transform)

viz.plot()
