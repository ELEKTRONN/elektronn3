from new_knossos import KnossosLabelsNozip
from lsd import LSDGaussVdtCom 
from elektronn3.data import transforms
conf_path_raw = "/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf"
ceg_path = "/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/"

conf_path_labels = "/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/j0251_72_seg_20210127_agglo2.pyk.conf"

local_shape_descriptor = LSDGaussVdtCom()
common_transforms = [
    transforms.Normalize(mean=0, std=255.),
    local_shape_descriptor

]
train_transform = transforms.Compose(common_transforms + [
])
loader = KnossosLabelsNozip(conf_path_label = conf_path_labels, conf_path_raw_data = conf_path_raw, patch_shape=(44,88,88),transform=train_transform, raw_mode="caching")

data = loader[0]
inp = data["inp"]
targ = data["target"]


print("input shape: {}".format(inp.shape))
print("target shape after transform: {}".format(targ.shape))
#print("target axistags: {}".format(targ.axistags))
