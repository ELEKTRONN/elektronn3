import numpy as np
from elektronn3.data.knossos_labels import KnossosLabels
import snappy

conf_path_raw = "/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf"
conf_path_label = "/wholebrain/songbird/j0251/groundtruth/j0251.conf"
dir_path_label = "/wholebrain/songbird/j0251/groundtruth/segmentation_gt"

label_names = ('sj', 'vc', 'mitos')
loader = KnossosLabels(conf_path_label = conf_path_label, conf_path_raw_data = conf_path_raw,
        dir_path_label = dir_path_label,
        mag = 1, patch_shape = [100,100,100],
        label_names = label_names)


item = loader[0]

inp = item["inp"]
targ = item["target"]

print("input shape: {}".format(inp.shape))
print("target shape: {}".format(targ.shape))

