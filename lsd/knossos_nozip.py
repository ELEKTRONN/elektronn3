import knossos_utils
import numpy as np

kd = knossos_utils.KnossosDataset("/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf")

seg_chunk = kd.load_seg(offset=(0,0,0), size=(27119,27350,15494), mag=1, datatype=np.uint8)
