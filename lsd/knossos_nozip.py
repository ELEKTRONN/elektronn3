import knossos_utils
import numpy as np

kd = knossos_utils.KnossosDataset("/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf")

seg_chunk = kd.load_seg(offset=(0,0,0), size=(128,256,256), mag=1)
