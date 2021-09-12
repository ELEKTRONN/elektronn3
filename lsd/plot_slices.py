from new_knossos import KnossosLabelsNozip
import matplotlib.pyplot as plt


conf_path_raw = "/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf"
ceg_path = "/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/"

conf_path_labels = "/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/j0251_72_seg_20210127_agglo2.pyk.conf"
loader = KnossosLabelsNozip(conf_path_label = conf_path_labels, conf_path_raw_data = conf_path_raw, patch_shape=(50,50,50))

data = loader[0]
inp = data["inp"]
targ = data["target"]

inp_slice = inp[0,25,:,:]
targ_slice = targ[25,:,:]

ncols = inp_slice.shape[1]

import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols,2, figsize=(25, ncols*10))

for count,ax in enumerate(axs):
    ax[0].imshow(inp[0,count,:,:])
    ax[0].axis("off")
    ax[1].imshow(targ[count,:,:])
    ax[1].axis("off")

fig.tight_layout()
fig.savefig("testslice.png")
