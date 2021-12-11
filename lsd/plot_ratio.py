
from visualizer import Visualizer
from new_knossos import KnossosLabelsNozip
from lsd import LSDGaussVdtCom 
from elektronn3.data import transforms
from visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
conf_path_raw = "/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf"

conf_path_labels = "/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/knossos.pyk.conf"


np.random.seed(1)
loader = KnossosLabelsNozip(conf_path_label = conf_path_labels, conf_path_raw_data =
    conf_path_raw, patch_shape=(44,88,88), raw_mode="caching", threshold_background_fraction = 0, raw_cache_reuses = 1)

#nplots = (50,30)
#fig, axs = plt.subplots(nplots[0], nplots[1], figsize=(nplots[0]*10, nplots[1]*10))
#ratios_list = []
#for count, ax in enumerate(axs.flat):
#    data =loader[0]
#    sample = data["inp"].detach().cpu().numpy()
#    ax.imshow(sample[0,0], cmap="gray")
#    nonzero_ratio =np.count_nonzero(sample)/np.prod(sample.shape)
#    ratios_list.append(nonzero_ratio)
#    ax.set_title("ratio of nonzero to all: {}".format(nonzero_ratio))
    
#plt.tight_layout()
#plt.savefig("plots/plots_ratio_nonzero.png") 
num_samples = 1000
    
ratios_list = []
coords_xyz_list = []
for i in range(num_samples):
    data =loader[0]
    sample = data["inp"].detach().cpu().numpy()
    offset_xyz = data["coordinate_raw_xyz"]
    coords_xyz_list.append(offset_xyz)
    nonzero_ratio =np.count_nonzero(sample)/np.prod(sample.shape)
    ratios_list.append(nonzero_ratio)

plt.figure()
plt.plot(ratios_list)
plt.title("distribution of ratios over samples")
plt.savefig("plots/plots_ratio_distribution.png")

plt.figure()
plt.hist(ratios_list, bins=33)
plt.title("histogram of ratios over samples")
plt.xlabel("ratio of volume_nonzero/volume")
plt.ylabel("frequency")
plt.savefig("plots/plots_ratio_hist.png")


###################################
##### look how unique the values are ######
unique_coords, unique_counts = np.unique(coords_xyz_list, axis=0,return_counts=True)
print("number of samples: {}".format(num_samples))
print("number of unique samples: {}".format(len(unique_coords)))
for coord, count in zip(unique_coords,unique_counts):
    print("coordinate (xyz) {} occurred {} times".format(coord, count))
