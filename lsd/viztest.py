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
#model_path_old = "/wholebrain/scratch/fkies/e3training/lsd/L1_seed0_SGD/model_best.pt"
model_path_new = "/wholebrain/scratch/fkies/e3training/lsd/L1_seed0_SGD__nocom/model_best.pt"
import numpy as np
#np.random.seed(0)
#viz_old = Visualizer(conf_path_labels, conf_path_raw, model_path_old, patch_shape = (70, 150, 200), transform=transform)
np.random.seed(1)
viz_new = Visualizer(conf_path_labels, conf_path_raw, model_path_new, patch_shape = (70, 150, 200), transform=transform)

#viz_old.plot_vdt("old_BVDT")
#viz_old.plot_vdt_norm("old_norm_BVDT")
#viz_old.plot_gauss_div("old_gauss_div")
#viz_old.plot_com("old_com")
#viz_old.plot_raw("old_raw")


nplots = 5

for count in range(nplots):
    viz_new._generate_sample()
    viz_new._generate_sample()
    viz_new._load_model()
    viz_new._make_prediction()
    #viz_new.plot_vdt(str(count) + "new_BVDT")
    #viz_new.plot_vdt_norm(str(count) + "new_norm_BVDT")
    #viz_new.plot_gauss_div(str(count) +"new_gauss_div")
    ##viz_new.plot_com(str(count) +"new_com")
    #viz_new.plot_raw(str(count) +"new_raw")
    viz_new.plot_all()
