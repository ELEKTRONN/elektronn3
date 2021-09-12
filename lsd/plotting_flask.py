from new_knossos import KnossosLabelsNozip
import io
import matplotlib.pyplot as plt


conf_path_raw = "/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf"
ceg_path = "/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/"

def do_plot():
    conf_path_labels = "/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/j0251_72_seg_20210127_agglo2.pyk.conf"
    loader = KnossosLabelsNozip(conf_path_label = conf_path_labels, conf_path_raw_data = conf_path_raw, patch_shape=(50,50,50))
    
    data = loader[0]
    inp = data["inp"]
    targ = data["target"]
    
    inp_slice = inp[0,25,:,:]
    targ_slice = targ[25,:,:]
    
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subfigure(2,1, figsize = (23,10))
    
    axs[0].imshow(inp_slice)
    axs[1].imshow(targ_slice)
    
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    return bytes_image
