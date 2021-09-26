import torch
import numpy as np
import matplotlib.pyplot as plt
from new_knossos import KnossosLabelsNozip
import vigra as v

class Visualizer():
    def __init__(self, conf_path_raw, conf_path_labels,
                model_path,
                patch_shape=(70, 150, 150),#zyx
                label_offset = 0,#zyx or 0
                transform = None, nsamples = 1,
                device = "cuda", dtype = torch.float):
        
        self.conf_path_raw = conf_path_raw
        self.conf_path_label = conf_path_labels
        self.model_path = model_path
        self.patch_shape = patch_shape#zyx
        self.label_offset = 0
        self.transform = transform
        self.loader = KnossosLabelsNozip(
            conf_path_label = self.conf_path_label,
            conf_path_raw_data = self.conf_path_raw,
            label_offset = self.label_offset,
            patch_shape=self.patch_shape,transform=self.transform,
            raw_mode="caching")
        self.samples_list = []
        self.device = device
        self.dtype = dtype
        self.nsamples = nsamples
        self._generate_sample()
        self._load_model()
        self._make_prediction()

    def _generate_sample(self):
        self.sample = self.loader[0]
        self.inp = torch.unsqueeze(self.sample["inp"],0).to(self.device, dtype = self.dtype)
        self.target = self.sample["target"].to(self.device, dtype = self.dtype)
        self.coordinate_raw = self.sample["coordinate_raw"]

    def _load_model(self):
        self.model = torch.load(self.model_path)
        self.model.eval()
        self.input_channels = self.model.in_channels
        self.output_channels = self.model.out_channels
        self.model_dim = self.model.dim

    def _make_prediction(self):
        self.prediction = self.model(self.inp)
    
    def _rescale(self, array, minimum, maximum):
        target = (array-minimum)/(maximum-minimum)
        return target

    def plot(self):
        
        """LSD output dimensions:
            Generates LSD for a segmented dataset with 8 channels
            concatenation of (vdt_target (3), vdt_norm_target(1),
            gauss_target(1), com_lsd(3))"""

        #for each of the local shape descriptors a different representation is
        #needed depending on the kind of shape descriptor
        
        #general 3d data color coding:
        #red:x, green:y, blue:z

        #print("input shape: {}".format(self.inp.shape)) #(bs=1,c,d/z,h/y,w/x)
        #print("target shape: {}".format(self.target.shape)) #(c,d/z,h/y,w/x)
        #print("prediction shape: {}".format(self.prediction.shape)) #(bs=1,c,d/z,h/y, w/x)
       
        #plot a slice from the xy-plane in the middle of the z-axis
        z_plot_coord = self.inp.shape[2]//2 
        
        ################VectorDistanceTransform:#################
        self.targ_vdt = self.target.cpu().detach().numpy()[:3, z_plot_coord,:,:]
        self.targ_vdt = np.transpose(self.targ_vdt, (2,1,0)) #for matplotlib to display an RGB image, put the vdt channels as last axis and use w/x axis at first place, while h/y axis at second place
        self.targ_vdt = self.targ_vdt[:,:,::-1] #rearrange dimension axis of the vdt_target s.t. the colormapping is red(x), green(y), blue(z)
        #self.targ_vdt = self._rescale(self.targ_vdt, np.amin(self.targ_vdt), np.amax(self.targ_vdt)) #rescale to [0,1] interval

        self.pred_vdt = self.prediction.cpu().detach().numpy()[0,:3, z_plot_coord,:,:]
        self.pred_vdt = np.transpose(self.pred_vdt, (2,1,0))
        self.pred_vdt = self.pred_vdt[:,:,::-1]
        #self.pred_vdt = self._rescale(self.pred_vdt, np.amin(self.targ_vdt), np.amax(self.targ_vdt))

        fig, axs = plt.subplots(1,2, figsize=(14,6))
        fig.suptitle("Boundary VDT at (z,y,x) = ({},{}:{},{}:{})".format(z_plot_coord + self.label_offset+self.coordinate_raw[0],
                        self.coordinate_raw[1] - self.patch_shape[1]//2, self.coordinate_raw[1] + self.patch_shape[1]//2,
                        self.coordinate_raw[2] - self.patch_shape[2]//2, self.coordinate_raw[2] + self.patch_shape[2]//2))#test this with different patch size
        axs[0].set_title("target")
        axs[0].imshow(self.targ_vdt)
        
        axs[1].set_title("prediction")
        axs[1].imshow(self.pred_vdt)
        
        plt.savefig("plots/vdt.png")
