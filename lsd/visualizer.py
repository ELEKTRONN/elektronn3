#import torch
#import numpy as np
#import matplotlib.pyplot as plt
#from new_knossos import KnossosLabelsNozip
#import vigra as v
#import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import ImageGrid
#
#class Visualizer():
#    def __init__(self, conf_path_label, conf_path_raw,
#                model_path,
#                patch_shape=(70, 150, 150),#zyx
#                label_offset = 0,#zyx or 0
#                transform = None, nsamples = 1,
#                device = "cuda", dtype = torch.float):
#        
#        self.conf_path_raw = conf_path_raw
#        self.conf_path_label = conf_path_label
#        self.model_path = model_path
#        self.patch_shape = patch_shape#zyx
#        self.label_offset = 0
#        self.transform = transform
#        self.loader = KnossosLabelsNozip(
#            conf_path_label = self.conf_path_label,
#            conf_path_raw_data = self.conf_path_raw,
#            #label_offset = self.label_offset,
#            patch_shape=self.patch_shape,transform=self.transform,
#            raw_mode="caching")
import torch
import numpy as np
import matplotlib.pyplot as plt
from new_knossos import KnossosLabelsNozip
import vigra as v
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
import os

class Visualizer():
    def __init__(self, conf_path_label, conf_path_raw,
                model_path,
                patch_shape=(70, 150, 150),#zyx
                label_offset = 0,#zyx or 0
                transform = None, nsamples = 1,
                device = "cuda", dtype = torch.float):
        
        self.conf_path_raw = conf_path_raw
        self.conf_path_label = conf_path_label
        self.model_path = model_path
        self.patch_shape = patch_shape#zyx
        self.label_offset = 0
        self.transform = transform
        self.loader = KnossosLabelsNozip(
            conf_path_label = self.conf_path_label,
            conf_path_raw_data = self.conf_path_raw,
            #label_offset = self.label_offset,
            patch_shape=self.patch_shape,transform=self.transform,
            raw_mode="caching")
        self.samples_list = []
        self.device = device
        self.dtype = dtype
        self.fig_save_path = "plots/" + self.model_path.replace("/","%").replace(".","$") + "/"
        os.mkdir(os.path.join(os.getcwd(),self.fig_save_path))
        self.nsamples = nsamples
        self._generate_sample()
        self._load_model()
        self._make_prediction()

    def _generate_sample(self):
        self.sample = self.loader[0]
        self.inp = torch.unsqueeze(self.sample["inp"],0).to(self.device, dtype = self.dtype)
        self.inp_seg = torch.unsqueeze(self.sample["segmentation"],0).to(self.device, dtype = self.dtype)
        self.target = self.sample["target"].to(self.device, dtype = self.dtype)
        self.coordinate_raw = self.sample["coordinate_raw"]#coordinates given xyz !!! look at KnossosRawData documentation
        self.z_plot_coord = self.inp.shape[2]//2#plot xy-plane in the middle of z-axis of the cube
        self.suptitle_string = "at (z,y,x) = ({},{}:{},{}:{})".format(self.z_plot_coord + self.label_offset+self.coordinate_raw[0],
                        self.coordinate_raw[1] - self.patch_shape[1]//2, self.coordinate_raw[1] + self.patch_shape[1]//2,
                        self.coordinate_raw[0] - self.patch_shape[2]//2, self.coordinate_raw[0] + self.patch_shape[2]//2)#self.coordinate_raw is xyz, but self.patch_shape is zyx!!! 
        self.coord_string = "_zyx__{}__{}-{}__{}-{}".format(self.z_plot_coord + self.label_offset+self.coordinate_raw[0],
                        self.coordinate_raw[1] - self.patch_shape[1]//2, self.coordinate_raw[1] + self.patch_shape[1]//2,
                        self.coordinate_raw[0] - self.patch_shape[2]//2, self.coordinate_raw[0] + self.patch_shape[2]//2)#self.coordinate_raw is xyz, but self.patch_shape is zyx!!!

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

    def plot_vdt(self, filename = "BVDT"):
        
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
        ################BoundaryVectorDistanceTransform:#################
        self.targ_vdt = self.target.cpu().detach().numpy()[:3, self.z_plot_coord,:,:]
        self.targ_vdt = np.transpose(self.targ_vdt, (1,2,0)) #for matplotlib to display an RGB image, put the vdt channels as last axis and use w/x axis at first place, while h/y axis at second place
        self.targ_vdt = self.targ_vdt[:,:,::-1] #rearrange dimension axis of the vdt_target s.t. the colormapping is red(x), green(y), blue(z)
        self.targ_vdt_min = np.amin(self.targ_vdt)
        self.targ_vdt_max = np.amax(self.targ_vdt)
        self.targ_vdt = self._rescale(self.targ_vdt, self.targ_vdt_min, self.targ_vdt_max) #rescale to [0,1] interval

        self.pred_vdt = self.prediction.cpu().detach().numpy()[0,:3, self.z_plot_coord,:,:]
        self.pred_vdt = np.transpose(self.pred_vdt, (1,2,0))
        self.pred_vdt = self.pred_vdt[:,:,::-1]
        self.pred_vdt_min = np.amin(self.pred_vdt)
        self.pred_vdt_max = np.amax(self.pred_vdt)
        self.pred_vdt = self._rescale(self.pred_vdt, self.pred_vdt_min, self.pred_vdt_max)

        fig_vdt, axs = plt.subplots(2,2, figsize=(30,20), sharex = True, sharey = True)
        fig_vdt.suptitle("BVDT at "+self.suptitle_string, size = 20)#test this with different patch size
        axs[0,0].set_title("target, scale min: {:8.4f}, max: {:8.4f}".format(self.targ_vdt_min, self.targ_vdt_max), fontsize = 15)
        axs[0,0].imshow(self.targ_vdt)
        axs[0,0].set_ylabel("y", fontsize = 13)
        
        axs[0,1].set_title("prediction, scale min: {:8.4f}, max: {:8.4f}".format(self.pred_vdt_min, self.pred_vdt_max), fontsize = 15)
        axs[0,1].imshow(self.pred_vdt)
        
        #############Norm of BoundaryVectorDistanceTransform##############
        self.targ_vdt_norm = self.target.cpu().detach().numpy()[3, self.z_plot_coord,:,:]
        self.pred_vdt_norm = self.prediction.cpu().detach().numpy()[0,3, self.z_plot_coord,:,:]
        
        axs[1,0].set_title("norm of vdt target")
        axs[1,0].set_ylabel("y", fontsize=13)
        axs[1,0].set_xlabel("x", fontsize = 13)
        #axs[1,0].set_title("target, scale min: {}, max: {}".format(self.targ_vdt_min, self.targ_vdt_max)))
        targ_vdt_norm = axs[1,0].imshow(self.targ_vdt_norm, cmap = "gray")
        axs[1,1].set_title("norm of vdt prediction")
        axs[1,1].set_xlabel("x", fontsize = 13)
        #axs[1,1].set_title("prediction, scale min: {}, max: {}".format(self.pred_vdt_min, self.pred_vdt_min))
        pred_vdt_norm = axs[1,1].imshow(self.pred_vdt_norm, cmap = "gray")
        
        #fig_vdt.subplots_adjust(bottom=0.85)
        #cbar_ax = fig_vdt.add_axes([0.1, 0.95, 0.7, 0.03])
        #cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat], location = "bottom", orientation = "horizontal")
        #fig_vdt.colorbar(targ_vdt_norm, cax = cax, **kw)
        
        fig_vdt.tight_layout()
        fig_vdt.savefig(self.fig_save_path + filename + self.coord_string + ".png")
    
    def plot_vdt_quiver(self, filename = "quiver_BVDT"):
       
        #for the quiver plots 
        rangex = np.arange(self.patch_shape[2])
        rangey = np.arange(self.patch_shape[1])
        xquiver, yquiver = np.meshgrid(rangex, rangey)
        ################BoundaryVectorDistanceTransform:#################
        self.targ_vdt = self.target.cpu().detach().numpy()[:3, self.z_plot_coord,:,:]
        self.targ_vdt = np.transpose(self.targ_vdt, (1,2,0)) #for matplotlib to display an RGB image, put the vdt channels as last axis and use w/x axis at first place, while h/y axis at second place
        self.targ_vdt = self.targ_vdt[:,:,::-1] #rearrange dimension axis of the vdt_target s.t. the colormapping is red(x), green(y), blue(z)
        self.targ_vdt_quiver = self.targ_vdt[:,:,:-1]#clip the z-component of the vector-field
        self.targ_vdt_min = np.amin(self.targ_vdt)
        self.targ_vdt_max = np.amax(self.targ_vdt)
        self.targ_vdt = self._rescale(self.targ_vdt, self.targ_vdt_min, self.targ_vdt_max) #rescale to [0,1] interval
        self.pred_vdt = self.prediction.cpu().detach().numpy()[0,:3, self.z_plot_coord,:,:]
        self.pred_vdt = np.transpose(self.pred_vdt, (1,2,0))
        self.pred_vdt = self.pred_vdt[:,:,::-1]
        self.pred_vdt_quiver = self.targ_vdt[:,:,:-1]#clip the z-component of the vector-field
        self.pred_vdt_min = np.amin(self.pred_vdt)
        self.pred_vdt_max = np.amax(self.pred_vdt)
        self.pred_vdt = self._rescale(self.pred_vdt, self.pred_vdt_min, self.pred_vdt_max)

        fig_vdt_quiver, axs = plt.subplots(1,2, figsize=(30,20), sharex = True, sharey = True)
        fig_vdt_quiver.suptitle("BVDT with xy-projection at "+self.suptitle_string, size = 20)#test this with different patch size
        axs[0].set_title("target, scale min: {:8.4f}, max: {:8.4f}".format(self.targ_vdt_min, self.targ_vdt_max), fontsize = 15)
        axs[0].imshow(self.targ_vdt)
        axs[0].quiver(rangex, rangey, self.targ_vdt_quiver[:,:,0], self.targ_vdt_quiver[:,:,1])
        axs[0].set_ylabel("y", fontsize = 13)
        
        axs[1].set_title("prediction, scale min: {:8.4f}, max: {:8.4f}".format(self.pred_vdt_min, self.pred_vdt_max), fontsize = 15)
        axs[1].imshow(self.pred_vdt)
        axs[1].quiver(rangex, rangey, self.pred_vdt_quiver[:,:,0], self.targ_vdt_quiver[:,:,1])

        fig_vdt_quiver.savefig(self.fig_save_path + filename + self.coord_string + ".png")

    def plot_vdt_norm(self, filename="norm_BVDT"):
        self.targ_vdt_norm = self.target.cpu().detach().numpy()[3, self.z_plot_coord,:,:]
        self.pred_vdt_norm = self.prediction.cpu().detach().numpy()[0,3, self.z_plot_coord,:,:]
        
        fig= plt.figure(figsize=(30,10))
        fig.suptitle("Norm of Boundary VDT at " + self.suptitle_string, size = 20)
        grid = ImageGrid(fig, 111, nrows_ncols=(1,2),
                        share_all=True,
                        cbar_location ="right",
                        cbar_mode="single",
                        cbar_size = "7%", cbar_pad=0.15)
        grid[0].set_title("norm of vdt target", fontsize = 17)
        grid[0].set_ylabel("y", fontsize=13)
        grid[0].set_xlabel("x", fontsize = 13)
        
        targ_vdt_norm = grid[0].imshow(self.targ_vdt_norm, cmap = "gray")
        grid[1].set_title("norm of vdt prediction", fontsize = 17)
        grid[1].set_xlabel("x", fontsize = 13)
        pred_vdt_norm = grid[1].imshow(self.pred_vdt_norm, cmap = "gray")
        
        grid[1].cax.colorbar(pred_vdt_norm)
        grid[1].cax.toggle_label(True)
        fig.savefig(self.fig_save_path + filename + self.coord_string + ".png")


    def plot_gauss_div(self, filename="gauss_div"):
        self.targ_gauss_norm = self.target.cpu().detach().numpy()[4, self.z_plot_coord,:,:]
        self.pred_gauss_norm = self.prediction.cpu().detach().numpy()[0,4, self.z_plot_coord,:,:]
        
        fig= plt.figure(figsize=(30,10))
        fig.suptitle("Gaussian Divergence at " + self.suptitle_string, size = 20)
        grid = ImageGrid(fig, 111, nrows_ncols=(1,2),
                        share_all=True,
                        cbar_location ="right",
                        cbar_mode="single",
                        cbar_size = "7%", cbar_pad=0.15)
        grid[0].set_title("target", fontsize = 17)
        grid[0].set_ylabel("y", fontsize=13)
        grid[0].set_xlabel("x", fontsize = 13)
        
        targ_vdt_norm = grid[0].imshow(self.targ_gauss_norm, cmap = "gray")
        grid[1].set_title("prediction", fontsize = 17)
        grid[1].set_xlabel("x", fontsize = 13)
        pred_vdt_norm = grid[1].imshow(self.pred_gauss_norm, cmap = "gray")
        
        grid[1].cax.colorbar(pred_vdt_norm)
        grid[1].cax.toggle_label(True)
        fig.savefig(self.fig_save_path + filename + self.coord_string + ".png")


    def plot_com(self, filename = "com"):
        
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
        ################CenterOfMass:#################
        self.targ_com = self.target.cpu().detach().numpy()[5:8, self.z_plot_coord,:,:]
        self.targ_com = np.transpose(self.targ_com, (1,2,0)) #for matplotlib to display an RGB image, put the com channels as last axis and use w/x axis at first place, while h/y axis at second place
        self.targ_com = self.targ_com[:,:,::-1] #rearrange dimension axis of the com_target s.t. the colormapping is red(x), green(y), blue(z)
        self.targ_com_min = np.amin(self.targ_com)
        self.targ_com_max = np.amax(self.targ_com)
        self.targ_com = self._rescale(self.targ_com, self.targ_com_min, self.targ_com_max) #rescale to [0,1] interval

        self.pred_com = self.prediction.cpu().detach().numpy()[0,5:8, self.z_plot_coord,:,:]
        self.pred_com = np.transpose(self.pred_com, (1,2,0))
        self.pred_com = self.pred_com[:,:,::-1]
        self.pred_com_min = np.amin(self.pred_com)
        self.pred_com_max = np.amax(self.pred_com)
        self.pred_com = self._rescale(self.pred_com, self.pred_com_min, self.pred_com_max)

        fig_com, axs = plt.subplots(2,2, figsize=(30,20), sharex = True, sharey = True)
        fig_com.suptitle("COM at "+self.suptitle_string, size = 20)#test this with different patch size
        axs[0,0].set_title("target, scale min: {:8.4f}, max: {:8.4f}".format(self.targ_com_min, self.targ_com_max), fontsize = 15)
        axs[0,0].imshow(self.targ_com)
        axs[0,0].set_ylabel("y", fontsize = 13)
        
        axs[0,1].set_title("prediction, scale min: {:8.4f}, max: {:8.4f}".format(self.pred_com_min, self.pred_com_max), fontsize = 15)
        axs[0,1].imshow(self.pred_com)
        
        #############Norm of Center of Mass##############
        self.targ_com_norm = np.linalg.norm(self.targ_com,axis=2)
        self.pred_com_norm = np.linalg.norm(self.pred_com, axis=2)
        
        axs[1,0].set_title("norm of com target")
        axs[1,0].set_ylabel("y", fontsize=13)
        axs[1,0].set_xlabel("x", fontsize = 13)
        targ_com_norm = axs[1,0].imshow(self.targ_com_norm, cmap = "gray")
        axs[1,1].set_title("norm of com prediction")
        axs[1,1].set_xlabel("x", fontsize = 13)
        pred_com_norm = axs[1,1].imshow(self.pred_com_norm, cmap = "gray")
        
        #fig_com.subplots_adjust(bottom=0.85)
        #cbar_ax = fig_com.add_axes([0.1, 0.95, 0.7, 0.03])
        #cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat], location = "bottom", orientation = "horizontal")
        #fig_com.colorbar(targ_com_norm, cax = cax, **kw)
        
        fig_com.tight_layout()
        fig_com.savefig(self.fig_save_path  + filename + self.coord_string + ".png")
        #print("input shape: {}".format(self.inp.shape)) #(bs=1,c,d/z,h/y,w/x)


    def plot_raw(self, filename="raw"):
        
        inp_slice = self.inp.cpu().detach().numpy()[0,0, self.z_plot_coord,:,:]
        inp_slice_seg = self.inp_seg.cpu().detach().numpy()[0,self.z_plot_coord,:,:]
        
        fig, axs = plt.subplots(1,2,figsize=(30,20))
        plt.title("Raw at "+ self.coord_string, fontsize = 20)
        raw_plot = axs[0].imshow(inp_slice, cmap = "gray")
        axs[1].imshow(inp_slice, cmap = "gray")
        axs[1].imshow(inp_slice_seg, cmap = "jet", interpolation = "none", alpha = 0.7)
        
        fig.savefig(self.fig_save_path + filename + self.coord_string + ".png")

    def plot_all(self, filename = "all"):
        
        fig, axs = plt.subplots(4,2,figsize=(30,80))

        fig.suptitle("Model visualization at"+self.suptitle_string, size = 20)#test this with different patch size

        ################BoundaryVectorDistanceTransform:#################
        self.targ_vdt = self.target.cpu().detach().numpy()[:3, self.z_plot_coord,:,:]
        self.targ_vdt = np.transpose(self.targ_vdt, (1,2,0)) #for matplotlib to display an RGB image, put the vdt channels as last axis and use w/x axis at first place, while h/y axis at second place
        self.targ_vdt = self.targ_vdt[:,:,::-1] #rearrange dimension axis of the vdt_target s.t. the colormapping is red(x), green(y), blue(z)
        self.targ_vdt_min = np.amin(self.targ_vdt)
        self.targ_vdt_max = np.amax(self.targ_vdt)
        self.targ_vdt = self._rescale(self.targ_vdt, self.targ_vdt_min, self.targ_vdt_max) #rescale to [0,1] interval

        self.pred_vdt = self.prediction.cpu().detach().numpy()[0,:3, self.z_plot_coord,:,:]
        self.pred_vdt = np.transpose(self.pred_vdt, (1,2,0))
        self.pred_vdt = self.pred_vdt[:,:,::-1]
        self.pred_vdt_min = np.amin(self.pred_vdt)
        self.pred_vdt_max = np.amax(self.pred_vdt)
        self.pred_vdt = self._rescale(self.pred_vdt, self.pred_vdt_min, self.pred_vdt_max)

        axs[0,0].set_title("vector distance trafo target, scale min: {:8.4f}, max: {:8.4f}".format(self.targ_vdt_min, self.targ_vdt_max), fontsize = 15)
        axs[0,0].imshow(self.targ_vdt)
        axs[0,0].set_ylabel("y", fontsize = 13)
        
        axs[0,1].set_title("vector distance trafo prediction, scale min: {:8.4f}, max: {:8.4f}".format(self.pred_vdt_min, self.pred_vdt_max), fontsize = 15)
        axs[0,1].imshow(self.pred_vdt)
        
        #############Norm of BoundaryVectorDistanceTransform##############
        self.targ_vdt_norm = self.target.cpu().detach().numpy()[3, self.z_plot_coord,:,:]
        self.pred_vdt_norm = self.prediction.cpu().detach().numpy()[0,3, self.z_plot_coord,:,:]
        
        axs[1,0].set_title("norm of vdt target", fontsize = 15)
        axs[1,0].set_ylabel("y", fontsize=13)
        axs[1,0].set_xlabel("x", fontsize = 13)
        #axs[1,0].set_title("target, scale min: {}, max: {}".format(self.targ_vdt_min, self.targ_vdt_max)))
        targ_vdt_norm = axs[1,0].imshow(self.targ_vdt_norm, cmap = "gray")
        axs[1,1].set_title("norm of vdt prediction", fontsize = 15)
        axs[1,1].set_xlabel("x", fontsize = 13)
        #axs[1,1].set_title("prediction, scale min: {}, max: {}".format(self.pred_vdt_min, self.pred_vdt_min))
        pred_vdt_norm = axs[1,1].imshow(self.pred_vdt_norm, cmap = "gray")


        ############################GaussianDivergence###################
        self.targ_gauss_norm = self.target.cpu().detach().numpy()[4, self.z_plot_coord,:,:]
        self.pred_gauss_norm = self.prediction.cpu().detach().numpy()[0,4, self.z_plot_coord,:,:]


        axs[2,0].set_title("gaussian divergence target", fontsize = 15)
        axs[2,0].set_ylabel("y", fontsize=13)
        axs[2,0].set_xlabel("x", fontsize = 13)
        
        targ_vdt_norm = axs[2,0].imshow(self.targ_gauss_norm, cmap = "gray")
        axs[2,1].set_title("gaussian divergence prediction", fontsize = 15)
        axs[2,1].set_xlabel("x", fontsize = 13)
        pred_vdt_norm = axs[2,1].imshow(self.pred_gauss_norm, cmap = "gray")
        

        ############################RawPlot##############################
        inp_slice = self.inp.cpu().detach().numpy()[0,0, self.z_plot_coord,:,:]
        inp_slice_seg = self.inp_seg.cpu().detach().numpy()[0, self.z_plot_coord,:,:]
        
        raw_plot = axs[3,0].imshow(inp_slice, cmap = "gray")
        axs[3,0].set_title("raw input", fontsize = 15)
        axs[3,1].imshow(inp_slice, cmap = "gray")
        axs[3,1].set_title("raw input segmentation", fontsize = 15)
        axs[3,1].imshow(inp_slice_seg, cmap = "jet", interpolation = "none", alpha = 0.7)
        
        plt.tight_layout() 

        fig.savefig(os.path.join(self.fig_save_path,filename + self.coord_string + ".png"))
