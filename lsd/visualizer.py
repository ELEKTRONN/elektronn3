import torch
import numpy as np
import matplotlib.pyplot as plt
from new_knossos import KnossosLabelsNozip
import vigra as v

class Visualizer():
    def __init__(self, conf_path_raw, conf_path_labels,
                model_path,
                patch_shape=(70, 150, 150),
                transform = None, nsamples = 1,
                device = "cuda", dtype = torch.float):
        
        self.conf_path_raw = conf_path_raw
        self.conf_path_label = conf_path_labels
        self.model_path = model_path
        self.patch_shape = patch_shape#zyx
        self.transform = transform
        self.loader = KnossosLabelsNozip(
            conf_path_label = self.conf_path_label,
            conf_path_raw_data = self.conf_path_raw,
            patch_shape=self.patch_shape,transform=self.transform,
            raw_mode="caching")
        self.samples_list = []
        self.device = device
        self.dtype = dtype
        self.nsamples = nsamples
        self._generate_sample()
        self._load_model()
        #self._make_prediction()

    def _generate_sample(self):
        self.sample = self.loader[0]
        self.inp = torch.unsqueeze(self.sample["inp"],0).to(self.device, dtype = self.dtype)
        self.target = self.sample["target"].to(self.device, dtype = self.dtype)

    def _load_model(self):
        self.model = torch.load(self.model_path)
        self.model.eval()
        self.input_channels = self.model.in_channels
        self.output_channels = self.model.out_channels
        self.model_dim = self.model.dim

    def _make_prediction(self):
        self.prediction = self.model(self.inp)

    def plot(self):
        pass
