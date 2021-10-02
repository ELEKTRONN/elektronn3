from typing import Sequence, Tuple, Optional, Dict, Any, Callable, Union

import warnings
import numpy as np
import skimage.exposure
import skimage.transform

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import distance_transform_edt


from elektronn3.data.transforms import random_blurring
from elektronn3.data.transforms.random import Normal, HalfNormal, RandInt

import vigra as v
from scipy import ndimage as im


Transform = Callable[
    [np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, Optional[np.ndarray]]
]



class LSDGaussVdtCom:
    
    """Generates LSD for a segmented dataset with 8 channels
        concatenation of (vdt_target (3), vdt_norm_target(1),
        gauss_target(1), com_lsd(3))"""

    def __init__(
            self,
            #func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ):
        #self.func = func
        self.vdtTransformer = v.filters.boundaryVectorDistanceTransform
        self.gaussDiv = v.filters.gaussianDivergence
        self.labeller = im.label

    def __call__(
            self,
            inp: np.ndarray,
            target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        #use np.expand_dims(target) to fix correct axistags for safe assignment of vigra axistags
        vtarget = v.VigraArray(np.expand_dims(target, axis=0), axistags = v.defaultAxistags('czyx'))
        #vector distance transform and norm
        vdt_target = self.vdtTransformer(vtarget)
        vdt_norm_target = np.expand_dims(np.linalg.norm(vdt_target, axis=0), axis=0)

        gauss_target = self.gaussDiv(vdt_target/vdt_norm_target)#gaussian divergence of normalized VDT

        #discard center of mass lsd because inaccurate to implement
        #center of mass transform
        """vtarget_seg = np.zeros_like(vtarget)
        vtarget_seg[vtarget>0]=1
        labels = self.labeller(vtarget_seg)[0]
        #print("labels: {}".format(labels))
        #print(np.nonzero(np.unique(labels)))
        
        com = np.array(im.measurements.center_of_mass(vtarget_seg, labels,np.unique(labels)[1:]))
        #print("Center of masses type: {}".format(type(com)))
        #print("Centers of mass: \n{}".format(com))
        
        shape = vtarget.shape
        coords = np.mgrid[:shape[1], :shape[2], :shape[3]]
        coords[:, (vtarget==0)[0]]=0
        com_lsd = np.copy(coords).astype(float)
        for i in np.unique(labels)[1:]:
            size_each_label = com_lsd[:,(labels==i)[0]].shape[1]
            com_lsd[:,(labels==i)[0]] -= (np.tile(com[i-1][1:], (size_each_label, 1))).T        

        #for i in np.unique(labels)[1:]:
        #            com_lsd[:, (labels==i)[0]] = np.tile(com[i-1].reshape(-1,1), com_lsd[:, (labels==i)[0]].shape[1])[0].T
        """
        #now stack everything on top along 0th axis to form the 5 (8)D LSD
        #3 for vdt_target, 1 for vdt_norm_target, 1 for gauss_target,( 3 for com_lsd, but not anymore)
        output = np.vstack((vdt_target, vdt_norm_target, gauss_target))
        return (inp, output)


