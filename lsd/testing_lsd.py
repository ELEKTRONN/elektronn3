import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import ndimage as im

one = np.array([[0,0,0,0],
                [0,0,1,1],
                [0,0,1,1],
                [1,0,0,0]])


two = np.array([[1,1,0,0],
                [0,1,1,0],
                [0,1,1,0],
                [1,0,0,0]])


three = np.array([[0,0,0,0],
                [1,0,0,1],
                [1,0,0,1],
                [1,0,0,0]])

#################################################################
#### Centers of Mass ############################################

array = np.stack((one, two, three), axis=0)
print("Input Array shape: {}".format(array.shape))

labels = im.label(array)[0]
print("labels: {}".format(labels))
print(np.nonzero(np.unique(labels)))

com = np.array(im.measurements.center_of_mass(array, labels,np.unique(labels)[1:]))
print("Center of masses (type {}) {}".format(type(com),com))

#################################################################
#################################################################
