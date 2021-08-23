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
print("Center of masses type: {}".format(type(com)))
print("Centers of mass: \n{}".format(com))


#Calculate the vector field that has at each position where there is labelling
#the distance of the particular label to the center of mass of the object that the label belongs to
#coords has a 3d vector field [dimensions 1,2,3] and along dimension 0 it has the coordinates of the point in the field (\vector{x} vector field)
#the com_lsd is the same field as coords but instead of containing the coordinates of a point in the field it contains the position vector of the center of mass where there are labelled objects
#their difference gives the vector field of distances to the center of mass.
shape = array.shape
coords = np.mgrid[:shape[0], :shape[1], :shape[2]]
coords[:, array==0]=0
com_lsd = np.copy(coords).astype(float)
for i in np.unique(labels)[1:]:
    lsd[:, labels==i] = np.tile(com[i-1].reshape(-1,1), lsd[:, labels==i].shape[1])

#################################################################
#################################################################
