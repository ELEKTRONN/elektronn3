import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import ndimage as im
import vigra as v

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
    com_lsd[:, labels==i] = np.tile(com[i-1].reshape(-1,1), com_lsd[:, labels==i].shape[1])

#################################################################
#################################################################

#################################################################
######## Vigranumpy VectorDistanceTransform, gaussianDivergence##
array = array.reshape((1,)+array.shape)
print("new segmentation like array shape: {}".format(array.shape))
vigra_array = v.VigraArray(array, axistags=v.defaultAxistags('czyx'))
vdt = v.filters.boundaryVectorDistanceTransform(vigra_array.astype(np.uint32))
print("vector distance transform shape: {}".format(vdt.shape))
print("vector distance transform axistags: {}".format(vdt.axistags))

norm_vdt = np.linalg.norm(vdt, axis=0)
print("vector distance transform (normalised) shape: {}".format(norm_vdt.shape))

#gaussian_divergence = v.filters.gaussianDivergence(vigra_array))
#print("gaussian divergence shape: {}".format(gaussian_divergence.shape))
narray= np.random.rand(3,20,20,10)
nvigra_array=v.VigraArray(narray, axistags=v.defaultAxistags("czyx"))
divergence = v.filters.gaussianDivergence(nvigra_array)
print("gaussian divergence shape: {}".format(divergence.shape))

"""#test gaussianDivergence with a View
nvigra_view = v.taggedView(narray.astype(np.float32), axistags=v.defaultAxistags('czyx'))
divview = v.filters.gaussianDivergence(nvigra_view)
print("divview shape: {}".format(divview.shape))
commented because this raises an error. raised an issue on github ukoethe/vigra #498"""


################################################################
###### Test the LSD class that I wrote #########################

from lsd import LSDGaussVdtCom

transformer = LSDGaussVdtCom()
print("Class LSDGaussVdtCom constructed")

#create test data:
testing_inputs = np.random.rand(1,10,20,20)
testing_labels = np.random.choice([0,1], size=(1,10,20,20))
print("""testing data has been created with shape {}""".format(testing_labels.shape))

#apply the lsd_transformer to the data
inputs, lsd_output=transformer(testing_inputs,testing_labels)
print("output has been computed with shape {}".format(lsd_output.shape))
