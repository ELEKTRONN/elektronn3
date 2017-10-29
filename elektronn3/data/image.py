# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip
__all__ = ['downsample_xy', 'ids2barriers', 'smearbarriers',
           'center_cubes', ]
from functools import reduce
import numba
import scipy.ndimage.filters as filters
import numpy as np
from . import utils
from scipy.misc import imsave
from PIL import Image
from .. import floatX


def downsample_xy(d, l, factor):
    """
    Downsample by averaging
    :param d: data
    :param l: label
    :param factor:
    :return:
    """
    f     = int(factor)
    l_sh  = l.shape
    cut   = np.mod(l_sh, f)

    d     = d[:, :, :l_sh[-2]-cut[-2], :l_sh[-1]-cut[-1]]
    sh    = d[:, :, ::f, ::f].shape
    new_d = np.zeros(sh, dtype=floatX)

    l     = l[:, :, l_sh[-2]-cut[-2], :l_sh[-1]-cut[-1]]
    sh    = l[:, :, :f, ::f].shape
    new_l = np.zeros(sh, dtype=l.dtype)

    for i in range(f):
        for j in range(f):
            new_d += d[:, :, i::f, j::f]
            new_l += l[:,    i::f, j::f]

    d = new_d / f**2
    l = new_l / f**2

    return d, l


@utils.timeit
@numba.jit(nopython=True)
def _ids2barriers(ids, barriers, dilute, connectivity):
    """
    Draw a 2 or 4 pix barrier where label IDs are different

    :param ids:  (x,y,z)
    :param barriers:
    :param dilute: e.g. [False, True, True]
    :param connectivity: e.g. [True, True, True]
    :return:
    """
    nx = ids.shape[0]
    ny = ids.shape[1]
    nz = ids.shape[2]

    for x in np.arange(nx-1):
        for y in np.arange(ny-1):
            for z in np.arange(nz-1):
                if connectivity[0]:
                    if ids[x,y,z]!=ids[x+1,y,z]:
                        barriers[x,y,z]   = 1
                        barriers[x+1,y,z] = 1
                        if dilute[0]:
                            if x>0:    barriers[x-1,y,z] = 1
                            if x<nx-2: barriers[x+2,y,z] = 1

                if connectivity[1]:
                    if ids[x,y,z]!=ids[x,y+1,z]:
                        barriers[x,y,z]   = 1
                        barriers[x,y+1,z] = 1
                        if dilute[1]:
                            if y>0:    barriers[x,y-1,z] = 1
                            if y<ny-2: barriers[x,y+2,z] = 1

                if connectivity[2]:
                    if ids[x,y,z]!=ids[x,y,z+1]:
                        barriers[x,y,z]   = 1
                        barriers[x,y,z+1] = 1
                        if dilute[2]:
                            if z>0:    barriers[x,y,z-1] = 1
                            if z<nz-2: barriers[x,y,z+2] = 1


def ids2barriers(ids, dilute=[True,True, True],
                 connectivity=[True, True, True],
                 ecs_as_barr=True,
                 smoothen=False):
    dilute = np.array(dilute)
    connectivity = np.array(connectivity)
    barriers = np.zeros_like(ids, dtype=np.int16)

    _ids2barriers(ids, barriers, dilute, connectivity)
    _ids2barriers(ids[::-1,::-1,::-1],
                  barriers[::-1,::-1,::-1],
                  dilute, connectivity) # apply backwards as lazy hack to fix boundary
                  
    if smoothen:
        kernel = np.array([[[0.1, 0.2, 0.1],
                            [0.2, 0.3, 0.2],
                            [0.1, 0.2, 0.1]],

                           [[0.3, 0.5, 0.3],
                            [0.5, 1.0, 0.5],
                            [0.3, 0.5, 0.3]],

                           [[0.1, 0.2, 0.1],
                            [0.2, 0.3, 0.2],
                            [0.1, 0.2, 0.1]]])
                            
        barriers_s = filters.convolve(barriers.astype(floatX),
                                      kernel.astype(floatX))
        barriers = (barriers_s>4).astype(np.int16) # (old - new).mean() ~ 0

    if ecs_as_barr=='new_class':
        ecs  = np.logical_and( (ids==0), (barriers!=1))
        barriers[ecs] = 2
    
    elif ecs_as_barr:
        ecs  = (ids==0).astype(np.int16)
        barriers = np.maximum(ecs, barriers)

    return barriers


def blob(sizes):
    """
    Return Gaussian blob filter
    """
    grids = [np.linspace(-2.2,2.2,size) for size in sizes]
    grids = np.meshgrid(*grids, indexing='ij')
    ret = np.exp(-0.5*(reduce(np.add, list(map(np.square, grids)))))
    ret = ret / np.square(ret).sum()
    return ret


def _smearbarriers(barriers, kernel):
    # Note: this is good but makes holes to small,
    #  besides we must raise/lower all confidences in GT
    barriers = barriers.astype(floatX)
    if kernel is None:
        kernel = np.array([
            [[ 0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.1,  0.,  0.],
             [ 0.,  0.1,  0.2,  0.1,  0.],
             [ 0.,  0.,  0.1,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.]],

            [[ 0.,  0.,  0.1,  0.,  0.],
             [ 0.,  0.2,  0.4,  0.2,  0.],
             [ 0.1,  0.4,  1.,  0.4,  0.1],
             [ 0.,  0.2,  0.4,  0.2,  0.],
             [ 0.,  0.,  0.1,  0.,  0.]],

            [[ 0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.1,  0.,  0.],
             [ 0.,  0.1,  0.2,  0.1,  0.],
             [ 0.,  0.,  0.1,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.]],
        ]).T
    else:
        sizes = kernel
        kernel = blob(sizes)
        index = np.subtract(sizes, 1)
        index = np.divide(index, 2)
        kernel[tuple(index)] = 1.0 # set center to 1


    barriers = filters.convolve(barriers, kernel)
    barriers = np.minimum(barriers, 1.0)
    return barriers


def smearbarriers(barriers, kernel=None):
    """
    barriers: 3d volume (z,x,y)
    """
    pos = _smearbarriers(barriers, kernel)
    neg = 1.0 - _smearbarriers(1.0 - barriers, kernel)
    barriers = 0.5 * (pos + neg)
    #barriers = np.minimum(barriers, 1.0)
    return barriers


@numba.jit(nopython=True)
def _grow_seg(seg, grow, mask):
    nx = seg.shape[0]
    ny = seg.shape[1]
    nz = seg.shape[2]
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            for z in range(1,nz-1):

                if mask[0] and (seg[x,y,z]!=0) and (seg[x-1,y,z]==0):
                    grow[x-1,y,z]   = seg[x,y,z]
                if mask[0] and (seg[x,y,z]!=0) and (seg[x+1,y,z]==0):
                    grow[x+1,y,z]   = seg[x,y,z]

                if mask[1] and (seg[x,y,z]!=0) and (seg[x,y-1,z]==0):
                    grow[x,y-1,z]   = seg[x,y,z]
                if mask[1] and (seg[x,y,z]!=0) and (seg[x,y+1,z]==0):
                    grow[x,y+1,z]   = seg[x,y,z]

                if mask[2] and (seg[x,y,z]!=0) and (seg[x,y,z-1]==0):
                    grow[x,y,z-1]   = seg[x,y,z]
                if mask[2] and (seg[x,y,z]!=0) and (seg[x,y,z+1]==0):
                    grow[x,y,z+1]   = seg[x,y,z]


def grow_seg(seg, pixel=[1,3,3]):
    """
    Grow segmentation labels into ECS/background by n pixel
    """
    if isinstance(pixel, (list, tuple, np.ndarray)):
        n = np.max(pixel)
    else:
        n = pixel
        pixel = [n,] * 3

    if n==0:
        return seg

    grow = seg.copy()
    for i in range(n):
        mask = np.greater(pixel, 0)
        _grow_seg(seg, grow, mask)
        seg = grow.copy()
        pixel = np.subtract(pixel, 1)

    return seg


def center_cubes(cube1, cube2, crop=True):
    """
    shapes (ch,x,y,z) or (x,y,z)
    """
    is_3d = [False, False]
    if cube1.ndim==3:
        cube1 = cube1[None]
        is_3d[0] = True
    if cube2.ndim==3:
        cube2 = cube2[None]
        is_3d[1] = True

    diffs = np.subtract(cube1.shape, cube2.shape)[1:]
    assert np.all(diffs%2==0)
    diffs //= 2

    slices1 = [slice(None)]
    pad1    = [(0,0)]
    slices2 = [slice(None)]
    pad2    = [(0,0)]
    for d in diffs:
        if d>0: # 1 is larger than 2
            if crop:
                slices1.append(slice(d, -d))
                pad1.append((0,0))

                slices2.append(slice(None))
                pad2.append((0,0))
            else:
                slices1.append(slice(None))
                pad1.append((0,0))

                slices2.append(slice(None))
                pad2.append((d,d))
        elif d<0:
            if crop:
                slices2.append(slice(-d, d))
                pad2.append((0,0))

                slices1.append(slice(None))
                pad1.append((0,0))
            else:
                slices2.append(slice(None))
                pad2.append((0,0))

                slices1.append(slice(None))
                pad1.append((-d, -d))
        else:
            slices2.append(slice(None))
            pad2.append((0,0))

            slices1.append(slice(None))
            pad1.append((0,0))

    cube1 = cube1[slices1]
    cube2 = cube2[slices2]
    cube1 = np.pad(cube1, pad1, 'constant')
    cube2 = np.pad(cube2, pad2, 'constant')

    if is_3d[0]:
        cube1 = cube1[0]
    if is_3d[1]:
        cube2 = cube2[0]

    return cube1, cube2


def write_overlayimg(dest_path, raw, pred, fname, nb_of_slices, thresh=0.1):
    if thresh is not None:
        pred = (pred > thresh).astype(np.uint)
    ixs = np.arange(len(raw))
    np.random.seed(0)
    np.random.shuffle(ixs)
    if nb_of_slices is not None:
        ixs = ixs[:nb_of_slices]
    for i in ixs:
        create_label_overlay_img(pred[i], dest_path + "/%s_%d.png" % (fname, i),
                                 background=raw[i] * 255,
                                 save_raw_img=False)


def create_label_overlay_img(labels, save_path, background=None, cvals=None,
                             save_raw_img=True):
    """
    Adapted from Sven Dorkenwald

    """
    if cvals is None:
        cvals = {}
    else:
        assert isinstance(cvals, dict)

    np.random.seed(111)

    label_prob_dict = {}

    unique_labels = np.unique(labels)
    for unique_label in unique_labels:
        if unique_label == 0:
            continue
        label_prob_dict[unique_label] = (labels == unique_label).astype(np.int)

        if not unique_label in cvals:
            cvals[unique_label] = [np.random.rand() for _ in range(3)] + [1]

    if len(label_prob_dict) == 0:
        print("No labels detected! No overlay image created")
    else:
        create_prob_overlay_img(label_prob_dict, save_path,
                                background=background, cvals=cvals,
                                save_raw_img=save_raw_img)


def create_prob_overlay_img(label_prob_dict, save_path, background=None,
                            cvals=None, save_raw_img=True):
    """
    Adapted from Sven Dorkenwald

    """
    assert isinstance(label_prob_dict, dict)
    if cvals is not None:
        assert isinstance(cvals, dict)

    np.random.seed(0)
    label_prob_dict_keys = list(label_prob_dict.keys())
    sh = label_prob_dict[label_prob_dict_keys[0]].shape[:2]
    imgs = []
    for key in label_prob_dict_keys:
        label_prob = np.array(label_prob_dict[key])

        if label_prob.dtype == np.uint8:
            label_prob = label_prob.astype(np.float) / 255

        label_prob = label_prob.squeeze()

        if key in cvals:
            cval = cvals[key]
        else:
            cval = [np.random.rand() for _ in range(3)] + [1]

        this_img = np.zeros([sh[0], sh[1], 4], dtype=floatX)
        this_img[label_prob > 0] = np.array(cval) * 255
        this_img[:, :, 3] = label_prob * 100
        imgs.append(this_img)

    if background is None:
        background = np.ones(imgs[0].shape)
        background[:, :, 3] = np.ones(sh)
    elif len(np.shape(background)) == 2:
        t_background = np.zeros(imgs[0].shape)
        for ii in range(3):
            t_background[:, :, ii] = background

        t_background[:, :, 3] = np.ones(background.squeeze().shape) * 255
        background = t_background
    elif len(np.shape(background)) == 3:
        background = np.array(background)[:, :, 0]
        background = np.array([background, background, background,
                               np.ones_like(background) * 255])

    if np.max(background) <= 1:
        background *= 255.
    else:
        background = np.array(background, dtype=np.float)

    comp = imgs[0]
    for img in imgs[1:]:
        comp = alpha_composite(comp, img)

    comp = alpha_composite(comp, background)

    if save_path is not None:
        imsave(save_path, comp)

    if save_raw_img and background is not None:
        raw_save_path = "".join(save_path.split(".")[:-1]) + "_raw." + save_path.split(".")[-1]
        imsave(raw_save_path, background)


def alpha_composite(src, dst):
    ''' http://stackoverflow.com/questions/3374878/with-the-python-imaging-library-pil-how-does-one-compose-an-image-with-an-alp/3375291#3375291
    Return the alpha composite of src and dst.
    Sven Dorkenwald
    Parameters:
    src -- PIL RGBA Image object
    dst -- PIL RGBA Image object

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    '''
    # http://stackoverflow.com/a/3375291/190597
    src = np.asarray(src)
    dst = np.asarray(dst)
    out = np.empty(src.shape, dtype = 'float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    src_a = src[alpha]/255.0
    dst_a = dst[alpha]/255.0
    out[alpha] = src_a+dst_a*(1-src_a)
    old_setting = np.seterr(invalid = 'ignore')
    out[rgb] = (src[rgb]*src_a + dst[rgb]*dst_a*(1-src_a))/out[alpha]
    np.seterr(**old_setting)
    out[alpha] *= 255
    np.clip(out,0,255)
    # astype('uint8') maps np.nan (and np.inf) to 0
    out = out.astype('uint8')
    out = Image.fromarray(out, 'RGBA')
    return out
