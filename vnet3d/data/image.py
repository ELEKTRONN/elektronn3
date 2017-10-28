# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip


__all__ = ['downsample_xy', 'ids2barriers', 'smearbarriers',
           'center_cubes', ]


import multiprocessing
from functools import reduce

import numba
from scipy import ndimage
import scipy.ndimage.filters as filters
from skimage.morphology import watershed
import numpy as np
from . import utils


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
    new_d = np.zeros(sh, dtype=np.float32)

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
                            
        barriers_s = filters.convolve(barriers.astype(np.float32),
                                      kernel.astype(np.float32))
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
    barriers = barriers.astype(np.float32)
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
