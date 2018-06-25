# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert, Marius Killinger


__all__ = ['warp_slice', 'get_warped_slice', 'WarpingOOBError']

import itertools
from functools import reduce, lru_cache
import numpy as np
import numba
from elektronn3 import floatX
from elektronn3.data.utils import slice_h5

# TODO: A major refactoring is required here:
#  This module should not perform any data I/O itself. Instead it should provide a
#  framework for generating and transforming source coordinates (with
#  support for user-defined transforms, similar to the image transforms pipeline).
#  Code for HDF5 slicing and voxel value interpolation should be in separate modules.


# TODO: Revise this. Especially the clipping is very destructive!
# This is currently broken because it expects floats in the range (0, 1)
# and we can't supply those (except by re-breaking normalization or awkward
# and lossy transforms before and after calling this function).
def grey_augment(d, channels, rng):
    """
    Performs grey value (histogram) augmentations on ``d``. This is only
    applied to ``channels`` (list of channels indices), ``rng`` is a random
    number generator
    """
    raise NotImplementedError
    if channels == []:
        return d
    else:
        k = len(channels)
        d = d.copy()  # d is still just a view, we don't want to change the original data so copy it
        alpha = 1 + (rng.rand(k) - 0.5) * 0.3 # ~ contrast
        c     = (rng.rand(k) - 0.5) * 0.3 # mediates whether values are clipped for shadows or lights
        gamma = 2.0 ** (rng.rand(k) * 2 - 1) # sample from [0.5,2] with mean 0

        d[channels] = d[channels] * alpha[:,None,None] + c[:,None,None]
        d[channels] = np.clip(d[channels], 0, 1)
        d[channels] = d[channels] ** gamma[:,None,None]
    return d


@numba.guvectorize(['void(float32[:,:,:], float32[:], float32[:], float32[:,],)'],
              '(x,y,z),(i),(i)->()', nopython=True)#target='parallel',
def map_coordinates_nearest(src, coords, lo, dest):
    u = np.int32(np.round(coords[0] - lo[0]))
    v = np.int32(np.round(coords[1] - lo[1]))
    w = np.int32(np.round(coords[2] - lo[2]))
    dest[0] = src[u,v,w]


@numba.guvectorize(['void(float32[:,:,:], float32[:], float32[:], float32[:,],)'],
              '(x,y,z),(i),(i)->()', nopython=True)# target='parallel'
def map_coordinates_linear(src, coords, lo, dest):
    u = coords[0] - lo[0]
    v = coords[1] - lo[1]
    w = coords[2] - lo[2]
    u0 = np.int32(u)
    u1 = u0 + 1
    du = u - u0
    v0 = np.int32(v)
    v1 = v0 + 1
    dv = v - v0
    w0 = np.int32(w)
    w1 = w0 + 1
    dw = w - w0
    val = src[u0, v0, w0] * (1-du) * (1-dv) * (1-dw) +\
          src[u1, v0, w0] * du * (1-dv) * (1-dw) +\
          src[u0, v1, w0] * (1-du) * dv * (1-dw) +\
          src[u0, v0, w1] * (1-du) * (1-dv) * dw +\
          src[u1, v0, w1] * du * (1-dv) * dw +\
          src[u0, v1, w1] * (1-du) * dv * dw +\
          src[u1, v1, w0] * du * dv * (1-dw) +\
          src[u1, v1, w1] * du * dv * dw
    dest[0] = val


@lru_cache(maxsize=1)
def identity():
    return np.eye(4, dtype=floatX)


def translate(dz, dy, dx):
    return np.array([
        [1.0, 0.0, 0.0,  dz],
        [0.0, 1.0, 0.0,  dy],
        [0.0, 0.0, 1.0,  dx],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=floatX)


def rotate_z(a):
    return np.array([
        [1.0, 0.0,    0.0,     0.0],
        [0.0, np.cos(a), -np.sin(a), 0.0],
        [0.0, np.sin(a), np.cos(a),  0.0],
        [0.0, 0.0,    0.0,     1.0]
    ], dtype=floatX)


def rotate_y(a):
    return np.array([
        [np.cos(a), -np.sin(a), 0.0, 0.0],
        [np.sin(a),  np.cos(a), 0.0, 0.0],
        [0.0,        0.0, 1.0, 0.0],
        [0.0,        0.0, 0.0, 1.0]
    ], dtype=floatX)


def rotate_x(a):
    return np.array([
        [np.cos(a),  0.0, np.sin(a), 0.0],
        [0.0,     1.0, 0.0,    0.0],
        [-np.sin(a), 0.0, np.cos(a), 0.0],
        [0.0,     0.0, 0.0,    1.0]
    ], dtype=floatX)


def scale_inv(mz, my, mx):
    return np.array([
        [1/mz,  0.0,    0.0,  0.0],
        [0.0,   1/my,   0.0,  0.0],
        [0.0,   0.0,    1/mx, 0.0],
        [0.0,   0.0,    0.0,  1.0]
    ], dtype=floatX)


@lru_cache()
def scale(mz, my, mx):
    return np.array([
        [mz,  0.0, 0.0, 0.0],
        [0.0, my,  0.0, 0.0],
        [0.0, 0.0, mx,  0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=floatX)


def chain_matrices(mat_list):
    return reduce(np.dot, mat_list, identity())


def get_random_rotmat(lock_z=False, amount=1.0, rng=None):
    rng = np.random.RandomState() if rng is None else rng

    gamma = rng.rand() * 2 * np.pi * amount
    if lock_z:
        return rotate_z(gamma)

    phi = rng.rand() * 2 * np.pi * amount
    theta = np.arcsin(rng.rand()) * amount

    R1 = rotate_z(-phi)
    R2 = rotate_y(-theta)
    R3 = rotate_z(gamma)
    R = chain_matrices([R3, R2, R1])
    return R


def get_random_flipmat(no_x_flip=False, rng=None):
    rng = np.random.RandomState() if rng is None else rng
    F = np.eye(4, dtype=floatX)
    flips = rng.binomial(1, 0.5, 4) * 2 - 1
    flips[3] = 1 # don't flip homogeneous dimension
    if no_x_flip:
        flips[2] = 1

    np.fill_diagonal(F, flips)
    return F


def get_random_swapmat(lock_z=False, rng=None):
    rng = np.random.RandomState() if rng is None else rng
    S = np.eye(4, dtype=floatX)
    if lock_z:
        swaps = [[0, 1, 2, 3],
                 [0, 2, 1, 3]]
    else:
        swaps = [[0, 1, 2, 3],
                 [0, 2, 1, 3],
                 [1, 0, 2, 3],
                 [1, 2, 0, 3],
                 [2, 0, 1, 3],
                 [2, 1, 0, 3]]

    i = rng.randint(0, len(swaps))
    S = S[swaps[i]]
    return S


def get_random_warpmat(lock_z=False, perspective=False, amount=1.0, rng=None):
    W = np.eye(4, dtype=floatX)
    amount *= 0.1
    perturb = np.random.uniform(-amount, amount, (4, 4))
    perturb[3,3] = 0
    if lock_z:
        perturb[0] = 0
        perturb[:,0] = 0
    if not perspective:
        perturb[3] = 0

    perturb[3,:3] *= 0.05 # perspective parameters need to be very small
    np.clip(perturb[3,:3], -3e-3, 3e-3, out=perturb[3,:3])

    return W + perturb


@lru_cache()
def make_dest_coords(sh):
    """
    Make coordinate list for destination array of shape sh
    """
    zz,xx,yy = np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
    hh = np.ones(sh, dtype=np.int)
    coords = np.concatenate([zz[...,None], xx[...,None],
                             yy[...,None], hh[...,None]], axis=-1)
    return coords.astype(floatX)


@lru_cache()
def make_dest_corners(sh):
    """
    Make coordinate list of the corners of destination array of shape sh
    """
    corners = list(itertools.product(*([0,1],)*3))
    sh = np.subtract(sh, 1) # 0-based indices
    corners = np.multiply(sh, corners)
    corners = np.hstack((corners, np.ones((8,1)))) # homogeneous coords
    return corners


class WarpingOOBError(ValueError):
    def __init__(self, *args, **kwargs):
        super(WarpingOOBError, self).__init__( *args, **kwargs)


def warp_slice(inp_src, ps, M, target_src=None, target_ps=None,
               target_discrete_ix=None):
    """
    Cuts a warped slice out of the input image and out of the target_src image.
    Warping is applied by multiplying the original source coordinates with
    the inverse of the homogeneous (forward) transformation matrix ``M``.

    "Source coordinates" (``src_coords``) signify the coordinates of voxels in
    ``inp_src`` and ``target_src`` that are used to compose their respective warped
    versions. The idea here is that not the images themselves, but the
    coordinates from where they are read are warped. This allows for much higher
    efficiency for large image volumes because we don't have to calculate the
    expensive warping transform for the whole image, but only for the voxels
    that we eventually want to use for the new warped image.
    The transformed coordinates usually don't align to the discrete
    voxel grids of the original images (meaning they are not integers), so the
    new voxel values are obtained by linear interpolation.

    Parameters
    ----------
    inp_src: h5py.Dataset
        Input image source (in HDF5)
    ps: tuple
        (spatial only) Patch size ``(D, H, W)``
        (spatial shape of the neural network's input node)
    M: np.ndarray
        Forward warping tansformation matrix (4x4).
        Must contain translations in source and target_src array.
    target_src: h5py.Dataset or None
        Optional target source array to be extracted from in the same way.
    target_ps: tuple
        Patch size for the ``target_src`` array.
    target_discrete_ix: list
        List of target channels that contain discrete values.
        By default (``None``), every channel is is seen as discrete (this is
        generally the case for classification tasks).
        This information is used to decide what kind of interpolation should
        be used for reading target data:
        - discrete targets are obtained by nearest-neighbor interpolation
        - non-discrete (continuous) targets are linearly interpolated.

    Returns
    -------
    inp: np.ndarray
        Warped input image slice
    target: np.ndarray or None
        Warped target_src image slice
        or ``None``, if ``target_src is None``.
    """

    ps = tuple(ps)
    if len(inp_src.shape) == 3:
        print(f'inp_src.shape: {inp_src.shape}')
        raise NotImplementedError(
            'elektronn3 has dropped support for data stored in raw 3D form without a channel axis. '
            'Please always supply it with a prepended channel, so it\n'
            'has the form (C, D, H, W) (or in ELEKTRONN2 terms: (f, z, x, y)).'
        )
    elif len(inp_src.shape) == 4:
        n_f = inp_src.shape[0]
        sh = inp_src.shape[1:]
    else:
        raise ValueError('inp_src wrong dim/shape')

    M_inv = np.linalg.inv(M.astype(np.float64)).astype(floatX) # stability...
    dest_corners = make_dest_corners(ps)
    src_corners = np.dot(M_inv, dest_corners.T).T
    if np.any(M[3,:3] != 0): # homogeneous divide
        src_corners /= src_corners[:,3][:,None]

    # check corners
    src_corners = src_corners[:,:3]
    lo = np.min(np.floor(src_corners), 0).astype(np.int)
    hi = np.max(np.ceil(src_corners + 1), 0).astype(np.int) # add 1 because linear interp
    if np.any(lo < 0) or np.any(hi >= sh):
        raise WarpingOOBError("Out of bounds")
    # compute/transform dense coords
    dest_coords = make_dest_coords(ps)
    src_coords = np.tensordot(dest_coords, M_inv, axes=[[-1],[1]])
    if np.any(M[3,:3] != 0): # homogeneous divide
        src_coords /= src_coords[...,3][...,None]
    # cut patch
    src_coords = src_coords[...,:3]
    # Add 1 to hi to include this coordinate!
    img_cut = slice_h5(inp_src, lo, hi + 1, dtype=floatX)

    inp = np.zeros((n_f,)+ps, dtype=floatX)
    lo = lo.astype(floatX)
    for k in range(n_f):
        map_coordinates_linear(img_cut[k], src_coords, lo, inp[k])
    if target_src is not None:
        target_ps = tuple(target_ps)
        n_f_t = target_src.shape[0]

        off = np.subtract(sh, target_src.shape[1:])
        if np.any(np.mod(off, 2)):
            raise ValueError("targets must be centered w.r.t. images")
        off //= 2

        off_ps = np.subtract(ps, target_ps)
        if np.any(np.mod(off_ps, 2)):
            raise ValueError("targets must be centered w.r.t. images")
        off_ps //= 2

        src_coords_target = src_coords[
            off_ps[0]:off_ps[0]+target_ps[0],
            off_ps[1]:off_ps[1]+target_ps[1],
            off_ps[2]:off_ps[2]+target_ps[2]
        ]
        # shift coords to be w.r.t. to origin of target_src array
        lo_targ = np.floor(src_coords_target.min(2).min(1).min(0) - off).astype(np.int)
        # add 1 because linear interp
        hi_targ = np.ceil(src_coords_target.max(2).max(1).max(0) - off + 1).astype(np.int)
        if np.any(lo_targ < 0) or np.any(hi_targ >= target_src.shape[1:]):
             raise WarpingOOBError("Out of bounds for target_src")
        # dtype is float as well here because of the static typing of the
        # numba-compiled map_coordinates functions
        target_cut = slice_h5(target_src, lo_targ, hi_targ + 1, dtype=floatX)

        # TODO: This and the checks below only make sense for discrete targets. Continuous targets are currently BROKEN.
        n_target_classes = target_cut.max()
        src_coords_target = np.ascontiguousarray(src_coords_target, dtype=floatX)
        target = np.zeros((n_f_t,) + target_ps, dtype=floatX)
        lo_targ = (lo_targ + off).astype(floatX)
        if target_discrete_ix is None:
            target_discrete_ix = [True for i in range(n_f_t)]
        else:
            target_discrete_ix = [i in target_discrete_ix for i in range(n_f_t)]

        for k, discr in enumerate(target_discrete_ix):
            if discr:
                map_coordinates_nearest(target_cut[k], src_coords_target, lo_targ, target[k])
            else:
                map_coordinates_linear(target_cut[k], src_coords_target, lo_targ, target[k])

        if np.any(target > n_target_classes):
            print(f'warp_slice: Invalid target: max = {target.max()}. Clipping target...')
            target = target.clip(0, n_target_classes)
            # TODO: Or should we just throw an error? (~ WarpingOOB)
    else:
        target = None
    return inp, target


def get_warped_slice(inp_src, ps, aniso_factor=2, sample_aniso=True,
                     warp_amount=1.0, lock_z=True, no_x_flip=False, perspective=False,
                     target_src=None, target_ps=None,
                     target_discrete_ix=None, rng=None):
    """
    (Wraps :py:meth:`elektronn2.data.coord_transforms.warp_slice()`)

    Generates the warping transformation parameters and composes them into a
    single 4D homogeneous transformation matrix ``M``.
    Then this transformation is applied to ``inp_source`` and ``target`` in the
    ``warp_slice()`` function and the transformed input and target image are
    returned.

    Parameters
    ----------
    inp_src: h5py.Dataset
        Input image source (in HDF5)
    ps: np.array
        Patch size (spatial shape of the neural network's input node)
    aniso_factor: float
        Anisotropy factor that determines an additional scaling in ``z``
        direction.
    sample_aniso: bool
        Scale coordinates by ``1 / aniso_factor`` while warping.
    warp_amount: float
        Strength of the random warping transformation. A lower ``warp_amount``
        will lead to less distorted images.
    lock_z: bool
        Exclude ``z`` coordinates from the random warping transformations.
    no_x_flip: bool
        Don't flip ``x`` axis during random warping.
    perspective: bool
        Apply perspective transformations (in addition to affine ones).
    target_src: h5py.Dataset
        Target image source (in HDF5)
    target_ps: np.array
        Target patch size
    target_discrete_ix:
        List of target channels that contain discrete values.
        By default (``None``), every channel is is seen as discrete (this is
        generally the case for classification tasks).
        This information is used to decide what kind of interpolation should
        be used for reading target data:
        - discrete targets are obtained by nearest-neighbor interpolation
        - non-discrete (continuous) targets are linearly interpolated.
    rng: np.random.mtrand.RandomState
        Random number generator state (obtainable by
        ``np.random.RandomState()``). Passing a known state makes the random
        transformations reproducible.

    Returns
    -------
    inp: np.ndarray
        (Warped) input image slice
    target: np.ndarray
        (Warped) target slice
    """
    # TODO: Ensure everything assumes (f, z, x, y) shape

    rng = np.random.RandomState() if rng is None else rng

    strip_2d = False
    if len(ps)==2:
        strip_2d = True
        ps = np.array([1]+list(ps))
        if target_src is not None:
            target_ps = np.array([1]+list(target_ps))

    dest_center = np.array(ps, dtype=np.float) / 2
    src_remainder = np.array(np.mod(ps, 2), dtype=np.float) / 2

    if target_ps is not None:
        t_center = np.array(target_ps, dtype=np.float) / 2
        off = np.subtract(inp_src.shape[1:], target_src.shape[1:])
        off //= 2
        lo_pos = np.maximum(dest_center, t_center+off)
        hi_pos = np.minimum(inp_src.shape[1:] - dest_center, target_src.shape[1:] - t_center + off)
    else:
        lo_pos = dest_center
        hi_pos = inp_src.shape[1:] - dest_center
    assert np.all([lo_pos[i] < hi_pos[i] for i in range(3)])
    z = rng.randint(lo_pos[0], hi_pos[0]) + src_remainder[0]
    y = rng.randint(lo_pos[1], hi_pos[1]) + src_remainder[1]
    x = rng.randint(lo_pos[2], hi_pos[2]) + src_remainder[2]
    F = get_random_flipmat(no_x_flip, rng)
    if no_x_flip:
        S = np.eye(4, dtype=floatX)
    else:
        S = get_random_swapmat(lock_z, rng)

    if np.isclose(warp_amount, 0):
        R = np.eye(4, dtype=floatX)
        W = np.eye(4, dtype=floatX)
    else:
        R = get_random_rotmat(lock_z, warp_amount, rng)
        W = get_random_warpmat(lock_z, perspective, warp_amount, rng)

    T_src = translate(-z, -y, -x)
    S_src = scale(aniso_factor, 1, 1)

    if sample_aniso:
        S_dest = scale(1.0 / aniso_factor, 1, 1)
    else:
        S_dest = identity()
    T_dest = translate(dest_center[0], dest_center[1], dest_center[2])

    M = chain_matrices([T_dest, S_dest, R, W, F, S, S_src, T_src])

    inp, target = warp_slice(
        inp_src,
        ps,
        M,
        target_src=target_src,
        target_ps=target_ps,
        target_discrete_ix=target_discrete_ix
    )

    if strip_2d:
        inp = inp[:,0]
        if target_src is not None:
            target = target[:,0]

    return inp, target
