# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert, Marius Killinger


import numpy as np
from PIL import Image
from scipy.misc import imsave

from elektronn3 import floatX


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

    #normalizing the arrays
    comp = normalise(comp)
    background = normalise(background)

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
    src -- 0 to 255 normalised image array
    dst -- 0 to 255 normalised image array

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    '''

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
    out = out.astype('uint8')
    out = Image.fromarray(out, 'RGBA')
    return out



def normalise(src):
    """
    Normalizes the arrays from 0 to 255, uint8 for senseful png

    Parameters
    ----------
    src: np_array to be normalized

    Returns
    -------
    normalized src
    """
    #scaling to 0
    src = src - np.min(src)

    #normalising from 0 to 255
    src = np.array(src * 255 / src.max(), dtype='uint8')

    return src
