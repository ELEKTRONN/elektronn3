# -*- coding: utf-8 -*-
# ELEKTRONN2 - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger, Philipp Schubert

import logging
import os
from collections import OrderedDict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

from elektronn3 import floatX
from elektronn3.data.locking import FileLock

logger = logging.getLogger('elektronn3log')


class Scroller:
    def __init__(self, axes, images, names, init_z=None):
        self.axes = axes
        for ax in axes:
            ax.grid(b=False)
        # ax.set_title('use scroll wheel to navigate images')

        self.images = list(map(np.ascontiguousarray, images))
        self.n_slices = images[0].shape[0]
        self.ind = self.n_slices // 2 if init_z is None else init_z

        self.imgs = []
        for ax, dat, name in zip(axes, images, names):
            if name in ['id', 'ids', 'ID', 'IDs', 'seg', 'SEG', 'Seg', 'lab',
                        'label', 'Label']:
                cmap = 'nipy_spectral'
            else:
                cmap = 'gray'

            self.imgs.append(
                ax.imshow(dat[self.ind], interpolation='None', cmap=cmap))
            ax.set_xlabel(name)

        self.update()

    def onscroll(self, event):
        # print ("%s %s" % (event.button, event.step))
        if event.button=='up':
            self.ind = np.clip(self.ind + 1, 0, self.n_slices - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.n_slices - 1)

        self.update()

    def update(self):
        for ax, im, dat in zip(self.axes, self.imgs, self.images):
            im.set_data(dat[self.ind])
            ax.set_ylabel('slice %s' % self.ind)
            im.axes.figure.canvas.draw()


def scroll_plot(images, names=None, init_z=None):
    """
    Creates a plot 1x2 image plot of 3d volume images
    Scrolling changes the displayed slices

    Parameters
    ----------

    images: list of arrays (or single)
      Each array of shape (z,y,x) or (z,y,x,RGB)
    names: list of strings (or single)
      Names for each image

    Usage
    -----

    For the scroll interaction to work, the "scroller" object
    must be returned to the calling scope

    >>> fig, scroller = _scroll_plot4(images, names)
    >>> fig.show()

    """
    if names is None:
        n = 1 if isinstance(images, np.ndarray) else len(images)
        names = [str(i) for i in range(n)]

    if isinstance(images, np.ndarray):
        return _scroll_plot1(images, names, init_z)
    elif len(images)==2:
        assert len(names) >= 2
        return _scroll_plot2(images, names, init_z)
    elif len(images)==4:
        assert len(names) >= 4
        return _scroll_plot4(images, names, init_z)


def _scroll_plot1(image, name, init_z):
    """
    Creates a plot of 3d volume images
    Scrolling changes the displayed slices

    Parameters
    ----------

    images:  array of shape (z,x,y) or (z,x,y,RGB)

    Usage
    -----

    For the scroll interaction to work, the "scroller" object
    must be returned to the calling scope

    >>> fig, scroller = scroll_plot(image, name)
    >>> fig.show()

    """
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(111)

    scroller = Scroller([ax1], [image, ], [name, ], init_z)
    fig.canvas.mpl_connect('scroll_event', scroller.onscroll)
    fig.tight_layout()
    return scroller


def _scroll_plot2(images, names, init_z):
    """
    Creates a plot 1x2 image plot of 3d volume images
    Scrolling changes the displayed slices

    Parameters
    ----------

    images: list of 2 arrays
      Each array of shape (z,y,x) or (z,y,x,RGB)
    names: list of 2 strings
      Names for each image

    Usage
    -----

    For the scroll interaction to work, the "scroller" object
    must be returned to the calling scope

    >>> fig, scroller = _scroll_plot4(images, names)
    >>> fig.show()

    """
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)

    scroller = Scroller([ax1, ax2], images, names, init_z)
    fig.canvas.mpl_connect('scroll_event', scroller.onscroll)
    fig.tight_layout()
    return scroller


def _scroll_plot4(images, names, init_z):
    """
    Creates a plot 2x2 image plot of 3d volume images
    Scrolling changes the displayed slices

    Parameters
    ----------

    images: list of 4 arrays
      Each array of shape (z,y,x) or (z,y,x,RGB)
    names: list of 4 strings
      Names for each image

    Usage
    -----

    For the scroll interaction to work, the "scroller" object
    must be returned to the calling scope

    >>> fig, scroller = _scroll_plot4(images, names)
    >>> fig.show()

    """
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)

    scroller = Scroller([ax1, ax2, ax3, ax4], images, names, init_z)
    fig.canvas.mpl_connect('scroll_event', scroller.onscroll)
    fig.tight_layout()
    return scroller


def _embed3d2d(a, border_width=1, normalize=False, output_ratio=1.5, ):
    """
    Embed an 3d array into an 2d matrix by tiling.
    The last two dimensions of ``a`` are assumed to be spatial, the first is tiled.
    """
    sh = a.shape
    assert len(sh)==3
    n = sh[0]
    nhor = int(np.ceil(np.sqrt(n * output_ratio)))  # aim: ratio 16:9
    nvert = int(np.ceil(float(n) / nhor))  # warning: too big: nvert*nhor >= n

    if normalize:
        maxs = [np.max(a[i, :, :]) + 1e-8 for i in range(n)]
        mins = [np.min(a[i, :, :]) for i in range(n)]
    else:
        maxs = [1] * n
        mins = [0] * n

    ret = np.zeros(
        (nvert * (border_width + sh[1]), nhor * (border_width + sh[2])),
        dtype=floatX)

    for j in range(nvert):
        for i in range(nhor):
            if i + j * nhor >= n:
                return ret
            ret[j*(border_width+sh[1]):j*(border_width+sh[1])+sh[1],
                i*(border_width+sh[2]):i*(border_width+sh[2])+sh[2]] = \
               (a[i+j*nhor,:,:]-mins[i+j*nhor])/(maxs[i+j*nhor]-mins[i+j*nhor])

    return ret


def embedfilters(filters, border_width=1, normalize=False, output_ratio=1.0,
                 rgb_axis=None):
    """
    Embed an nd array into an 2d matrix by tiling.
    The last two dimensions of ``a`` are assumed to be spatial,
    the others are tiled recursively.
    """
    if rgb_axis is not None:
        assert filters[rgb_axis]==3
        channels = []
        for i in range(3):
            slice = [slice(None), ] * filters.ndim
            slice[rgb_axis] = i
            f = filters[slice]
            channels.append(
                embedfilters(f, border_width, normalize, output_ratio))
        return np.dstack(channels)

    if filters.ndim==3:
        return _embed3d2d(filters, border_width, normalize, output_ratio)
    elif filters.ndim > 3:
        parts = []
        for f in filters:
            parts.append(
                embedfilters(f, border_width, normalize, output_ratio))
        parts = np.concatenate([x[None, ...] for x in parts])
        return embedfilters(parts, border_width, normalize, output_ratio)


def sma(c, n):
    """
    Returns box-SMA of c with box length n, the returned array has the same
    length as c and is const-padded at the beginning
    """
    if n==0:
        return c
    ret = np.cumsum(c, dtype=float)
    ret[n:] = (ret[n:] - ret[:-n]) / n
    m = min(n, len(c))
    ret[:n] = ret[:n] / np.arange(1, m + 1)  # unsmoothed
    return ret


def add_timeticks(ax, times, steps, time_str='mins', num=5):
    N = int(times[-1])
    k = max(N / num, 1)
    k = int(np.log10(k))  # 10-base of locators
    m = int(np.round(float(N) / (num * 10 ** k)))  # multiple of base
    s = max(m * 10 ** k, 1)
    x_labs = np.arange(0, N, s, dtype=np.int)
    x_ticks = np.interp(x_labs, times, steps)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labs)
    ax.set_xlim(0, steps[-1])
    ax.set_xlabel('Runtime [%s]' % time_str)  # (%s)'%("{0:,d}".format(N)))


def plot_hist(timeline, history, save_name, loss_smoothing_length=200,
              autoscale=True):
    """Plot graphical info during Training"""
    plt.ioff()
    try:
        # Subsample points for plotting
        N = len(timeline)
        x_timeline = np.arange(N)
        s = max((len(timeline) // 2000), 1)
        x_timeline = x_timeline[::s]
        timeline = timeline[::s]
        s = max((len(history) // 2000), 1)
        history = history[::s]

        if timeline['time'][-1] < 120 * 60:
            runtime = str(int(timeline['time'][-1] / 60)) + ' mins'
        else:
            runtime = "%.1f hours" % (timeline['time'][-1] / 3600)

        # check if valid data is available
        if not np.any(np.isnan(history['valid_loss'])):
            l = history['valid_loss'][-10:]
        else:
            l = history['train_loss'][-10:]

        loss_cap = l.mean() + 2 * l.std()
        lt = timeline['loss'][-200:]
        lt_m = lt.mean()
        lt_s = lt.std()
        loss_cap_t = lt_m + 2 * lt_s
        loss_cap = np.maximum(loss_cap, loss_cap_t)

        if np.all(timeline['loss'] > 0):
            loss_floor = 0.0
        else:
            loss_floor = lt_m - 2 * lt_s

        ### Timeline, Loss ###
        plt.figure(figsize=(16, 12))
        plt.subplot(211)
        plt.plot(x_timeline, timeline['loss'], 'b-', alpha=0.5,
                 label='Update Loss')
        loss_smooth = sma(timeline['loss'], loss_smoothing_length)
        plt.plot(x_timeline, loss_smooth, 'k-', label='Smooth update Loss',
                 linewidth=3)

        if autoscale:
            plt.ylim(loss_floor, loss_cap)
            plt.xlim(0, N)

        plt.legend(loc=0)
        plt.hlines(lt_m, 0, N, linestyle='dashed', colors='r', linewidth=2)
        plt.hlines(lt_m + lt_s, 0, N, linestyle='dotted', colors='r',
                   linewidth=1)
        plt.hlines(lt_m - lt_s, 0, N, linestyle='dotted', colors='r',
                   linewidth=1)
        plt.xlabel('Update steps %s, total runtime %s' % (N - 1, runtime))

        ax = plt.twiny()
        if timeline['time'][-1] > 120 * 60:
            add_timeticks(ax, timeline['time'] / 3600, x_timeline,
                          time_str='hours')
        else:
            add_timeticks(ax, timeline['time'] / 60, x_timeline,
                          time_str='mins')

        ### Loss vs Prevalence ###
        plt.subplot(212)
        c = 1.0 - (timeline['time'] / timeline['time'].max())
        plt.scatter(timeline['batch_char'], timeline['loss'], c=c, marker='.',
                    s=80, cmap='gray', edgecolors='face')
        if autoscale:
            bc = timeline['batch_char'][-200:]
            bc_m = bc.mean()
            bc_s = bc.std()
            bc_cap = bc_m + 2 * bc_s
            if np.all(bc > 0):
                bc_floor = -0.01
            else:
                bc_floor = bc_m - 2 * bc_s
            plt.ylim(loss_floor, loss_cap)
            plt.xlim(bc_floor, bc_cap)

        plt.xlabel('Mean target of batch')
        plt.ylabel('Loss')

        plt.tight_layout()
        with FileLock('plotting'):
            plt.savefig(save_name + ".timeline.png", bbox_inches='tight')

        ###################################################################
        ### History Loss ###
        plt.figure(figsize=(16, 12))
        plt.subplot(311)
        plt.plot(history['steps'], history['train_loss'], 'g-',
                 label='Train Loss', linewidth=3)
        plt.plot(history['steps'], history['valid_loss'], 'r-',
                 label='Valid Loss', linewidth=3)
        if autoscale:
            plt.ylim(loss_floor, loss_cap)
            plt.xlim(0, history['steps'][-1])

        plt.legend(loc=0)
        # plt.xlabel('Update steps %s, total runtime %s'%(N-1, runtime))

        ax = plt.twiny()
        if timeline['time'][-1] > 120 * 60:
            add_timeticks(ax, timeline['time'] / 3600, x_timeline,
                          time_str='hours')
        else:
            add_timeticks(ax, timeline['time'] / 60, x_timeline,
                          time_str='mins')

        ### History Loss gains ###
        plt.subplot(312)
        plt.plot(history['steps'], history['loss_gain'], 'b-',
                 label='Loss Gain at update', linewidth=3)
        plt.hlines(0, 0, history['steps'][-1], linestyles='dotted')
        plt.plot(history['steps'], history['lr'], 'r-', label='LR',
                 linewidth=3)
        # plt.xlabel('Update steps %s, total runtime %s'%(N-1, runtime))
        plt.legend(loc=3)
        std = history['loss_gain'][:5].std() * 2 if len(history) > 6 else 1.0
        if autoscale:
            # add epsilon to suppress matplotlib warning in case of CG
            plt.ylim(-std, std + 1e-10)
            plt.xlim(0, history['steps'][-1])

        ax2 = plt.twinx()
        ax2.plot(history['steps'], history['mom'], 'r-', label='MOM')
        ax2.plot(history['steps'], history['gradnetrate'], 'r-',
                 label='GradNetRate')

        ax2.set_ylim(-1, 1)
        if autoscale:
            ax2.set_xlim(0, history['steps'][-1])
        ax2.legend(loc=4)

        ### Errors ###
        plt.subplot(313)
        cutoff = 2
        if len(history) > (cutoff + 1):
            history = history[cutoff:]
        # check if valid data is available
        if not np.any(np.isnan(history['valid_err'])):
            e = history['valid_err'][-200:]
        else:
            e = history['train_err'][-200:]
        e_m = e.mean()
        e_s = e.std()
        err_cap = e_m + 2 * e_s
        if np.all(e > 0):
            err_floor = 0.0
        else:
            err_floor = e_m - 2 * e_s

        plt.plot(history['steps'], history['train_err'], 'g--',
                 label='Train error', linewidth=1)
        plt.plot(history['steps'], history['valid_err'], 'r--',
                 label='Valid Error', linewidth=1)

        plt.plot(history['steps'], sma(history['train_err'], 8), 'g-',
                 label='Smooth train error', linewidth=3)
        if not np.any(np.isnan(sma(history['valid_err'], 8))):
            plt.plot(history['steps'], sma(history['valid_err'], 8), 'r-',
                     label='Smooth valid Error', linewidth=3)
        if autoscale:
            plt.ylim(err_floor, err_cap)
            plt.xlim(0, history['steps'][-1])

        plt.grid()
        plt.legend(loc=0)
        plt.xlabel('Update steps %s, total runtime %s' % (N - 1, runtime))
        plt.tight_layout()
        with FileLock('plotting'):
            plt.savefig(save_name + ".history.png", bbox_inches='tight')

    except ValueError:
        # When arrays are empty
        logger.warning("An error occurred during plotting.")


def plot_var(var, save_name):
    # [i, nll, nll.std, conc.mean, conc.std,]
    plt.figure(figsize=(16, 12))
    plt.subplot(211)
    plt.plot(var[:, 0], var[:, 1], 'b-', alpha=0.6)
    plt.plot(var[:, 0], sma(var[:, 1], 100), 'g-', linewidth=3)
    plt.plot(var[:, 0], sma(var[:, 1] + var[:, 2], 100), 'r:', linewidth=2)
    plt.plot(var[:, 0], sma(var[:, 1] - var[:, 2], 100), 'r:', linewidth=2)
    plt.title("NLL")

    plt.subplot(212)
    plt.plot(var[:, 0], var[:, 3], 'b-', alpha=0.6)
    plt.plot(var[:, 0], sma(var[:, 3], 100), 'g-', linewidth=3)
    plt.plot(var[:, 0], sma(var[:, 3] + var[:, 4], 100), 'r:', linewidth=2)
    plt.plot(var[:, 0], sma(var[:, 3] - var[:, 4], 100), 'r:', linewidth=2)
    plt.title("Concentration")

    with FileLock('plotting'):
        plt.savefig(save_name + ".Beta1.png", bbox_inches='tight')

    plt.figure(figsize=(12, 12))
    c = 1.0 - ((var[:, 0]).astype(floatX) / var[-1, 0])

    plt.subplot(221)
    plt.scatter(var[:, 1], var[:, 3], c=c, marker='.', s=80, cmap='gray',
                edgecolors='face')
    plt.title("Concentration vs. NLL")

    plt.subplot(222)
    plt.scatter(var[:, 2], var[:, 3], c=c, marker='.', s=80, cmap='gray',
                edgecolors='face')
    plt.title("Concentration vs. NLL.std")

    plt.subplot(223)
    plt.scatter(var[:, 3], var[:, 4], c=c, marker='.', s=80, cmap='gray',
                edgecolors='face')
    plt.title("Concentration vs. Concentration.std")

    plt.subplot(224)
    plt.scatter(var[:, 1], var[:, 2], c=c, marker='.', s=80, cmap='gray',
                edgecolors='face')
    plt.title("NLL vs. NLL.std")

    with FileLock('plotting'):
        plt.savefig(save_name + ".Beta2.png", bbox_inches='tight')


def plot_debug(var, debug_output_names, save_name):
    # [i, nll, other....]
    s = max((len(var) // 2000), 1)
    var = var[::s]
    plt.figure(figsize=(16, 12))
    colors = ['gold', 'b', 'darkblue', 'crimson', 'navajowhite', 'deepskyblue',
              'darkgray', 'maroon', 'palevioletred', 'forestgreen', ] * 2
    n = len(colors) // 2
    marker = ['-', ] * n + [':'] * n
    lw_s = [2, ] * n + [3, ] * n
    maxima = []
    minima = []
    total = sma(var[:, 1], 70)
    maxima.append(total[-100:].max())
    minima.append(total[-100:].min())
    plt.plot(var[:, 0], total, 'k-', linewidth=4, label='total loss')
    for i in range(len(debug_output_names)):  ###TODO automatic std intervals
        name = debug_output_names[i]
        smooth = sma(var[:, i + 2], 70)
        plt.plot(var[:, 0], smooth, color=colors[i], linestyle=marker[i],
                 linewidth=lw_s[i], label=name)
        maxima.append(smooth[-100:].max())
        minima.append(smooth[-100:].min())

    plt.title("Debug Outputs")
    cap_hi = np.max([x for x in maxima if np.isfinite(x)]) * 1.5
    cap_lo = np.min([x for x in minima if np.isfinite(x)])
    plt.ylim(cap_lo, cap_hi)
    plt.legend(loc=0)
    plt.hlines(0, var[0, 0], var[-1, 0], linewidth=1)
    plt.grid()

    with FileLock('plotting'):
        plt.savefig(save_name + ".Debug.png", bbox_inches='tight')


def plot_regression(pred, target, save_name, loss_smoothing_length=200,
                    autoscale=True):
    """Plot graphical info during Training"""
    try:
        # Subsample points for plotting
        N = len(pred)
        s = max((len(pred) // 2000), 1)
        pred = pred[::s].ravel()
        target = target[::s].ravel()
        N = len(pred)
        x_timeline = np.arange(N)
        c = N - x_timeline
        plt.figure(figsize=(8, 8))
        ### Loss vs Prevalence ###
        plt.scatter(pred, target, c=c, marker='.', s=80, cmap='gray',
                    edgecolors='face')
        m = np.minimum(pred.min(), target.min())
        M = np.maximum(pred.max(), target.max())
        plt.plot([m, M], [m, M], 'r:')
        plt.ylim(m, M)
        plt.xlim(m, M)
        plt.xlabel('Prediction')
        plt.ylabel('Target')
        plt.tight_layout()
        with FileLock('plotting'):
            plt.savefig(save_name + ".regression.png", bbox_inches='tight')
    except ValueError:
        # When arrays are empty
        logger.warning("An error occurred during regression plotting.")


def plot_kde(pred, target, save_name, limit=90, scale='same', grid=50,
             take_last=4000):
    try:
        if take_last:
            pred = pred[-take_last:].ravel()
            target = target[-take_last:].ravel()

        if limit=='max':
            mp, mt = pred.min(), target.min()
            Mp, Mt = pred.max(), target.max()
        else:
            lo = 100 - limit
            mp, mt = np.percentile(pred, lo), np.percentile(target, lo)
            Mp, Mt = np.percentile(pred, limit), np.percentile(target, limit)

        if scale=='same':
            mp = np.minimum(mp, mt)
            Mp = np.maximum(Mp, Mt)
            mt = mp
            Mt = Mp

        if isinstance(grid, int):
            grid = [grid, grid]

        pg, tg = np.mgrid[mp:Mp:grid[0] * 1j, mt:Mp:grid[1] * 1j]

        positions = np.vstack([pg.ravel(), tg.ravel()])
        values = np.vstack([pred, target])
        kernel = stats.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, pg.shape)
        plt.figure()
        plt.xlim(mp, Mp)
        plt.ylim(mt, Mt)
        plt.xlabel("Prediction")
        plt.ylabel("Target")
        plt.imshow(np.rot90(f), cmap=plt.cm.gist_earth_r,
                   extent=[mp, Mp, mt, Mt])
        plt.contour(pg, tg, f)
        plt.plot([mt, Mt], [mt, Mt], 'r:')
        plt.tight_layout()
        with FileLock('plotting'):
            plt.savefig(save_name + ".regression_kde.png", bbox_inches='tight')
    except ValueError:
        # When arrays are empty
        logger.warning("An error occurred during regression kde plotting.")


def my_quiver(x, y, img=None, c=None):
    """
    first dim of x,y changes along vertical axis
    second dim changes along horizontal axis
    x: vertical vector component
    y: horizontal vector component
    """
    figure = plt.figure(figsize=(7, 7))

    if img is not None:
        plt.imshow(img, interpolation='none', alpha=0.22, cmap='gray')

    plt.quiver(x, y, c, angles='xy', units='xy', cmap='spring', pivot='middle',
               scale=0.5)
    return figure


def plot_trainingtarget(img, lab, stride=1):
    """
    Plots raw image vs target to check if valid batches are produced.
    Raw data is also shown overlaid with targets

    Parameters
    ----------

    img: 2d array
      raw image from batch
    lab: 2d array
      targets
    stride: int
      strides of targets

    """

    if len(lab) * stride!=len(img):
        off = (len(img) - stride * len(lab)) // 2 // stride
        if lab.ndim==3:
            assert lab.shape[2]==3
            new_t = np.zeros(
                (lab.shape[0] + 2 * off, lab.shape[1] + 2 * off, 3))
            new_t[off:-off, off:-off, :] = lab
        else:
            new_t = np.zeros((lab.shape[0] + 2 * off, lab.shape[1] + 2 * off))
            new_t[off:-off, off:-off] = lab

        lab = new_t

    if lab.ndim==3:
        assert lab.shape[2]==3
        img = img[:, :, None]
        img = np.repeat(img, 3, axis=2)

    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(img, interpolation='none', cmap=plt.get_cmap('gray'))
    plt.title('data')
    plt.subplot(132)
    plt.imshow(lab, interpolation='none', cmap=plt.get_cmap('gray'))
    plt.title('target')
    if img.shape==lab.shape:
        overlay = 0.75 * img + 0.25 * (1 - lab)
        plt.subplot(133)
        plt.imshow(overlay, interpolation='none', cmap=plt.get_cmap('gray'))
        plt.title('overlay')
    return img - lab


def plot_exectimes(exectimes, save_path='~/exectimes.png', max_items=32):
    """
    Plot neural execution time dict obtained from
    elektronn2.neuromancer.neural.Model.measure_exectimes()

    :param exectimes: OrderedDict of execution times
                      (output of Model.measure_exectimes())
    :param save_path: Where to save the plot
    :param max_items: Only the max_items largest execution times are given
                      names and are plotted independently.
                      Everything else is grouped under '(other nodes)'.
    """
    thresh_val = 0
    if len(exectimes) > max_items:
        thresh_val = sorted(list(exectimes.values()))[-max_items]
    filt_rtimes = OrderedDict()
    for key, val in exectimes.items():
        if val >= thresh_val:
            filt_rtimes[key] = val
    other = sum(exectimes.values()) - sum(filt_rtimes.values())
    node_names = list(filt_rtimes.keys())
    node_exectimes = list(filt_rtimes.values())
    if len(exectimes) > max_items:
        node_names += ['(other nodes)']
        node_exectimes += [other]
    cs = plt.cm.Set1(np.arange(len(node_exectimes)) / (len(node_exectimes)))

    sns.set_style("whitegrid")
    plt.figure(figsize=(13, 12))
    plt.title('Node execution times')
    plt.ylabel('Node')
    plt.xlabel('Time (in ms)')
    ax = sns.barplot(y=node_names, x=node_exectimes)
    with FileLock('plotting'):
        ax.get_figure().savefig(os.path.expanduser(save_path), bbox_inches='tight')
