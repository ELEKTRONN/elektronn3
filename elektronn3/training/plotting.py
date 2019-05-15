# -*- coding: utf-8 -*-
# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Marius Killinger, Martin Drawitsch, Philipp Schubert

import logging
import os

import numpy as np
from matplotlib import pyplot as plt
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


def plot_hist(timeline, history, save_path, loss_smoothing_length=200,
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
        plt.savefig(os.path.join(save_path, 'timeline.png'), bbox_inches='tight')

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
        plt.savefig(os.path.join(save_path, 'history.png'), bbox_inches='tight')

    except ValueError:
        # When arrays are empty
        logger.exception("An error occurred during plotting.")
