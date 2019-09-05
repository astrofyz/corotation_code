import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.filters import gaussian_filter, median_filter
from astropy.io import fits
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
# from astropy.stats import sigma_clipped_stats
from photutils.isophote import EllipseGeometry, Ellipse
from photutils import EllipticalAperture, Background2D, EllipticalAnnulus, aperture_photometry, RectangularAperture
from photutils.utils import calc_total_error
from scipy.interpolate import splrep, splev, UnivariateSpline
import scipy.signal as signal
import warnings
import numpy.ma as ma
import cv2
import scipy.fftpack as fft
from astropy.convolution import Gaussian1DKernel, convolve
import os
from mod_read import *


def find_outer(image, **kwargs):
    """ image : shifted to center segmentation map with main object == 1 or instance of image class
    :returns : r.hist.pix, r.max.pix (0.99th quantile), r.min.pix (0.05th quantile), FD """

    if isinstance(image, ImageClass):
        im = image['mask.center']
    else:
        im = image
    idx_main = np.array(np.where(im == 1))
    centre = np.array([int(dim / 2) for dim in np.shape(im)])
    r = np.array([np.sqrt(np.dot(centre-idx_main.T[i], centre-idx_main.T[i])) for i in range(len(idx_main.T))])
    r_max, r_min = np.quantile(r, [0.99, 0.05])  # in pixels
    FD_bin = 2*(np.quantile(r, 0.75) - np.quantile(r, 0.25))/(len(r))**(1./3.)  # ¯\_(ツ)_/¯
    return r, r_max, r_min, FD_bin


# def find_outer(image, **kwargs):
#     """ image : segmentation map with main object == 1 (or instance of image class (to be added))
#     returns : r_max (0.99th quantile), r_min(0.05th quantile), FD bin """
#     # вот это сделать функцией, которая работает только с изображением, а не с классом. и считать.
#     #а можно ли делать перенаправление какое-нибудь? чтобы функция замещалась при определенном аргументе?
#     idx_main = np.array(np.where(image == 1))
#     centre = np.array([int(dim / 2) for dim in np.shape(image)])
#     r = np.array([np.sqrt(np.dot(centre-idx_main.T[i], centre-idx_main.T[i])) for i in range(len(idx_main.T))])
#     r_max, r_min = np.quantile(r, [0.99, 0.05])  # in pixels
#     FD_bin = 2*(np.quantile(r, 0.75) - np.quantile(r, 0.25))/(len(r))**(1./3.)  # ¯\_(ツ)_/¯
#     if 'plot' in kwargs:
#         # реализовать плот отдельной функцией
#         plt.figure()
#         r_edges = np.arange(np.amin(r), np.amax(r), FD_bin)
#         plt.hist(r, bins=r_edges, density=True, alpha=0.5, color='lightseagreen')
#         plt.axvline(r_max, color='red', label='$r_{max}$')
#         plt.axvline(r_min, color='darkorange', label='$r_{min}$')
#         if 'petro' in kwargs:
#             plt.axvline(kwargs.get('petro'), color='indigo', label='petro')
#         if 'petro50' in kwargs:
#             plt.axvline(kwargs.get('petro50'), color='green', label='petro50')
#         if 'title' in kwargs:
#             plt.title(kwargs.get('title'))
#         plt.xlabel('r (pix)')
#         plt.legend()
#         plt.show()
#     return r_max, r_min, FD_bin


def ellipse_fit(image):
    """image : instance of ImageClass (in certain band) (could be adjusted)"""

    xc, yc = np.array([int(dim / 2) for dim in np.shape()])
    eps = np.sqrt(1-(image['cat'][1].data.T[0]['B_IMAGE']/image['cat'][1].data.T[0]['A_IMAGE'])**2)
    pa = image['cat'].data.T[0]['THETA_IMAGE']*np.pi/180.
    geom_inp = EllipseGeometry(x0=xc, y0=yc, sma=image['petroRad'], eps=eps, pa=pa)
    aper_inp = EllipticalAperture((geom_inp.x0, geom_inp.y0), geom_inp.sma, geom_inp.sma*np.sqrt(1 - geom_inp.eps**2),
                                  geom_inp.pa)
    # чтобы построить начальный эллипс, нужно либо делать эту функцию методом класса, либо хранить всякие характеристики эллипса
    ellipse = Ellipse(image['real.center'], geom_inp)
    # добавить флагов для различных методов и играться со всеми, пока не получится результат
