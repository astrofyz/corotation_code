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
from astropy.stats import SigmaClip
from photutils.utils import calc_total_error
from scipy.interpolate import splrep, splev, UnivariateSpline
import scipy.signal as signal
import warnings
import numpy.ma as ma
import cv2
import scipy.fftpack as fft
from astropy.convolution import Gaussian1DKernel, convolve
import os


def main_obj(cat, mask, **kwargs):
    """cat - catalog file, mask - image of segmentation file
    kwargs:
    xy - list of x,y pixels of object's centre (e.g. from funsky)
    radec - sky coord ra,dec of object's centre, need wcs
    wcs - world coordinate system from header of real file"""

    # w = wcs.WCS(real[0].header)
    # ra_real, dec_real = table.loc[all_table.objID14 == int(name), ['ra', 'dec']].values[0]
    if 'xy' in kwargs:
        x_real, y_real = kwargs.get('xy')
    elif 'radec' in kwargs:
        ra_real, dec_real = kwargs.get('radec')
        w = kwargs.get('wcs')
        x_real, y_real = w.wcs_world2pix(ra_real, dec_real, 1)
    else:
        print('Coordinates of object are not found!')
        return mask

    delta_x = abs(cat[1].data['X_IMAGE'] - x_real)
    delta_y = abs(cat[1].data['Y_IMAGE'] - y_real)

    id_x = np.argmin(delta_x)
    id_y = np.argmin(delta_y)

    idx = (id_x & id_y) + 1

    if idx != 1:
        idxs_real = mask == idx
        mask[~idxs_real] = 0
        mask[idxs_real] = 1
        return mask
    else:
        idxs_real = mask == 1
        mask[idxs_real] = 1
        mask[~idxs_real] = 0
        return mask


def calc_bkg(image, mask, **kwargs):
    """image - image 2D array
    mask - segmentation image
    kwargs:
    size - backfilter_size
    return:
    Background2D object"""

    if 'size' in kwargs:
        size = kwargs.get('size')
    else:
        size = int(np.shape(image)[0]/4)

    # if mask.all() != None:  # переписать, проверяя сначала isinstance NoneType, then else
    #     bkg = Background2D(image, (size, size), filter_size=(3, 3), mask=mask)
    # else:
    #     sigma_clip = SigmaClip(sigma=3.)
    #     bkg = Background2D(image, (size, size), filter_size=(3, 3), sigma_clip=sigma_clip)

    NoneType = type(None)
    if isinstance(mask, NoneType):
        sigma_clip = SigmaClip(sigma=3.)
        bkg = Background2D(image, (size, size), filter_size=(3, 3), sigma_clip=sigma_clip)
    else:
        bkg = Background2D(image, (size, size), filter_size=(3, 3), mask=mask)
    return bkg


def to_mag(image, zp, texp=53.907):
    """kwargs:
    image - image 2D array
    zp - zeropoint
    (for one instance now)"""

    return zp-2.5*np.log10(abs(image/texp))


def correct_FWHM(image, fwhm_res):
    """"image - input image
        fwhm_inp - original fwhm
        fwhm_res - result fwhm"""
    sigma_inp = image['seeing'] / 2. / np.sqrt(2. * np.log(2.))
    sigma_res = fwhm_res / 2. / np.sqrt(2. * np.log(2.))
    sigma_f = sigma_inp / sigma_res / 2. / np.sqrt(np.pi)
    image_res = gaussian_filter(image['real'], sigma_f)
    return image_res


def rotate_and_scale(image, angle, sx=1., sy=1.):
    x0, y0 = 0.5*np.array(np.shape(image))
    x1, y1 = 0.5*np.array(np.shape(image))

    rot_mtx = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])  #rotation matrix
    sca_mtx = np.array([[sx, 0], [0., sy]])  # scaling matrix; probably could be replased by s
    aff_mtx = np.dot(sca_mtx, rot_mtx)

    offset = np.array([x0, y0]) - np.dot(np.array([x1, y1]), aff_mtx)
    im_res = affine_transform(image, aff_mtx.T, mode='nearest', offset=offset)

    return im_res