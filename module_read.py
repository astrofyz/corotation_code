import numpy as np
import pandas as pd
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
from module_prep_im import *


class ImageClass(dict):
    """Class to store images and additional data"""
    def __init__(self): #можно инициировать какими-нибудь разумными значениями что-нибудь, если что-то отсутствует. например, радиус.
        self['name'] = 'name'
        self['objid'] = 'objid dr7'
        self['ra'] = 0.
        self['dec'] = 0.
        self = dict()
    #     self['gain'] = 0.
    #     self['kk'] = 0.
    #     self['airmass'] = 0.
    #     self['aa'] = 0.
    #     self['seeing'] = 0.
    #     self['petroRad'] = 0.
    #     self['petroR50'] = 0.

    def prop(self, property_name, **kw):
        if 'data' in kw:
            self[property_name] = kw.get('data')
        if property_name not in self:
            print('Error: Property keyword doesn\'t exist')
        else:
            return self[property_name]


def read_images(names, bands='all', types='all',
            path_table='/media/mouse13/My Passport/corotation/buta_gal/all_table_buta_rad_astrofyz.csv',
            path='/home/mouse13/corotation/clear_outer', **kwargs):
    """names : string of 19 digits; if list - list of dictionaries returned, else - dictionary class;
       path_table : path to table with all information for images;
       path : path to dir with images (default: /home/mouse13/corotation/clear_outer);
       types : any of [obj, aper, cat, seg, real];
       bands : any of [g, i, r, u, z]."""
    dict_type = {'obj': 'objects', 'aper': 'apertures', 'cat': 'catalog', 'seg': 'segmentation'}
    if bands == 'all':
        bands = ['g', 'i', 'r', 'u', 'z']
    if types == 'all':
        types = ['obj', 'cat', 'seg', 'real']
    images = []

    all_table = pd.read_csv(path_table)

    for name in names:
        image = ImageClass()
        for prop_name in ['name', 'objid14', 'ra', 'dec']:
            image.prop(prop_name, data=all_table.loc[all_table.objid14 == int(name), [prop_name]].values[0][0])
        for band in bands:
            image[band] = ImageClass()

            for prop_name in ['name', 'objid14', 'ra', 'dec']:
                image[band].prop(prop_name, data=image[prop_name])
            for prop_name in ['gain', 'kk', 'airmass', 'seeing', 'aa', 'petroRad', 'petroR50']:
                image[band].prop(prop_name, data=all_table.loc[all_table.objid14 == int(name),
                                                               [prop_name+'_{}'.format(band)]].values[0][0])

            for tp in types:
                if tp != 'real':
                    fname = '/'.join([path, 'se_frames', band, tp, band+name+'-'+dict_type[tp]+'.fits'])
                else:
                    fname = '/'.join([path, band, 'stamps128', band+name+'.fits'])

                if tp != 'cat':
                    image[band].prop(property_name=tp, data=fits.open(fname)[0].data)
                    image[band].prop(property_name=tp+'.header', data=fits.open(fname)[0].header)
                else:
                    image[band].prop(property_name=tp, data=fits.open(fname))
                fits.open(fname).close()
        if len(names) == 1:
            images = image
        else:
            images.append(image)

    return images
