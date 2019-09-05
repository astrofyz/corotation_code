import numpy as np
import pandas as pd
from astropy.wcs import wcs
from scipy.ndimage import shift
from mod_analysis import *
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.filters import gaussian_filter, median_filter
from astropy.io import fits
from astropy.visualization import LogStretch, LinearStretch
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
from mod_prep_im import *


class ImageClass(dict):
    """Class to store images and additional data"""
    def __init__(self): #можно инициировать какими-нибудь разумными значениями что-нибудь, если что-то отсутствует. например, радиус.
        self['name'] = 'name'
        self['objid'] = 'objid dr7'
        self['ra'] = 0.
        self['dec'] = 0.
        self = dict()  # а как сделать нормально?

    def prop(self, property_name, **kw):
        if 'data' in kw:
            if isinstance(property_name, list):
                for name, i in zip(property_name, range(len(property_name))):
                    self[name] = kw.get('data')[i]  #check if len(data) correspond to len(property_name)
            else:
                self[property_name] = kw.get('data')

        # if any(['+', '-', '*', '/']) in property_name:
            #write arithmetical expressions parser

        if isinstance(property_name, str):
            if property_name not in self:
                print('Error: Property keyword doesn\'t exist')
            else:
                return self[property_name]
        else:
            # print(property_name, self.keys())
            # if all(property_name) not in self.keys():
            #     print('Error: Property keywords doesn\'t exist')
            # else:
            return [self[name] for name in property_name]

    def plot_hist_r(self):
        if all(['r.' not in key.lower() for key in self.keys()]):
            self.prop(['r.hist.pix', 'r.max.pix', 'r.min.pix', 'FD'], data=find_outer(self))
        plt.figure()
        r_edges = np.arange(np.amin(self['r.min.pix']), np.amax(self['r.max.pix']), self['FD'])
        plt.hist(self['r.hist.pix'], bins=r_edges, density=True, alpha=0.5, color='lightseagreen')
        plt.axvline(self['r.max.pix'], color='red', label='$r_{max}$')
        plt.axvline(self['r.min.pix'], color='darkorange', label='$r_{min}$')
        plt.axvline(self['petroRad']/0.396, color='indigo', label='petroRad')
        plt.axvline(self['petroR50']/0.396, color='green', label='petroR50')
        plt.title(self['name'])
        plt.xlabel('r (pix)')
        plt.legend()
        plt.show()


def make_images(names, bands='all', types='all',
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
                                                               [prop_name+'_{}'.format(band)]].values[0][0])  #[0][0] — only for list of names (???)

            for tp in types:
                if tp != 'real':
                    fname = '/'.join([path, 'se_frames', band, tp, band+name+'-'+dict_type[tp]+'.fits'])  # возможно это стоит изменить
                else:
                    fname = '/'.join([path, band, 'stamps128', band+name+'.fits'])

                if tp != 'cat':
                    image[band].prop(property_name=tp, data=fits.open(fname)[0].data)
                    image[band].prop(property_name=tp+'.header', data=fits.open(fname)[0].header)
                else:
                    image[band].prop(property_name=tp, data=fits.open(fname))
                fits.open(fname).close()
        images.append(image)

    max_seeing = max([image[band]['seeing'] for band in bands])
    for band in bands:
        if image[band]['seeing'] != max_seeing:
            image[band]['real'] = correct_FWHM(image[band], max_seeing)

    for image in images:
        for band in bands:
            w = wcs.WCS(image[band]['real.header'])
            image[band].prop(property_name=['x.real', 'y.real'],
                             data=np.array(w.wcs_world2pix(image[band]['ra'], image[band]['dec'], 1)).flatten())
            xc, yc = [int(dim / 2) for dim in np.shape(image[band]['real'])]
            image[band].prop('mask', data=main_obj(cat=image[band]['cat'],
                                                   mask=image[band]['seg'],
                                                   xy=image[band].prop(['x.real', 'y.real'])))
            image[band].prop('mask.center', data=shift(image[band]['mask'],
                                                       [yc-image[band]['y.real'], xc-image[band]['x.real']], mode='nearest'))
            image[band].prop('real.center', data=shift(image[band]['real'],
                                                       [yc - image[band]['y.real'], xc - image[band]['x.real']],
                                                       mode='nearest'))
            image[band].prop('seg.center', data=shift(image[band]['seg'],
                                                      [yc - image[band]['y.real'], xc - image[band]['x.real']],
                                                      mode='nearest'))
            image[band].prop('bg', data=calc_bkg(image[band]['real.center'], image[band]['seg.center'], mode='nearest'))
            image[band].prop('real.bg', data=image[band]['real.center'] - image[band]['bg'].background)
            Apix = 0.396
            image[band].prop('zp', data=-(image[band]['aa']+image[band]['kk']*image[band]['airmass']) + 2.5*np.log10(Apix))
            image[band].prop('real.mag', data=to_mag(image=image[band]['real.bg'], zp=image[band]['zp']))

    if len(images) == 1:
        return images[0]
    else:
        return images
