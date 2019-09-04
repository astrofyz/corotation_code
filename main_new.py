from mod_read import *
from mod_analysis import *
import pandas as pd
from scipy.interpolate import splrep, splev
from scipy.ndimage import shift
from astropy.wcs import wcs
from scipy.signal import argrelextrema
import scipy.signal as signal
from mpl_toolkits import mplot3d
import csv
import numpy.ma as ma
from astropy.convolution import Gaussian1DKernel, convolve
import os
from contextlib import contextmanager

table_path = '/media/mouse13/My Passport/corotation/buta_gal/all_table_buta_rad_astrofyz.csv'
im_path = '/media/mouse13/My Passport/corotation/buta_gal/image'
out_path = '/media/mouse13/My Passport/corotation_code/data/check_fourier/'

images = make_images(names=['587739707948204093'], bands='all', types='all', path=im_path)

@contextmanager
def figure(*a, **kw):
    fig = plt.figure()
    yield fig
    if 'title' in kw:
        plt.title(kw.get('title'))
    if 'xlabel' in kw:
        plt.xlabel(kw.get('xlabel'))
    if 'ylabel' in kw:
        plt.xlabel(kw.get('ylabel'))
    plt.show()

# with figure() as fig:
#     plt.imshow(images[0]['r']['obj'], origin='lower', cmap='Greys', norm=ImageNormalize(stretch=LogStretch()))
# print(images['r'].prop(['x.real', 'y.real']))
# print(images['r']['y.real']

# with figure() as fig:
#     plt.imshow(images['r']['real'], origin='lower', cmap='Greys', norm=ImageNormalize(stretch=LogStretch()))
#
# with figure() as fig:
#     plt.imshow(images['r']['real.mag'], origin='lower', cmap='Greys', norm=ImageNormalize(stretch=LinearStretch()))


# print([images[band]['bg'].background_rms_median for band in ['g', 'i', 'r', 'u', 'z']])

find_outer(images['r']['mask.center'], plot=True)

