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
def figure(**kw):
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

# проверить вот эти все штуки --- роботоют
# images['r'].plot_hist_r()
# print(find_outer(images['r']))
# print(find_outer(images['r']['mask.center']))
# images['r'].prop(['r.max.pix', 'r.min.pix'], data=find_outer(images['r']['mask.center'][1:3]))
# print(images['r']['r.max.pix'])
# print(images['r']['r.min.pix'])
# images['r'].plot_hist_r()
# print(images['r']['FD'])
# print(images['r']['bg'].background_rms_median)
# print(images['r']['bg'].background_median)
# print(images['r']['bg'].background_rms)
# ellipse_fit(images['r'], maxsma=True)
# ellipse_fit(images['r'], fflag=True)
# ellipse_fit(images['r'], maxgerr=True)
# ellipse_fit(images['r'], maxgerr=True)
# for band in ['g', 'i', 'r', 'u', 'z']:
#     calc_sb(images[band], error=True)
# print(images['r']['sb.rad.pix'])
# print(images['r']['sb'])
# print(images['r']['sb.err'])

# with figure() as fig:
#     plt.title(images['name'])
#     for band in ['g', 'i', 'r', 'u', 'z']:
#         plt.plot(images[band]['sb.rad.pix'], images[band]['sb'])
#     plt.gca().invert_yaxis()

print(len(images))
image = images
for m in range(1):
    for band in ['g', 'i', 'r', 'u', 'z']:
        find_parabola(image[band])
        calc_sb(image[band], error=True)
    with figure(xlabel='r (arcsec)', ylabel='$\mu[g, i, r, u, z] \quad (mag\:arcsec^{-2})$') as fig:
        plt.title('{}\n ra={}; dec={}'.format(image['name'], np.round(image['ra'],3), np.round(image['dec'], 3)))
        plt.gca().invert_yaxis()
        for band, color in zip(['g', 'i', 'r', 'u', 'z'], ['blue', 'gold', 'r', 'm', 'g']):
            plt.plot(image[band]['sb.rad.pix']*0.396, image[band]['sb'], color=color,  label='{} : {}'''.format(band, np.round(image[band]['sb.rad.min'], 3)))
            plt.fill_between(image[band]['sb.rad.pix']*0.396, image[band]['sb']-image[band]['sb.err'], image[band]['sb']+image[band]['sb.err'], color=color,  alpha=0.2)
            plt.plot(image[band]['sb.rad.fit']*0.396, image[band]['sb.fit'], color='k')
            plt.axvline(image[band]['sb.rad.min']*0.396, color=color)
        plt.legend()
    image['r'].plot_slits(n_slit=40)
    idx = np.argmax([sum(abs(row)) for row in image['r']['residuals']])  # перенести это в функцию
    print(image['r']['pa'])
    print(image['r']['slits.angle'][idx])


# image = images[0]
# for band in ['g', 'i', 'r', 'u', 'z']:
#     find_parabola(image[band])
#     calc_sb(image[band], error=True)
#
# # сделать это методом класса, как и остальные рисунки
# with figure(xlabel='r (arcsec)', ylabel='$\mu[g, i, r, u, z] \quad (mag\:arcsec^{-2})$') as fig:
#     plt.title('{}\n ra={}; dec={}'.format(image['name'], np.round(image['ra'],3), np.round(image['dec'], 3)))
#     plt.gca().invert_yaxis()
#     for band, color in zip(['g', 'i', 'r', 'u', 'z'], ['blue', 'gold', 'r', 'm', 'g']):
#         plt.plot(image[band]['sb.rad.pix']*0.396, image[band]['sb'], color=color,  label='{} : {}'''.format(band, np.round(image[band]['sb.rad.min'], 3)))
#         plt.fill_between(image[band]['sb.rad.pix']*0.396, image[band]['sb']-image[band]['sb.err'], image[band]['sb']+image[band]['sb.err'], color=color,  alpha=0.2)
#         plt.plot(image[band]['sb.rad.fit']*0.396, image[band]['sb.fit'], color='k')
#         plt.axvline(image[band]['sb.rad.min']*0.396, color=color)
#     plt.legend()
# image['r'].plot_slits()

# calc_slit(images['r'], angle=miages['r']['pa'], n_slit=1, convolve=True)
# print(images['r'].keys())
# дальше функция фита эллипсом и другие возможные способы определить положение и размеры бара


# from scipy import ndimage, misc
# import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.gray()  # show the filtered result in grayscale
# ax1 = fig.add_subplot(121)  # left side
# ax2 = fig.add_subplot(122)  # right side
# result = ndimage.gaussian_gradient_magnitude(images[0]['r']['real.mag'], sigma=5)
# ax1.imshow(images[0]['r']['real.mag'])
# res = ax2.imshow(result, cmap='plasma')
# cbar = plt.colorbar(res)
# plt.show()

# ну это прикольно. а что с этим дальше делать?
