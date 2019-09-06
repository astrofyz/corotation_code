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
import scipy.optimize as opt
import warnings
import numpy.ma as ma
import cv2
import scipy.fftpack as fft
from astropy.convolution import Gaussian1DKernel, convolve
import os
import mod_read
from contextlib import contextmanager


def find_outer(image, **kwargs):
    """ image : shifted to center segmentation map with main object == 1 or instance of image class
    :returns : r.hist.pix, r.max.pix (0.99th quantile), r.min.pix (0.05th quantile), FD """

    if isinstance(image, mod_read.ImageClass):  # why import * and ImageClass is stayed not defined???
        im = image['mask.center']
    else:
        im = image
    idx_main = np.array(np.where(im == 1))
    centre = np.array([int(dim / 2) for dim in np.shape(im)])
    r = np.array([np.sqrt(np.dot(centre-idx_main.T[i], centre-idx_main.T[i])) for i in range(len(idx_main.T))])
    r_max, r_min = np.quantile(r, [0.99, 0.05])  # in pixels
    FD_bin = 2*(np.quantile(r, 0.75) - np.quantile(r, 0.25))/(len(r))**(1./3.)  # ¯\_(ツ)_/¯
    return r, r_max, r_min, FD_bin


@contextmanager
def figure():
    fig = plt.figure()
    yield fig
    plt.show()


def ellipse_fit(image, property_name=True, **kw):
    """image : instance of ImageClass (in certain band)
    **kwargs: 'maxsma', 'fflag', 'maxgerr' - for method of fitting;
            'plot' - for pictures
    """

    if all(['r.' not in key.lower() for key in image.keys()]):
        image.prop(['r.hist.pix', 'r.max.pix', 'r.min.pix'], data=find_outer(image['seg.center'])[:3])

    xc, yc = np.array([int(dim / 2) for dim in np.shape(image['real.center'])])
    eps = np.sqrt(1-(image['cat'][1].data.T[0]['B_IMAGE']/image['cat'][1].data.T[0]['A_IMAGE'])**2)
    pa = image['cat'][1].data.T[0]['THETA_IMAGE']*np.pi/180.
    # sma0 = image['petroRad']/0.396  # in pixels
    sma0 = image['petroR50']/0.396  # in pixels
    # sma0 = np.quantile(image['r.hist.pix'], 0.4)
    minsma = image['r.min.pix']
    maxsma = image['r.max.pix']
    step = 0.2
    fflag = 0.7
    maxgerr = 0.5

    geom_inp = EllipseGeometry(x0=xc, y0=yc, sma=sma0, eps=eps, pa=pa)
    aper_inp = EllipticalAperture((geom_inp.x0, geom_inp.y0), geom_inp.sma, geom_inp.sma*np.sqrt(1 - geom_inp.eps**2),
                                  geom_inp.pa)
    # чтобы построить начальный эллипс, нужно либо делать эту функцию методом класса, либо хранить всякие характеристики эллипса
    ellipse = Ellipse(image['real.center'], geom_inp)
    # добавить флагов для различных методов и играться со всеми, пока не получится результат

    if 'maxsma' in kw:
        aper_fin = EllipticalAperture((geom_inp.x0, geom_inp.y0), maxsma,
                                      maxsma * np.sqrt(1 - geom_inp.eps ** 2),
                                      geom_inp.pa)
        try:
            # warnings.simplefilter("error")
            isolist = ellipse.fit_image(step=step, maxsma=maxsma)
        except:
            print("No meaningful fit is possible with maxsma")
            isolist = []
            return 0

        if 'plot' in kw:
            with figure() as fig:
                plt.imshow(image['real.center'], origin='lower', cmap='Greys_r', norm=ImageNormalize(stretch=LogStretch()))
                aper_fin.plot(color='gold', alpha=0.5, lw=0.5)  # final ellipse guess
                aper_inp.plot(color='dodgerblue', alpha=0.5, lw=0.5)  # initial ellipse guess
                step_iso = int(len(isolist)/6.)
                for iso in isolist[::3]:
                    x, y, = iso.sampled_coordinates()
                    plt.plot(x, y, color='cyan', lw=1, alpha=0.3)

    if 'fflag' in kw:
        image_for_fit = ma.masked_array(image['real.center'], mask=np.ones_like(image['mask.center']) - image['mask.center'])
        ellipse = Ellipse(image_for_fit, geom_inp)

        try:
            isolist = ellipse.fit_image(step=step, maxgerr=maxgerr)
        except:
            print("No meaningful fit is possible with maxgerr")
            isolist = []
            return 0

        if 'plot' in kw:
            with figure() as fig:
                plt.imshow(image['real.center'], origin='lower', cmap='Greys_r',
                            norm=ImageNormalize(stretch=LogStretch()))
                aper_inp.plot(color='dodgerblue', alpha=0.5, lw=0.5)  # initial ellipse guess
                step_iso = int(len(isolist) / 6.)
                for iso in isolist[::3]:
                    x, y, = iso.sampled_coordinates()
                    plt.plot(x, y, color='cyan', lw=1, alpha=0.3)

    if 'maxgerr' in kw:
        try:
            isolist = ellipse.fit_image(step=step, maxgerr=maxgerr)
        except:
            print("No meaningful fit is possible with maxgerr")
            isolist = []
            return 0

        if 'plot' in kw:
            with figure() as fig:
                plt.imshow(image['real.center'], origin='lower', cmap='Greys_r',
                            norm=ImageNormalize(stretch=LogStretch()))
                aper_inp.plot(color='dodgerblue', alpha=0.5, lw=0.5)  # initial ellipse guess
                step_iso = int(len(isolist) / 6.)
                for iso in isolist[::3]:
                    x, y, = iso.sampled_coordinates()
                    plt.plot(x, y, color='cyan', lw=1, alpha=0.3)

    if property_name=='list':
        image.prop(['eps.list', 'pa.list'], data=[isolist.eps, isolist.pa])
    elif property_name=='single':
        image.prop(['eps', 'pa'], data=[isolist.eps[-1], isolist.pa[-1]])
    elif property_name:
        image.prop(['eps', 'pa'], data=[isolist.eps[-1], isolist.pa[-1]])


def calc_sb(image, **kw):
    """image - instance of ImageClass (in certain band)
       step - width of elliptical annulus
        f_max - maximal semimajor axis / sma_catalog
    :returns array of radii and corresponding array of surface brightnesses in rings; (in pixels and mag) + errors if bg_rms  in **kw"""

    xc, yc = np.array([int(dim / 2) for dim in np.shape(image['real.center'])])
    theta = image['cat'][1].data.T[0]['THETA_IMAGE']  #degrees???

    if all(['r.' not in key.lower() for key in image.keys()]):
        image.prop(['r.max.pix', 'r.min.pix', 'FD'], data=find_outer(image['seg.center'])[1:])
    step = image['FD']

    if 'eps' not in image:
        ellipse_fit(image, maxgerr=True)
    eps = image['eps']

    # print(step, image['r.max.pix'])
    a = np.arange(step, image['r.max.pix'], step)
    b = a*np.sqrt(1 - eps**2)

    annulae = []
    for i in range(1, len(a)):
        annulae.append(EllipticalAnnulus((xc, yc), a[i-1], a[i], b[i], theta=theta))

    if 'error' in kw:
        total_error = calc_total_error(image['real.mag'], image['bg'].background_rms, image['gain'])
        table_aper = aperture_photometry(image['real.mag'], annulae, error=total_error)
        num_apers = int((len(table_aper.colnames) - 3)/2)
        intens = []
        int_error = []
        for i in range(num_apers):
            intens.append(table_aper['aperture_sum_' + str(i)][0] / annulae[i].area())
            int_error.append(table_aper['aperture_sum_err_'+str(i)][0] / annulae[i].area())
        image.prop(['sb.rad.pix', 'sb', 'sb.err'], data=[(a[1:] + a[:-1]) / 2., np.array(intens), np.array(int_error)])
        return (a[1:] + a[:-1]) / 2., np.array(intens), np.array(int_error)
    else:
        table_aper = aperture_photometry(image['real.mag'], annulae)
        num_apers = len(table_aper.colnames) - 3
        intens = []
        for i in range(num_apers):
            intens.append(table_aper['aperture_sum_' + str(i)][0] / annulae[i].area())
        image.prop(['sb.rad.pix', 'sb'], data=[(a[1:] + a[:-1]) / 2., np.array(intens)])
        return (a[1:] + a[:-1]) / 2., np.array(intens)


def find_curvature(r, fc, **kwargs):
    rd1 = np.gradient(r)
    rd2 = np.gradient(rd1)
    sbd1 = np.gradient(fc)
    sbd2 = np.gradient(sbd1)
    curvature = (rd1 * sbd2 - sbd1 * rd2) / (rd1 ** 2 + sbd2 ** 2) ** (3. / 2.)
    return curvature


def find_parabola(image, **kw):
    """image : instance of ImageClass
    :returns"""

    if all(['r.' not in key.lower() for key in image.keys()]):
        image.prop(['r.max.pix', 'r.min.pix', 'FD'], data=find_outer(image['seg.center'])[1:])

    if all(['sb' not in key.lower() for key in image.keys()]):
        calc_sb(image, error=True)

    idxs = np.where(image['sb.rad.pix'] < image['r.max.pix'])  # это очень скользко, переделай

    conv_kernel = Gaussian1DKernel(stddev=2.*np.sqrt(max([np.mean(image['sb.err']), np.median(image['sb.err'])])))
    sb_conv = convolve(image['sb'], conv_kernel)
    curvature = find_curvature(image['sb.rad.pix'], sb_conv)[idxs]

    idxs_valid = np.where(abs(curvature) < 0.1)
    min_peak = signal.argrelextrema(curvature[idxs_valid], np.less)[0][0]  #должен быть способ изящнее
    zero_abs = np.where(np.diff(np.sign(curvature[idxs_valid])))[0]

    if min_peak > zero_abs[0]:
        low = idxs_valid[0][zero_abs[np.searchsorted(zero_abs, min_peak)-1]]
        top = idxs_valid[0][zero_abs[np.searchsorted(zero_abs, min_peak)]]
    else:
        low = np.where(abs(curvature) < 0.1)[0][0]
        top = idxs_valid[0][zero_abs[0]]
    print(low, top)

    fit_r = image['sb.rad.pix'][low:top+1]
    fit_sb = image['sb'][low:top+1]
    p = np.poly1d(np.polyfit(fit_r, fit_sb, deg=2))

    if 'plot' in kw:
        f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 2]}, sharex=True, figsize=(8, 10))
        ax1.plot(image['sb.rad.pix'], image['sb'], color='darkred', lw=1, label='profile')
        ax1.plot(image['sb.rad.pix'], sb_conv, color='darkmagenta', alpha=0.2, lw=6, label='convolved profile')
        ax1.plot(fit_r, p(fit_r), color='k', label='approx')
        ax1.axvline(image['sb.rad.pix'][low])
        ax1.axvline(image['sb.rad.pix'][top])
        ax1.set_xlabel('r (arcsec)')
        ax1.set_ylabel('$\mu \quad (mag\:arcsec^{-2})$')
        ax1.legend()
        ax1.set_ylim(max(image['sb']), min(image['sb']))
        ax2.scatter(image['sb.rad.pix'], abs(curvature), s=14, label='|curvature|')
        ax2.scatter(image['sb.rad.pix'], curvature, s=14, label='curvature')
        ax2.axhline(0.)
        ax2.legend()
        plt.grid()
        plt.show()
    # нужно что-то записать в класс. посмотреть, что нужно дальше и записать его
    image.prop(['sb.rad.fit', 'sb.fit', 'sb.rad.min'],
               data=[fit_r, p(fit_r), opt.minimize_scalar(-p, method='Bounded', bounds=[fit_r[0], fit_r[-1]]).x])
    return fit_r, p(fit_r), image['sb.rad.pix'][idxs_valid[0][-1]]  # pix vs arcsec flag








