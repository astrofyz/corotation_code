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
from scipy.interpolate import splrep, splev, UnivariateSpline, interp1d
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
from mod_prep_im import *
import math
from skimage import exposure


def find_outer(image, **kwargs):
    """ image : shifted to center segmentation map with main object == 1 or instance of image class
    :returns : r.hist.pix, r.max.pix (0.99th quantile), r.min.pix (0.05th quantile), FD """

    if isinstance(image, mod_read.ImageClass):  # why import * and ImageClass is stayed not defined???
        im = image['mask']
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
    plt.close()


def ellipse_fit(image, property_name=True, **kw):
    """image : instance of ImageClass (in certain band)
    **kwargs: 'maxsma', 'fflag', 'maxgerr' - for method of fitting;
            'plot' - for pictures
    """

    if all(['r.' not in key.lower() for key in image.keys()]):
        image.prop(['r.hist.pix', 'r.max.pix', 'r.min.pix'], data=find_outer(image['seg'])[:3])

    xc, yc = np.array([int(dim / 2) for dim in np.shape(image['real'])])
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
    ellipse = Ellipse(image['real.mag'], geom_inp)
    # добавить флагов для различных методов и играться со всеми, пока не получится результат

    eps = np.sqrt(1 - (image['cat'][1].data.T[0]['B_IMAGE']/image['cat'][1].data.T[0]['A_IMAGE'])**2)
    pa = image['cat'][1].data.T[0]['THETA_IMAGE']*np.pi/180.

    if 'maxsma' in kw:
        aper_fin = EllipticalAperture((geom_inp.x0, geom_inp.y0), maxsma,
                                      maxsma * np.sqrt(1 - geom_inp.eps ** 2),
                                      geom_inp.pa)
        try:
            # warnings.simplefilter("error")
            isolist = ellipse.fit_image(step=step, maxsma=maxsma)
            if property_name == 'list':
                image.prop(['eps.list', 'pa.list'], data=[isolist.eps, isolist.pa])
            elif property_name == 'single':
                if len(isolist.eps) < 1:
                    image.prop(['eps', 'pa'], data=[eps, pa])
                else:
                    image.prop(['eps', 'pa'], data=[isolist.eps[-1], isolist.pa[-1]])
            elif property_name:
                if len(isolist.eps) < 1:
                    image.prop(['eps', 'pa'], data=[eps, pa])
                else:
                    image.prop(['eps', 'pa'], data=[isolist.eps[-1], isolist.pa[-1]])
        except:
            print("No meaningful fit is possible with maxsma")
            isolist = []
            image.prop(['eps', 'pa'], data=[eps, pa])
            return 0

        if 'plot' in kw:
            with figure() as fig:
                plt.imshow(image['real.mag'], origin='lower', cmap='Greys_r')
                aper_fin.plot(color='gold', alpha=0.5, lw=0.5)  # final ellipse guess
                aper_inp.plot(color='dodgerblue', alpha=0.5, lw=0.5)  # initial ellipse guess
                step_iso = int(len(isolist)/6.)
                for iso in isolist[::3]:
                    x, y, = iso.sampled_coordinates()
                    plt.plot(x, y, color='cyan', lw=1, alpha=0.3)

    if 'fflag' in kw:
        image_for_fit = ma.masked_array(image['real'], mask=np.ones_like(image['mask']) - image['mask'])
        ellipse = Ellipse(image_for_fit, geom_inp)

        try:
            isolist = ellipse.fit_image(step=step, maxgerr=maxgerr)
            if property_name == 'list':
                image.prop(['eps.list', 'pa.list'], data=[isolist.eps, isolist.pa])
            elif property_name == 'single':
                if len(isolist.eps) < 1:
                    image.prop(['eps', 'pa'], data=[eps, pa])
                else:
                    image.prop(['eps', 'pa'], data=[isolist.eps[-1], isolist.pa[-1]])
            elif property_name:
                if len(isolist.eps) < 1:
                    image.prop(['eps', 'pa'], data=[eps, pa])
                else:
                    image.prop(['eps', 'pa'], data=[isolist.eps[-1], isolist.pa[-1]])
        except:
            print("No meaningful fit is possible with maxgerr")
            isolist = []
            return 0

        if 'plot' in kw:
            with figure() as fig:
                plt.imshow(image['real'], origin='lower', cmap='Greys_r',
                            norm=ImageNormalize(stretch=LogStretch()))
                aper_inp.plot(color='dodgerblue', alpha=0.5, lw=0.5)  # initial ellipse guess
                step_iso = int(len(isolist) / 6.)
                for iso in isolist[::3]:
                    x, y, = iso.sampled_coordinates()
                    plt.plot(x, y, color='cyan', lw=1, alpha=0.3)

    if 'maxgerr' in kw:
        try:
            isolist = ellipse.fit_image(step=step, maxgerr=maxgerr)
            if property_name == 'list':
                image.prop(['eps.list', 'pa.list'], data=[isolist.eps, isolist.pa])
            elif property_name == 'single':
                if len(isolist.eps) < 1:
                    image.prop(['eps', 'pa'], data=[eps, pa])
                else:
                    image.prop(['eps', 'pa'], data=[isolist.eps[-1], isolist.pa[-1]])
            elif property_name:
                if len(isolist.eps) < 1:
                    image.prop(['eps', 'pa'], data=[eps, pa])
                else:
                    image.prop(['eps', 'pa'], data=[isolist.eps[-1], isolist.pa[-1]])
        except:
            print("No meaningful fit is possible with maxgerr")
            isolist = []
            return 0

        if 'plot' in kw:
            with figure() as fig:
                plt.imshow(image['real'], origin='lower', cmap='Greys_r',
                            norm=ImageNormalize(stretch=LogStretch()))
                aper_inp.plot(color='dodgerblue', alpha=0.5, lw=0.5)  # initial ellipse guess
                step_iso = int(len(isolist) / 6.)
                for iso in isolist[::3]:
                    x, y, = iso.sampled_coordinates()
                    plt.plot(x, y, color='cyan', lw=1, alpha=0.3)


def calc_sb(image, error=True, circ_aper=False, **kw):
    """image - instance of ImageClass (in certain band)
       step - width of elliptical annulus
        f_max - maximal semimajor axis / sma_catalog
    :returns array of radii and corresponding array of surface brightnesses in rings; (in pixels and mag) + errors if bg_rms  in **kw"""

    xc, yc = np.array([int(dim / 2) for dim in np.shape(image['real'])])
    theta = image['cat'][1].data.T[0]['THETA_IMAGE']*np.pi/180.  #degrees???

    seg_func = lambda x: x['seg'] if 'seg' in x.keys() else x['seg']

    if all(['r.' not in key.lower() for key in image.keys()]):
        image.prop(['r.max.pix', 'r.min.pix', 'FD'], data=find_outer(seg_func(image))[1:])

    if 'step' in kw:
        step = kw['step']
    else:
        step = find_outer(seg_func(image)[1:])[-1]*0.8

    if 'eps' not in image:
        try:
            # ellipse_fit(image, maxgerr=True)
            eps = np.sqrt(1 - (image['cat'][1].data.T[0]['B_IMAGE'] / image['cat'][1].data.T[0]['A_IMAGE']) ** 2)
            image['eps'] = eps
        except:
            eps = 0.
            image['eps'] = 0.
    else:
        eps = image['eps']

    if circ_aper:
        eps = 0.

    a = np.arange(step, image['r.max.pix'], step)
    b = a*np.sqrt(1 - eps**2)

    annulae = []
    for i in range(1, len(a)):
        annulae.append(EllipticalAnnulus((xc, yc), a[i-1], a[i], b[i], theta=theta))

    # plt.figure()
    # plt.imshow(image['real.mag'], origin='lower', cmap='Greys')
    # for ann in annulae[::2]:
    #     ann.plot(lw=0.1)
    # plt.show()
    # plt.close()
    # print('fig end')

    if error:
        total_error = calc_total_error(image['real'], image['bg'].background_rms, image['gain'])

        image.prop('total_error', data=total_error)
        if 'adjust_contrast' in kw:
            v_min, v_max = np.percentile(image['real.bg'], (kw['adjust_contrast'], 1-kw['adjust_contrast']))
            image_work = exposure.rescale_intensity(image['real.bg'], in_range=(v_min, v_max))
        else:
            image_work = image['real.bg']

        table_aper = aperture_photometry(image_work, annulae, error=image['total_error'])
        num_apers = int((len(table_aper.colnames) - 3)/2)
        intens = []
        int_error = []
        for i in range(num_apers):
            try:
                intens.append(table_aper['aperture_sum_' + str(i)] / annulae[i].area)
                int_error.append(table_aper['aperture_sum_err_'+str(i)] / (annulae[i].area))
            except:
                intens.append(table_aper['aperture_sum_' + str(i)] / annulae[i].area())
                int_error.append(table_aper['aperture_sum_err_'+str(i)] / np.sqrt(annulae[i].area()))
        intens = np.array(intens).flatten()
        int_error = np.array(int_error).flatten()
        image.prop(['sb.rad.pix', 'sb', 'sb.err'], data=[(a[1:] + a[:-1]) / 2., intens, int_error])
        image.prop(['sb.mag', 'sb.err.mag'],
                   data=[to_mag(intens, zp=image['zp'], texp=image['texp']), abs(2.5*np.log10(1+int_error/intens))])
        return (a[1:] + a[:-1]) / 2., intens, int_error
    else:
        table_aper = aperture_photometry(image['real.bg'], annulae)
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
    """image : instance of ImageClass or False
    kw : rad_pix, r_max, sb_err, sb
    :returns sb.rad.fit, sb.fit, """

    if isinstance(image, mod_read.ImageClass):
        if all(['r.' not in key.lower() for key in image.keys()]):
            seg_func = lambda x: x['seg'] if 'seg' in x.keys() else x['seg']
            image.prop(['r.max.pix', 'r.min.pix', 'FD'], data=find_outer(seg_func(image))[1:])

        if all(['sb' not in key.lower() for key in image.keys()]):
            calc_sb(image, error=True)

        rad_pix = image['sb.rad.pix']
        r_max = image['r.max.pix']
        sb_err = image['sb.err']
        sb = image['sb']

    else:
        rad_pix = kw['rad_pix']
        r_max = kw['r_max']
        sb_err = kw['sb_err']
        sb = kw['sb']

    idxs = np.where(rad_pix < r_max)  # это очень скользко, переделай

    conv_kernel = Gaussian1DKernel(stddev=2.*np.sqrt(max([np.mean(sb_err), np.median(sb_err)])))
    sb_conv = convolve(sb, conv_kernel)
    curvature = find_curvature(rad_pix, sb_conv)[idxs]

    idxs_valid = np.where(abs(curvature) < 0.1)
    min_peak = signal.argrelextrema(curvature[idxs_valid], np.less)[0][0]  #должен быть способ изящнее
    zero_abs = np.where(np.diff(np.sign(curvature[idxs_valid])))[0]

    # try:
    if (min_peak > zero_abs[0]) & (min_peak < zero_abs[-1]):
        low = idxs_valid[0][zero_abs[np.searchsorted(zero_abs, min_peak)-1]]
        top = idxs_valid[0][zero_abs[np.searchsorted(zero_abs, min_peak)]]
    else:
        low = np.where(abs(curvature) < 0.1)[0][0]
        top = idxs_valid[0][zero_abs[0]]
    # except:


    # print(low, top)

    fit_r = rad_pix[low:top+1]
    fit_sb = sb[low:top+1]
    p = np.poly1d(np.polyfit(fit_r, fit_sb, deg=2))

    def func(x, a, b, c):
        return a * np.log10(b * x) + c

    popt, pcorr = opt.curve_fit(func, rad_pix, sb)

    if 'plot' in kw:
        f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 2]}, sharex=True, figsize=(8, 10))
        ax1.plot(rad_pix, sb, color='darkred', lw=1, label='profile')
        ax1.plot(rad_pix, sb_conv, color='darkmagenta', alpha=0.2, lw=6, label='convolved profile')
        ax1.plot(fit_r, p(fit_r), color='k', label='approx')
        ax1.axvline(rad_pix[low])
        ax1.axvline(rad_pix[top])
        ax1.set_xlabel('r (pix)')
        ax1.set_ylabel('$\mu \quad (mag\:arcsec^{-2})$')
        ax1.set_title(f"{image['objID']}\n{kw['band']}")
        ax1.legend()
        ax1.set_ylim(max(sb), min(sb))
        # ax2.scatter(rad_pix, abs(curvature), s=14, label='|curvature|')
        # ax2.scatter(rad_pix, curvature, s=14, label='curvature')
        ax2.scatter(rad_pix, sb-func(rad_pix, *popt), marker='.')
        ax2.axhline(0.)
        ax2.legend()
        plt.grid()
        plt.savefig(kw['savename'])
        plt.show()
        plt.close()
    # нужно что-то записать в класс. посмотреть, что нужно дальше и записать его
    try:
        r_min = opt.minimize_scalar(-p, method='Bounded', bounds=[fit_r[0], fit_r[-1]]).x
    except:
        r_min = 0.
        print("couldn't find parabolda minimum")
    if isinstance(image, mod_read.ImageClass):
        image.prop(['sb.rad.fit', 'sb.fit', 'sb.rad.min'],
               data=[fit_r, p(fit_r), r_min])
    return fit_r, p(fit_r), r_min  #rad_pix[idxs_valid[0][-1]]  # pix vs arcsec flag


def calc_slit(image, n_slit=1, angle=0., step=1.2, width=3.5, **kw):
    """image : instance of ImageClass
    n_slit : number of slits
    angle : starting angle
    step : distance between two succesive apertures along radius
    width : width of slit
    kw: convolve: background is required"""
    if all(['r.' not in key.lower() for key in image.keys()]):
        if ('seg' in image.keys()) & ('petro' not in kw):  #change centered to without 
            try:
                image.prop(['r.max.pix', 'r.min.pix', 'FD'], data=find_outer(image['seg'])[1:])
            except:
                image['r.max.pix'] = image['petroR90'] * 3
        elif ('seg' not in image.keys()) or ('petro' in kw):
            image['r.max.pix'] = image['petroR90']  # or petroR90 * 2.; check .prop()

    slits = []
    errors = []
    centre = np.array([int(dim / 2) for dim in np.shape(image['real'])])

    image_work = np.zeros_like(image['real'])

    if 'mag' in kw:
        image_work[:, :] = image['real.mag'][:, :]
    else:
        image_work[:, :] = image['real'][:, :]

    if 'mask' in kw:
        if ('seg' in image.keys()) & ('cat' in image.keys()):
            try:
                main_obj_mask = main_obj(image['cat'], image['seg'], xy=centre)
                image_work[main_obj_mask == 0] = image['bg'].background[main_obj_mask == 0]
            except:
                print('WARNING; main_obj_mask')

    if n_slit > 1:
        pa_space = np.linspace(0, np.pi/2., n_slit)
    else:
        pa_space = np.array([angle])

    # print(pa_space[0], step, step*np.cos(pa_space[i]), step*np.sin(pa_space[i]))

    for i in range(n_slit):
        slit_par = list([centre])
        slit_per = list([centre])
        dr = 0
        step_par = np.array([step*np.cos(pa_space[i]), step*np.sin(pa_space[i])])
        step_per = np.array([step*np.cos(pa_space[i]+np.pi/2.), step*np.sin(pa_space[i]+np.pi/2.)])
        j = 1
        while dr < image['r.max.pix']:
            slit_par.append(centre + j * step_par)
            slit_par.append(centre - j * step_par)
            slit_per.append(centre + j * step_per)
            slit_per.append(centre - j * step_per)
            # print(centre+j*step_par)
            dr += step
            j += 1

        slit_par, slit_per = np.array([slit_par, slit_per])
        # print(slit_par, np.shape(slit_par))
        r_par = [slit_par[i] for i in np.lexsort([slit_par.T[0], slit_par.T[1]])]
        r_per = [slit_per[i] for i in np.lexsort([slit_par.T[0], slit_par.T[1]])]

        apertures_par = RectangularAperture(r_par, width, step, pa_space[i])
        apertures_per = RectangularAperture(r_per, width, step, pa_space[i] + np.pi / 2.)

        if 'mag' in kw:  # чо???
            table_par = aperture_photometry(image_work, apertures_par, error=image['total_error'])
            table_per = aperture_photometry(image_work, apertures_per, error=image['total_error'])
        else:
            table_par = aperture_photometry(image_work, apertures_par, error=image['total_error'])
            table_per = aperture_photometry(image_work, apertures_per, error=image['total_error'])

        # print(table_par)

        area = step * width

        intense_par = [elem / area for elem in table_par['aperture_sum']]
        intense_per = [elem / area for elem in table_per['aperture_sum']]

        error_par = [elem / area for elem in table_par['aperture_sum_err']]
        error_per = [elem / area for elem in table_per['aperture_sum_err']]

        if 'convolve' in kw:
            kernel = Gaussian1DKernel(stddev=image['bg'].background_rms_median)
            intense_par = convolve(np.array(intense_par), kernel)
            intense_per = convolve(np.array(intense_per), kernel)

        slits.append([intense_par, intense_per])
        errors.append([error_par, error_per])

    rad = np.array([k*step for k in range(-j+1, j, 1)])

    slits = np.array(slits)
    image.prop('slits.rad.pix', data=rad)
    image.prop('slits', data=slits)
    image.prop('residuals', data=[(slit[0]-slit[1]) for slit in slits])
    image.prop('slits.angle', data=pa_space)  # проверь, что размерность совпадает, а не в два раза меньше!

    idx = np.argmax([sum(abs(row)) for row in image['residuals']])
    image.prop('slit.min', data=slits[idx][np.argmax([sum(abs(slits[idx][0])), sum(abs(slits[idx][1]))])])
    image.prop('slit.max', data=slits[idx][np.argmin([sum(abs(slits[idx][0])), sum(abs(slits[idx][1]))])])
    image.prop('slit.min.err', data=errors[idx][np.argmax([sum(abs(slits[idx][0])), sum(abs(slits[idx][1]))])])
    image.prop('slit.max.err', data=errors[idx][np.argmin([sum(abs(slits[idx][0])), sum(abs(slits[idx][1]))])])
    image.prop('angle.max', data=pa_space[idx])
    return rad, slits


def fourier_harmonics(image, harmonics=[1, 2, 3, 4], sig=5, plot=True, **kw):
    image_work = image['real.mag']
    value = np.sqrt(((image_work.shape[0] / 2.0) ** 2.0) + ((image_work.shape[1] / 2.0) ** 2.0))

    polar_image = cv2.linearPolar(image_work, (image_work.shape[0] / 2, image_work.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)
    # print(type(image_work), np.shape(image_work))

    norm = ImageNormalize(stretch=LogStretch())
    # plt.figure()
    # plt.imshow(polar_image, origin='lower', cmap='Greys')
    # ticks = np.linspace(0, image_work.shape[1], 10)  # y or x len in case of non-square image?
    # plt.yticks(ticks, [str(np.round(tick * 2. * np.pi / image_work.shape[1], 1)) for tick in ticks])
    # plt.show()

    # r_range = np.linspace(0, nx, 50)
    # phi_range = np.linspace(0, 2 * np.pi, 150)

    if all(['r.' not in key.lower() for key in image.keys()]):
        if ('seg' in image.keys()) & ('petro' not in kw):  #change centered to without 
            try:
                image.prop(['r.max.pix', 'r.min.pix', 'FD'], data=find_outer(image['seg'])[1:])
            except:
                image['r.max.pix'] = image['petroR90'] * 3
        elif ('seg' not in image.keys()) or ('petro' in kw):
            image['r.max.pix'] = image['petroR90']  # or petroR90 * 2.; check .prop()

    len_I = int(math.ceil(image['r.max.pix']))
    I = np.zeros((len(harmonics), len_I))

    j = 0
    for r in range(sig, len_I-sig):
        # data_r = polar_image[:, r]
        data_r = [np.mean(row) for row in polar_image[:, r-sig:r+sig]]
        data_fft = fft.dct(data_r)
        i = 0
        for harmonic in harmonics:
            I[i][j] = abs(data_fft[harmonic])/abs(data_fft[0])
            i += 1
        j += 1
        # if r == 40:
        #     freq = fft.fftfreq(len(data_r), 1. / len(data_r))
        #     nx = image.shape[0]
        #     plt.figure()
        #     plt.plot(np.linspace(0, nx, nx) * 2. * np.pi / nx, polar_image[:, r])
        #     plt.plot(np.linspace(0, nx, nx) * 2. * np.pi / nx, 1. / nx * sum(
        #         [data_fft[i] * np.cos(freq[i] * np.linspace(0, nx, nx) * np.pi / nx) for i in range(len(data_fft))]))
        #     plt.show()
    r = range(sig, len_I-sig)
    if plot:
        plt.figure()
        for i in range(len(harmonics)):
            plt.plot(r, I[i][:len(r)], label=harmonics[i])
        plt.legend()
        if 'savename' in kw:
            plt.savefig(kw['savename'])
        plt.show()
        plt.close()
    image.prop('fourier.harm', data=I)
    return I

