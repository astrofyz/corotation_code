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
    plt.close()


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
        if len(isolist.eps) < 1:
            image.prop(['eps', 'pa'], data=[0., 0.])
        else:
            image.prop(['eps', 'pa'], data=[isolist.eps[-1], isolist.pa[-1]])
    elif property_name:
        if len(isolist.eps) < 1:
            image.prop(['eps', 'pa'], data=[0., 0.])
        else:
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
        try:
            ellipse_fit(image, maxgerr=True)
            eps = image['eps']
        except:
            eps = 0.
    else:
        eps = image['eps']

    # print(step, image['r.max.pix'])
    a = np.arange(step, image['r.max.pix'], step)
    b = a*np.sqrt(1 - eps**2)

    annulae = []
    for i in range(1, len(a)):
        annulae.append(EllipticalAnnulus((xc, yc), a[i-1], a[i], b[i], theta=theta))

    if 'error' in kw:
        total_error = calc_total_error(image['real.mag'], image['bg'].background_rms, image['gain'])
        image.prop('total_error', data=total_error)
        table_aper = aperture_photometry(image['real.mag'], annulae, error=total_error)
        # print(len(annulae), 'ann')
        # print((table_aper['aperture_sum_3']))
        num_apers = int((len(table_aper.colnames) - 3)/2)
        intens = []
        int_error = []
        for i in range(num_apers):
            # print(table_aper['aperture_sum_' + str(i)], annulae[i].area)
            try:
                intens.append(table_aper['aperture_sum_' + str(i)] / annulae[i].area)
                int_error.append(table_aper['aperture_sum_err_'+str(i)] / annulae[i].area)
            except:
                intens.append(table_aper['aperture_sum_' + str(i)] / annulae[i].area())
                int_error.append(table_aper['aperture_sum_err_'+str(i)] / annulae[i].area())
        intens = np.array(intens).flatten()
        int_error = np.array(int_error).flatten()
        image.prop(['sb.rad.pix', 'sb', 'sb.err'], data=[(a[1:] + a[:-1]) / 2., intens, int_error])
        return (a[1:] + a[:-1]) / 2., intens, int_error
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


def find_fancy_parabola(image, **kw):
    """image : instance of ImageClass or False
    kw : rad_pix, r_max, sb_err, sb
    :returns sb.rad.fit, sb.fit, """

    if isinstance(image, mod_read.ImageClass):
        if all(['r.' not in key.lower() for key in image.keys()]):
            image.prop(['r.max.pix', 'r.min.pix', 'FD'], data=find_outer(image['seg.center'])[1:])

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

    # idxs = np.where(rad_pix < r_max)  # это очень скользко, переделай

    conv_kernel = Gaussian1DKernel(stddev=2.*np.sqrt(np.mean(sb_err)))
    sb_conv = convolve(sb, conv_kernel)
    curvature = find_curvature(rad_pix, sb_conv)

    idxs_valid = np.where(abs(curvature) < 0.1)
    min_peak = signal.argrelextrema(curvature[idxs_valid], np.less)[0][0]  #должен быть способ изящнее
    zero_abs = np.where(np.diff(np.sign(curvature[idxs_valid])))[0]
    max_peak = signal.argrelextrema(curvature[idxs_valid], np.greater)[0]
    possible_bounds = np.sort(np.concatenate([zero_abs, max_peak]))

    # with figure() as fig:
    #     plt.plot(rad_pix, sb_conv)
    #     plt.plot(rad_pix, sb)
    #     plt.axvline(rad_pix[idxs_valid][min_peak])
    #     plt.vlines(rad_pix[idxs_valid][zero_abs], color='k', ymin=min(sb), ymax=max(sb))
    #     plt.vlines(rad_pix[idxs_valid][max_peak], color='k', ymin=min(sb), ymax=max(sb))
    #     plt.show()
    #
    # with figure() as fig:
    #     plt.plot(rad_pix, curvature)
    #     plt.axhline(0.)
    #     plt.plot(np.linspace(rad_pix[0], rad_pix[-1], int(len(rad_pix)*2)),
    #              interp1d(rad_pix, curvature)(np.linspace(rad_pix[0], rad_pix[-1], int(len(rad_pix)*2))))
    #     plt.axvline(rad_pix[idxs_valid][min_peak])
    #     plt.vlines(rad_pix[idxs_valid][zero_abs], color='k', ymin=min(curvature), ymax=max(curvature))
    #     plt.vlines(rad_pix[idxs_valid][max_peak], color='k', ymin=min(curvature), ymax=max(curvature))
    #     plt.show()

    try:
        low = idxs_valid[0][possible_bounds[np.searchsorted(possible_bounds, min_peak)-1]]
        top = idxs_valid[0][possible_bounds[np.searchsorted(possible_bounds, min_peak)]]
    except:
        try:
            low = idxs_valid[0][signal.argrelextrema(abs(np.gradient(sb_conv[idxs_valid])), np.greater)[0]]
            top = idxs_valid[0][signal.argrelextrema(abs(np.gradient(sb_conv[idxs_valid])), np.greater)[1]]
            print('WARNING: interval of parabola fitting was found using gradient of sb_conv')
        except:
            low = idxs_valid[0]
            top = idxs_valid[-1]
            print("WARNING: interval of parabola fitting wasn't found")

    min_peak = idxs_valid[0][min_peak]

    if (min_peak > low)&(min_peak < top):
        dist = min([min_peak-low, top-min_peak])
        low = min_peak-dist
        top = min_peak+dist

    fit_r = rad_pix[low:top+1]
    fit_sb = sb[low:top+1]
    p = np.poly1d(np.polyfit(fit_r, fit_sb, deg=2))

    if 'plot' in kw:
        f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 2]}, sharex=True, figsize=(8, 10))
        ax1.plot(rad_pix, sb, color='darkred', lw=1, label='profile')
        ax1.plot(rad_pix, sb_conv, color='darkmagenta', alpha=0.2, lw=6, label='convolved profile')
        ax1.plot(fit_r, p(fit_r), color='k', label='approx')
        ax1.axvline(rad_pix[low], color='y')
        ax1.axvline(rad_pix[top], color='g')
        ax1.axvline(rad_pix[min_peak], color='red')
        ax1.set_xlabel('r (arcsec)')
        ax1.set_ylabel('$\mu \quad (mag\:arcsec^{-2})$')
        ax1.legend()
        ax1.set_ylim(max(sb), min(sb))
        ax2.scatter(rad_pix, abs(curvature), s=14, label='|curvature|')
        ax2.scatter(rad_pix, curvature, s=14, label='curvature')
        ax2.axhline(0.)
        ax2.legend()
        plt.grid()
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


def find_parabola(image, **kw):
    """image : instance of ImageClass or False
    kw : rad_pix, r_max, sb_err, sb
    :returns sb.rad.fit, sb.fit, """

    if isinstance(image, mod_read.ImageClass):
        if all(['r.' not in key.lower() for key in image.keys()]):
            image.prop(['r.max.pix', 'r.min.pix', 'FD'], data=find_outer(image['seg.center'])[1:])

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

    if 'plot' in kw:
        f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 2]}, sharex=True, figsize=(8, 10))
        ax1.plot(rad_pix, sb, color='darkred', lw=1, label='profile')
        ax1.plot(rad_pix, sb_conv, color='darkmagenta', alpha=0.2, lw=6, label='convolved profile')
        ax1.plot(fit_r, p(fit_r), color='k', label='approx')
        ax1.axvline(rad_pix[low])
        ax1.axvline(rad_pix[top])
        ax1.set_xlabel('r (arcsec)')
        ax1.set_ylabel('$\mu \quad (mag\:arcsec^{-2})$')
        ax1.legend()
        ax1.set_ylim(max(sb), min(sb))
        ax2.scatter(rad_pix, abs(curvature), s=14, label='|curvature|')
        ax2.scatter(rad_pix, curvature, s=14, label='curvature')
        ax2.axhline(0.)
        ax2.legend()
        plt.grid()
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
    width : width of slit"""

    if all(['r.' not in key.lower() for key in image.keys()]):
        image.prop(['r.max.pix', 'r.min.pix', 'FD'], data=find_outer(image['seg.center'])[1:])

    slits = []
    errors = []
    centre = np.array([int(dim / 2) for dim in np.shape(image['real.center'])])
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

        table_par = aperture_photometry(image['real.mag'], apertures_par, error=image['total_error'])
        table_per = aperture_photometry(image['real.mag'], apertures_per, error=image['total_error'])

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
    image.prop('slit.max', data=slits[idx][np.argmax([sum(abs(slits[idx][0])), sum(abs(slits[idx][1]))])])
    image.prop('slit.min', data=slits[idx][np.argmin([sum(abs(slits[idx][0])), sum(abs(slits[idx][1]))])])
    image.prop('slit.max.err', data=errors[idx][np.argmax([sum(abs(slits[idx][0])), sum(abs(slits[idx][1]))])])
    image.prop('slit.min.err', data=errors[idx][np.argmin([sum(abs(slits[idx][0])), sum(abs(slits[idx][1]))])])
    image.prop('angle.max', data=pa_space[idx])
    return rad, slits




