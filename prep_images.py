import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import affine_transform, rotate
from scipy.ndimage.filters import gaussian_filter
from astropy.io import fits
from astropy import wcs
from astropy.visualization import LogStretch, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import sigma_clipped_stats
import pandas as pd
from photutils.isophote import EllipseGeometry, Ellipse
from photutils import EllipticalAperture, Background2D, make_source_mask, EllipticalAnnulus, aperture_photometry


def read_images(name, **kwargs):
    """name - string of 19 digits
    kwargs:
    path - default: /home/mouse13/corotation/clear_outer
    type - obj, aper, cat, seg, real
    band - g, i, r, u, z"""
    dict_type = {'obj': 'objects', 'aper': 'apertures', 'cat': 'catalog', 'seg': 'segmentation'}
    if kwargs.get('path'):
        path = kwargs.get('path')
    else:
        path = '/home/mouse13/corotation/clear_outer'

    if kwargs.get('band'):
        bands = kwargs.get('band')
    else:
        bands = ['r', 'g']

    if kwargs.get('type'):
        types = kwargs.get('type')
    else:
        types = ['cat', 'obj']

    images = []
    for band in bands:
        for tp in types:
            if tp != 'real':
                fname = '/'.join([path, 'se_frames', band, tp, band+name+'-'+dict_type[tp]+'.fits'])
            else:
                fname = '/'.join([path, band, 'stamps128', band+name+'.fits'])
            images.append(fits.open(fname))
    return images


def main_obj(cat, mask, **kwargs):
    """cat - catalog file, mask - image of segmentation file
    kwargs:
    xy - list of x,y pixels of object's centre (e.g. from funsky)
    radec - sky coord ra,dec of object's centre, need wcs
    wcs - world coordinate system from header of real file"""
    # w = wcs.WCS(real[0].header)
    # ra_real, dec_real = table.loc[all_table.objid14 == int(name), ['ra', 'dec']].values[0]
    if kwargs.get('xy'):
        x_real, y_real = kwargs.get('xy')
    elif kwargs.get('radec'):
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

    idx = 1 + id_x & id_y

    if idx + 1 != 1:
        r_seg_new = np.zeros_like(mask)
        r_seg_new[:, :] = mask[:, :]
        idxs_real = np.where(mask == idx + 1)
        idxs_fake = np.where(mask == 1)
        r_seg_new[idxs_fake] = idx + 1
        r_seg_new[idxs_real] = 1
        return r_seg_new
    else:
        return mask


def rotate_and_scale(image, angle, sx, sy):
    x0, y0 = 0.5*np.array(np.shape(image))
    x1, y1 = 0.5*np.array(np.shape(image))

    # plt.figure()
    # norm = ImageNormalize(stretch=LogStretch())
    # plt.imshow(image, norm=norm, origin='lower', cmap='Greys_r')
    # plt.show()

    rot_mtx = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])  #rotation matrix
    sca_mtx = np.array([[sx, 0], [0., sy]])  # scaling matrix; probably could be replased by s
    aff_mtx = np.dot(sca_mtx, rot_mtx)

    offset = np.array([x0, y0]) - np.dot(np.array([x1, y1]), aff_mtx)
    im_res = affine_transform(image, aff_mtx.T, mode='constant', offset=offset)

    # plt.figure()
    # norm = ImageNormalize(stretch=LogStretch())
    # plt.imshow(im_res, norm=norm, origin='lower', cmap='Greys_r')
    # plt.show()

    return im_res


def common_FWHM(image, fwhm_inp, fwhm_res):
    """"image - input image
        fwhm_inp - original fwhm
        fwhm_res - result fwhm
        (for one instance now)"""
    sigma_inp = fwhm_inp / 2. / np.sqrt(2. * np.log(2.))
    sigma_res = fwhm_res / 2. / np.sqrt(2. * np.log(2.))
    sigma_f = sigma_inp / sigma_res / 2. / np.sqrt(np.pi)
    image_res = gaussian_filter(image, sigma_f)
    return image_res


def zeropoint(**kwargs):
    """
    kwargs:
    name - list of names (list!)
    table - pandas table
    band - list(!) of bands
    """
    names = kwargs.get('name')
    table = kwargs.get('table')  # как сделать глобальную таблицу константой?
    bands = kwargs.get('band')

    Apix = 0.396*np.ones(len(bands))
    t = 53.907*np.ones(len(bands))

    zp = np.zeros((len(names), len(bands)))

    i = 0
    for name in names:
        a = table.loc[table.objid14 == int(name), ['aa_'+band for band in bands]].values[0]
        k = table.loc[table.objid14 == int(name), ['kk_' + band for band in bands]].values[0]
        x = table.loc[table.objid14 == int(name), ['airmass_' + band for band in bands]].values[0]
        zp[i, :] = -(a + k*x) + 2.5*np.log10(Apix*t)
        i += 1
    return zp


def to_mag(**kwargs):
    """kwargs:
    image - image 2D array
    zp - zeropoint
    (for one instance now)"""

    texp = 53.907
    image = kwargs.get('image')
    zp = kwargs.get('zp')
    return zp-2.5*np.log10(abs(image/texp))

def ellipse_fit(**kwargs):
    """kwargs:
    cat - string of catalog (e.g. r_cat[1].data.T[0]
    image - image 2D array
    step - step between fitting ellipses (default = 0.1)
    f - initial sma = sma_catalog / f (default = 5)"""

    cat = kwargs.get('cat')  # print(r_cat[1].data.T[0]['X_IMAGE']) подавать на вход строку транспонированного каталога
    image = kwargs.get('image')
    if kwargs.get('f'):
        f = kwargs.get('f')
    else:
        f = 5

    if kwargs.get('step'):
        step = kwargs.get('step')
    else:
        step = 1.

    plt.figure('r_norm')
    norm = ImageNormalize(stretch=LogStretch())
    plt.imshow(image, norm=norm, origin='lower', cmap='Greys_r')

    x0 = cat['X_IMAGE']
    y0 = cat['Y_IMAGE']
    sma0 = cat['A_IMAGE']
    eps0 = np.sqrt(1-(cat['B_IMAGE']/cat['A_IMAGE'])**2)
    theta0 = cat['THETA_IMAGE']
    print(eps0, theta0)

    geom_inp = EllipseGeometry(x0=x0, y0=y0, sma=sma0/f, eps=eps0, pa=theta0*np.pi/180.)  # initial ellipse

    ellipse = Ellipse(image, geom_inp)
    isolist = ellipse.fit_image(step=step)

    for iso in isolist:
        x, y, = iso.sampled_coordinates()
        # print(iso.eps, iso.pa)
        plt.plot(x, y, color='red', lw=1, alpha=0.4)
    plt.show()

    return isolist.eps[-1], isolist.pa[-1]*180./np.pi


def calc_sb(image, cat, **kwargs):
    """
    image - image 2D array
    cat - string of catalog (e.g. r_cat[1].data.T[0]
    step - width of elliptical annulus
    f_max - maximal semimajor axis / sma_catalog
    """
    x0 = cat['X_IMAGE']
    y0 = cat['Y_IMAGE']

    if kwargs.get('step'):
        step = kwargs.get('step')
    else:
        step = 5.5

    if kwargs.get('f_max'):
        f_max = kwargs.get('f_max')
    else:
        f_max = 4.

    if kwargs.get('eps'):
        eps0 = kwargs.get('eps')
    else:
        eps0 = np.sqrt(1 - (cat['B_IMAGE'] / cat['A_IMAGE']) ** 2)

    if kwargs.get('theta'):
        theta0 = kwargs.get('theta')
    else:
        theta0 = cat['THETA_IMAGE']

    if kwargs.get('sma'):
        sma0 = kwargs.get('sma')
    else:
        sma0 = cat['A_IMAGE']


#     aper_inp = EllipticalAperture((geom_inp.x0, geom_inp.y0), geom_inp.sma,
#                                   geom_inp.sma*np.sqrt(1 - geom_inp.eps**2), geom_inp.pa)
#     aper_inp.plot(color='green')  # initial ellipse guess
#
    a_in = []
    a_out = []
    b_out = []
    a_in.append(step)
    a_out.append(a_in[-1]+step)
    b_out.append(a_out[-1] * np.sqrt(1 - eps0 ** 2))
    while a_out[-1] < sma0*f_max:
        a_in.append(a_in[-1]+step)
        a_out.append(a_out[-1]+step)
        b_out.append(a_out[-1]*np.sqrt(1-eps0**2))

    a_in, a_out, b_out = np.array([a_in, a_out, b_out])
    # print(a_in)

    annulae = []
    for a_in_i, a_out_i, b_out_i in zip(a_in, a_out, b_out):
        annulae.append(EllipticalAnnulus((x0, y0), a_in_i, a_out_i, b_out_i, theta=theta0))

    table_aper = aperture_photometry(image, annulae)

    # print(table_aper.colnames)
    num_apers = len(table_aper.colnames) - 3
    print('number of apertures ', num_apers)

    intens = []
    for i in range(num_apers):
        intens.append(table_aper['aperture_sum_'+str(i)][0]/annulae[i].area())
        # print('aperture_sum', table_aper['aperture_sum_'+str(i)])
        # print('area', annulae[i].area())
        # print(table_aper['aperture_sum_'+str(i)][0])
#
#     for ann in annulae:
#         ann.plot(color='gold', alpha=0.05)
#
#     #
#     print(intens_ann, len(intens_ann))
#
#     plt.figure()
#     plt.scatter((a_out+a_in)/2., intens_ann)
#     plt.show()
#
    return (a_out+a_in)/2., np.array(intens)
#
#
#
# def ellipse_fit(**kwargs):
#     """kwargs:
#     cat - string of catalog (e.g. r_cat[1].data.T[0]
#     image - image 2D array"""
#
#     cat = kwargs.get('cat')  # print(r_cat[1].data.T[0]['X_IMAGE']) подавать на вход строку транспонированного каталога
#     image = kwargs.get('image')
#
#     plt.figure('r_norm')
#     norm = ImageNormalize(stretch=LogStretch())
#     plt.imshow(image, norm=norm, origin='lower', cmap='Greys_r')
#
#     x0 = cat['X_IMAGE']
#     y0 = cat['Y_IMAGE']
#     sma0 = cat['A_IMAGE']
#     eps0 = np.sqrt(1-(cat['B_IMAGE']/cat['A_IMAGE'])**2)
#     theta0 = cat['THETA_IMAGE']
#
#     print(x0,y0)
#
#
#     geom_inp = EllipseGeometry(x0=x0, y0=y0, sma=sma0, eps=eps0, pa=theta0*np.pi/180.)  # initial ellipse
#
#     aper_inp = EllipticalAperture((geom_inp.x0, geom_inp.y0), geom_inp.sma,
#                                   geom_inp.sma*np.sqrt(1 - geom_inp.eps**2), geom_inp.pa)
#     aper_inp.plot(color='green')  # initial ellipse guess
#
#     step = 5.5
#     a_in = []
#     a_out = []
#     b_out = []
#     a_in.append(sma0/10.)
#     a_out.append(a_in[-1]+step)
#     b_out.append(a_out[-1] * np.sqrt(1 - eps0 ** 2))
#     while a_out[-1] < sma0*4.5:
#         a_in.append(a_in[-1]+step)
#         a_out.append(a_out[-1]+step)
#         b_out.append(a_out[-1]*np.sqrt(1-eps0**2))
#
#     a_in, a_out, b_out = np.array([a_in, a_out, b_out])
#
#     annulae = []
#     for a_in_i, a_out_i, b_out_i in zip(a_in, a_out, b_out):
#         annulae.append(EllipticalAnnulus((x0, y0), a_in_i, a_out_i, b_out_i, theta=theta0))
#
#     table_aper = aperture_photometry(image, annulae)
#
#     # print(table_aper.colnames)
#     num_apers = len(table_aper.colnames) - 3
#     print(num_apers)
#     #
#     intens_ann = []
#     for i in range(num_apers):
#         intens_ann.append(aperture_photometry(image, annulae)['aperture_sum_'+str(i)][0])
#
#     for ann in annulae:
#         ann.plot(color='gold', alpha=0.05)
#
#     # ellipse = Ellipse(image, geom_inp)
#     # isolist = ellipse.fit_image(maxrit = geom_inp.sma/2., step=1.5, maxsma=geom_inp.sma*3.5)
#     # #
#     # intens = []
#     # sma = []
#     # sum = 0
#     # npix0 = 0
#     # tflux0 = 0
#     # for iso in isolist:
#     #     intens.append(iso.intens)
#     #     sma.append(iso.sma)
#     #     if type(iso.npix_e) is int:
#     #         print(iso.x0, iso.y0, iso.intens, (iso.tflux_e-tflux0)/(iso.npix_e-npix0), iso.tflux_e/iso.npix_e, iso.tflux_e-tflux0)
#     #         tflux0 = iso.tflux_e
#     #         npix0 = iso.npix_e
#     #     x, y, = iso.sampled_coordinates()
#     #     plt.plot(x, y, color='red', lw=1, alpha=0.1)
#     # intens = np.array(intens)
#     # sma = np.array(sma)
#     plt.show()
#     #
#     print(intens_ann, len(intens_ann))
#
#     plt.figure()
#     plt.scatter((a_out+a_in)/2., intens_ann)
#     plt.show()
#
#     # return sma, intens


def calc_bkg(image, mask, **kwargs):
    """image - image 2D array
    mask - segmentation image
    kwargs:
    size - backfilter_size
    return:
    Background2D object"""
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, mask=mask)
    print('background', (mean, median, std))

    if kwargs.get('size'):
        size = kwargs.get('size')
    else:
        size = int(np.shape(image)[0]/4)
    bkg = Background2D(image, (size, size), filter_size=(3, 3), mask=mask)
    return bkg










