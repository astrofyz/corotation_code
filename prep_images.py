from astropy.io import fits
from scipy.ndimage.interpolation import affine_transform, rotate
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import pandas as pd
from photutils.isophote import EllipseGeometry, Ellipse
from photutils import EllipticalAperture
import matplotlib.pyplot as plt
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy import wcs

def read_images(name, **kwargs):
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

def rotate_and_scale(image, angle, sx, sy):
    [x0, y0] = [elem[0] for elem in np.where(image == image.max())]  # max brightness of input

    rot_mtx = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]  #rotation matrix
    sca_mtx = [[sx, 0], [0., sy]]  # scaling matrix; probably could be replased by s
    aff_mtx = np.dot(rot_mtx, sca_mtx)

    im_out = affine_transform(image, aff_mtx, mode='nearest')
    [x1, y1] = [elem[0] for elem in np.where(im_out == im_out.max())]  # how calculate offset?

    offset = np.array([y0, x0]) - np.array([y1, x1])
    im_res = affine_transform(image, aff_mtx, mode='nearest', offset=offset)

    im_res = rotate(image, angle=angle)

    return im_res

# нельзя сдвигать по самому яркому, потому что там могут быть звёзды


def common_FWHM(image, fwhm_inp, fwhm_res):
    sigma_inp = fwhm_inp / 2. / np.sqrt(2. * np.log(2.))
    sigma_res = fwhm_res / 2. / np.sqrt(2. * np.log(2.))
    sigma_f = sigma_inp / sigma_res / 2. / np.sqrt(np.pi)
    image_res = gaussian_filter(image, sigma_f)
    return image_res


def zeropoint(**kwargs):
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
    texp = 53.907
    image = kwargs.get('image')
    zp = kwargs.get('zp')
    if kwargs.get('mask'):
        mask = kwargs.get('mask')[0]
        image_new = np.zeros_like(image)
        idxs = np.where(mask != 0)
        image_new[idxs] = zp-2.5*np.log10(abs(image[idxs]/texp))  # и почему там есть нули? ну пусть
        return image_new
    else:
        return zp-2.5*np.log10(abs(image/texp))


def average_sb(**kwargs):
    cat = kwargs.get('cat')  # print(r_cat[1].data.T[0]['X_IMAGE']) подавать на вход строку транспонированного каталога
    image = kwargs.get('image')

    plt.figure('r_norm')
    norm = ImageNormalize(stretch=LogStretch())
    plt.imshow(image, norm=norm, origin='lower', cmap='Greys_r')


    geom_inp = EllipseGeometry(x0=cat['X_IMAGE'], y0=cat['Y_IMAGE'], sma=cat['A_IMAGE'],
                               eps=(1-cat['B_IMAGE']/cat['A_IMAGE']), pa=cat['THETA_IMAGE']*np.pi/180.)  # initial ellipse

    aper_inp = EllipticalAperture((geom_inp.x0, geom_inp.y0), geom_inp.sma, geom_inp.sma*(1 - geom_inp.eps), geom_inp.pa)
    aper_inp.plot(color='green')  # initial ellipse guess

    ellipse = Ellipse(image, geom_inp)
    isolist = ellipse.fit_image(maxrit = geom_inp.sma/5., step=0.01, maxsma=geom_inp.sma*3.5)
    #
    intens = []
    sma = []
    # sum = 0
    # npix0 = 0
    # tflux0 = 0
    for iso in isolist:
        intens.append(iso.intens)
        sma.append(iso.sma)
        # if type(iso.npix_e) is int:
        #     print(iso.x0, iso.y0, iso.sma, iso.sma*np.sqrt(1-iso.eps**2), iso.intens, iso.npix_e, iso.tflux_e, (iso.tflux_e-tflux0)/(iso.npix_e-npix0), iso.tflux_e/iso.npix_e)
        #     tflux0 = iso.tflux_e
        #     npix0 = iso.npix_e
        x, y, = iso.sampled_coordinates()
        plt.plot(x, y, color='red', lw=1, alpha=0.1)
    intens = np.array(intens)
    sma = np.array(sma)
    plt.show()

    return sma, intens

def main_obj(name, cat, table, real):
    ra_real, dec_real = table.loc[all_table.objid14 == int(name), ['ra', 'dec']].values[0]
    # print(ra_real, dec_real)

    w = wcs.WCS(real[0].header)  # или удобнее в табличку добавить координаты в пикселях?
    # print(w.wcs.name)

    x_real, y_real = w.wcs_world2pix(ra_real, dec_real, 1)
    # print(x_real, y_real)

    delta_x = abs(cat[1].data['X_IMAGE'] - x_real)
    delta_y = abs(cat[1].data['Y_IMAGE'] - y_real)

    id_x = np.argmin(delta_x)
    id_y = np.argmin(delta_y)

    idx = 1 + id_x & id_y

    if idx + 1 != 1:
        r_seg_new = np.zeros_like(r_seg[0].data)
        r_seg_new[:, :] = r_seg[0].data[:, :]
        idxs_real = np.where(r_seg[0].data == idx + 1)
        idxs_fake = np.where(r_seg[0].data == 1)
        r_seg_new[idxs_fake] = idx + 1
        r_seg_new[idxs_real] = 1
        return r_seg_new
    else:
        return r_seg[0].data











