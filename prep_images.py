from astropy.io import fits
from scipy.ndimage.interpolation import affine_transform, rotate
from scipy.ndimage.filters import gaussian_filter
import numpy as np


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




