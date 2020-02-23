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
from photutils import EllipticalAperture, Background2D, EllipticalAnnulus, aperture_photometry, RectangularAperture, CircularAperture
from photutils.utils import calc_total_error
from scipy.interpolate import splrep, splev, UnivariateSpline, interp1d
import scipy.signal as signal
import warnings
import numpy.ma as ma
import cv2
import scipy.fftpack as fft
from astropy.convolution import Gaussian1DKernel, convolve
import os
from mod_prep_im import *
from scipy.optimize import minimize_scalar


class ImageClass(dict):
    """Class to store images and additional data"""
    def __init__(self): #можно инициировать какими-нибудь разумными значениями что-нибудь, если что-то отсутствует. например, радиус.
        self['name'] = 'name'
        self['objID'] = 'objID dr7'
        self['ra'] = 0.
        self['dec'] = 0.
        self = dict()  # а как сделать нормально?  наверное, надо делать наследование какое-то.

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

    def plot_slits(self, n_slit=1, a_stretch=5000., mag=False, rotate=True, **kw):
        if 'slits' not in self.keys():
            calc_slit(self, n_slit=n_slit, angle=self['pa'], convolve=True)

        if 'cut' in kw:
            f, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]}, figsize=(8, 12), sharex=True)
            min = int(np.min(self['slits.rad.pix']))
            max = int(np.max(self['slits.rad.pix']))
            if mag & ('real.mag' in self.keys()):
                ax1.imshow(self['real.mag'][256+min:256+max, 256+min:256+max], origin='lower', cmap='Greys',
                           extent=(min, max, min, max))
            else:
                if rotate:
                    ax1.imshow(rotate_and_scale(self['real'], self['angle.max']+np.pi/2.)[256+min:256+max, 256+min:256+max],
                               extent=(min, max, min, max), origin='lower', cmap='Greys',
                               norm=ImageNormalize(stretch=LogStretch(a=a_stretch)))
                else:
                    ax1.imshow(self['real'][256+min:256+max, 256+min:256+max],
                               extent=(min, max, min, max), origin='lower', cmap='Greys',
                               norm=ImageNormalize(stretch=LogStretch(a=a_stretch)))

        else:
            f, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]}, figsize=(8, 12))
            ax1.set_title('{}\nra={}, dec={}'.format(self['name'], np.round(self['ra'], 3), np.round(self['dec'], 3)))
            if 'real.mag' in self.keys():
                ax1.imshow(self['real.mag'], origin='lower', cmap='Greys')
            else:
                ax1.imshow(self['real'], origin='lower', cmap='Greys', norm=ImageNormalize(stretch=LogStretch()))
        # print(self['slits'])
        idx = np.argmax([sum(abs(row)) for row in self['residuals']])

        # residual = images[1]['z']['residuals'][idx]
        # rad = images[1]['z']['slits.rad.pix']
        err = np.mean(self['total_error'])*1.2*3.5
        print(self['slits.rad.pix'], self['residuals'][idx], err)
        try:
            minrad = np.min(self['slits.rad.pix'])
            maxrad = np.max(self['slits.rad.pix'])
            ring_rad = minimize_scalar(lambda x: abs(interp1d(self['slits.rad.pix'], abs(self['residuals'][idx]))(x) - err),
                                       method='Bounded', bounds=[minrad, maxrad]).x
        except:
            print('WARNING: no ring radius')
            ring_rad = self['petroR90']

        for i, slit in enumerate(self['slits']):
            if i == idx:
                ax2.plot(self['slits.rad.pix'], slit[0], color='coral', lw=1)
                ax2.plot(self['slits.rad.pix'], slit[1], color='navy', lw=1)
            else:
                ax2.plot(self['slits.rad.pix'], slit[0], color='gold', alpha=0.)
                ax2.plot(self['slits.rad.pix'], slit[1], color='dodgerblue', alpha=0.)
        ax2.grid()
        ax2.invert_yaxis()
        line_max = ax3.plot(self['slits.rad.pix'], self['residuals'][idx], color='crimson')  # add angle to label!
        for i in range(len(self['residuals'])):
            if i != idx:
                ax3.plot(self['slits.rad.pix'], self['residuals'][i], color='dodgerblue', alpha=0.2)
        ax3.set_xlabel('r, pix')
        ax3.axhline(0.)
        ax3.legend(handles=line_max, labels='pa = {}'.format(np.round(self['slits.angle'][idx], 3)))
        ax3.grid()

        if 'real.mag' in self.keys():
            xc, yc = np.array([int(dim / 2) for dim in np.shape(self['real.mag'])])
        else:
            xc, yc = np.array([int(dim / 2) for dim in np.shape(self['real'])])

        if 'cut' in kw:
            aper = CircularAperture([0, 0], abs(ring_rad))
            aper.plot(axes=ax1, lw=0.2, color='blue', label='ring')
        else:
            aper = CircularAperture([xc, yc], abs(ring_rad))
            aper.plot(axes=ax1, lw=0.2, color='blue', label='ring')

        if 'cut' in kw:
            if rotate:  #переписать это надо красиов.
                ax1.plot(self['slits.rad.pix'] * np.cos(0.),
                         self['slits.rad.pix'] * np.sin(0.), color='orange')
                ax1.plot(self['slits.rad.pix'] * np.cos(0. + np.pi / 2.),
                         self['slits.rad.pix'] * np.sin(0. + np.pi / 2.), color='navy')
            else:
                ax1.plot(self['slits.rad.pix'] * np.cos(self['angle.max']),
                         self['slits.rad.pix'] * np.sin(self['angle.max']), color='orange')
                ax1.plot(self['slits.rad.pix'] * np.cos(self['angle.max'] + np.pi / 2.),
                         self['slits.rad.pix'] * np.sin(self['angle.max'] + np.pi / 2.), color='navy')
            # ax1.plot(self['slits.rad.pix'] * np.cos(self['slits.angle'][idx]),
            #          self['slits.rad.pix'] * np.sin(self['slits.angle'][idx]), color='orange')
            # ax1.plot(self['slits.rad.pix'] * np.cos(self['slits.angle'][idx] + np.pi / 2.),
            #          self['slits.rad.pix'] * np.sin(self['slits.angle'][idx] + np.pi / 2.), color='navy')
        else:
            ax1.plot(xc+self['slits.rad.pix']*np.cos(self['slits.angle'][idx]), yc+self['slits.rad.pix']*np.sin(self['slits.angle'][idx]), color='orange')
            ax1.plot(xc+self['slits.rad.pix']*np.cos(self['slits.angle'][idx]+np.pi/2.), yc+self['slits.rad.pix']*np.sin(self['slits.angle'][idx]+np.pi/2.), color='navy')
        plt.tight_layout()
        if 'savename' in kw:
            plt.savefig(kw.get('savename'), dpi=120)
        # plt.show()
        plt.close()



def make_images(names, bands='all', types='all',
            path_table='/media/mouse13/Seagate Expansion Drive/corotation/buta_gal/all_table_buta_rad_astrofyz.csv',
            path='/home/mouse13/corotation/clear_outer', **kw):
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

    if '.csv' in path_table:
        all_table = pd.read_csv(path_table)
    if '.cat' in path_table:
        all_table = pd.read_table(path_table, sep=' ')

    if 'SE' in kw:
        for name in names:
            image = ImageClass()
            for prop_name in ['name', 'objID', 'ra', 'dec']:
                image.prop(prop_name, data=all_table.loc[all_table.objID == int(name), [prop_name]].values[0][0])
            for band in bands:
                image[band] = ImageClass()

                for prop_name in ['name', 'objID', 'ra', 'dec']:
                    image[band].prop(prop_name, data=image[prop_name])
                if 'calibration' in kw:
                    for prop_name in ['gain', 'kk', 'airmass', 'seeing', 'aa', 'petroRad', 'petroR50']:
                        image[band].prop(prop_name, data=all_table.loc[all_table.objID == int(name),
                                                                       [prop_name+'_{}'.format(band)]].values[0][0])  #[0][0] — only for list of names (???)
                else:
                    for prop_name in ['petroRad', 'petroR50', 'petroR90']:  # why do I have petroR90????
                        image[band].prop(prop_name, data=(1./0.396)*all_table.loc[all_table.objID == int(name),
                                                                                  [prop_name + '_{}'.format(band)]].values[0][0])

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
            images.append(image)

    if 'manga' in kw:
        # print(all_table.keys())
        for name in names:
            # print(name)
            image = ImageClass()
            for prop_name in ['objID', 'ra', 'dec']:  # check column names
                image.prop(prop_name, data=all_table.loc[all_table.objID == name][prop_name])
                image['objID'] = name
            img_file = fits.open(path+f'input/{name}.fits')
            if bands == None:
                bands = img_file[0].header['BANDS']
            else:
                bands = bands
            for band in bands:
                image[band] = ImageClass()
                if (types == None) or ('real' in types):
                    id_band = img_file[0].header['BANDS'].find(band)
                    image[band].prop(property_name='real', data=img_file[0].data[id_band])
                    image[band].prop(property_name='real.header', data=img_file[0].header)
                if types != None:
                    for tp in list(set(types) - set(['real'])):
                        if tp != 'real':
                            fname = '/'.join([path, 'se_input', tp, name+'-'+dict_type[tp]+'.fits'])
                        if tp != 'cat':
                            image[band].prop(property_name=tp, data=fits.open(fname)[0].data)
                            image[band].prop(property_name=tp+'.header', data=fits.open(fname)[0].header)
                        else:
                            image[band].prop(property_name=tp, data=fits.open(fname))
                        fits.open(fname).close()
                for prop_name in ['gain', 'airmass', 'petroRad', 'petroR50', 'petroR90']:
                    try:
                        image[band].prop(prop_name, data=all_table.loc[all_table.objID == int(name),
                                                                       [prop_name + '_{}'.format(band)]].values[0][0])
                    except:
                        col_err = prop_name + f'_{band}'
                        image[band].prop(prop_name, data=0.)
                        if 'warning' in kw:
                            print(f'WARNING: no column "{col_err}"; set with 0')  # лучше с None
                seg_func = lambda x: x['seg'] if 'seg' in x.keys() else None
                image[band].prop('bg', data=calc_bkg(image[band]['real'], seg_func(image[band]), mode='nearest'))
                total_error = calc_total_error(image[band]['real'], image[band]['bg'].background_rms, image[band]['gain'])
                image[band].prop('total_error', data=total_error)
                image[band].prop('real.bg', data=image[band]['real'] - image[band]['bg'].background)
                image[band]['zp'] = 22.5
                image[band]['texp'] = 1.
                image[band].prop('real.mag', data=to_mag(image=image[band]['real.bg'], zp=image[band]['zp'], texp=image[band]['texp']))
                image[band].prop('total_error_mag', data=total_error/(image[band]['real']*image[band]['real.mag']))
            images.append(image)

    if 'seeing' in kw:
        max_seeing = max([image[band]['seeing'] for band in bands])
        for band in bands:
            if image[band]['seeing'] != max_seeing:
                image[band]['real'] = correct_FWHM(image[band], max_seeing)

    if 'correction' in kw:
        for image in images:
            for band in bands:
                w = wcs.WCS(image[band]['real.header'])
                image[band].prop(property_name=['x.real', 'y.real'],
                                 data=np.array(w.wcs_world2pix(image[band]['ra'], image[band]['dec'], 1)).flatten())
                xc, yc = [int(dim / 2) for dim in np.shape(image[band]['real'])]
                image[band].prop('mask', data=main_obj(cat=image[band]['cat'],
                                                       mask=image[band]['seg'],
                                                       xy=image[band].prop(['x.real', 'y.real'])))
                # put centered data into mask/real/seg
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

    return images
