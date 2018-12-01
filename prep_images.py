import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.filters import gaussian_filter, median_filter
from astropy.io import fits
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
# from astropy.stats import sigma_clipped_stats
from photutils.isophote import EllipseGeometry, Ellipse
from photutils import EllipticalAperture, Background2D, EllipticalAnnulus, aperture_photometry, RectangularAperture
from scipy.interpolate import splrep, splev
import scipy.signal as signal


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

    idx = (id_x & id_y) + 1

    if idx != 1:
        r_seg_new = np.zeros_like(mask)
        r_seg_new[:, :] = mask[:, :]
        idxs_real = np.where(mask == idx)
        idxs_fake = np.where(mask == 1)
        r_seg_new[idxs_fake] = idx + 1
        r_seg_new[idxs_real] = 1
        return r_seg_new
    else:
        return mask


def rotate_and_scale(image, angle, sx, sy):
    x0, y0 = 0.5*np.array(np.shape(image))
    x1, y1 = 0.5*np.array(np.shape(image))

    rot_mtx = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])  #rotation matrix
    sca_mtx = np.array([[sx, 0], [0., sy]])  # scaling matrix; probably could be replased by s
    aff_mtx = np.dot(sca_mtx, rot_mtx)

    offset = np.array([x0, y0]) - np.dot(np.array([x1, y1]), aff_mtx)
    im_res = affine_transform(image, aff_mtx.T, mode='nearest', offset=offset)

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

    if kwargs.get('step'):
        step = kwargs.get('step')
    else:
        step = 1.

    plt.figure('r_norm')
    norm = ImageNormalize(stretch=LogStretch())
    plt.imshow(image, norm=norm, origin='lower', cmap='Greys_r')

    x0 = kwargs.get('x')
    y0 = kwargs.get('y')
    sma0 = kwargs.get('sma')
    eps0 = kwargs.get('eps')
    theta0 = kwargs.get('theta')

    if kwargs.get('f'):
        f = kwargs.get('f')
        geom_inp = EllipseGeometry(x0=x0, y0=y0, sma=sma0 / f, eps=eps0, pa=theta0 * np.pi / 180.)  # initial ellipse

    if kwargs.get('rmin'):
        rmin = kwargs.get('rmin')
        geom_inp = EllipseGeometry(x0=x0, y0=y0, sma=rmin, eps=eps0, pa=theta0 * np.pi / 180.)  # initial ellipse

    aper_inp = EllipticalAperture((geom_inp.x0, geom_inp.y0), geom_inp.sma, geom_inp.sma*np.sqrt(1 - geom_inp.eps**2),
                                  geom_inp.pa)
    aper_inp.plot(color='red', alpha=0.3)  # initial ellipse guess

    # aper_inp = EllipticalAperture((geom_inp.x0, geom_inp.y0), f*geom_inp.sma, f*geom_inp.sma*np.sqrt(1 - geom_inp.eps**2),
    #                               geom_inp.pa)
    # aper_inp.plot(color='gold')  # final ellipse guess

    ellipse = Ellipse(image, geom_inp)

    if kwargs.get('rmax'):
        maxsma = kwargs.get('rmax')

        aper_inp = EllipticalAperture((geom_inp.x0, geom_inp.y0), maxsma,
                                      maxsma * np.sqrt(1 - geom_inp.eps ** 2),
                                      geom_inp.pa)
        aper_inp.plot(color='gold', alpha=0.3)  # final ellipse guess

        isolist = ellipse.fit_image(step=step, maxsma=maxsma)

        for iso in isolist:
            x, y, = iso.sampled_coordinates()
            plt.plot(x, y, color='cyan', lw=1, alpha=0.2)
        plt.xlabel('x (pix)')
        plt.ylabel('y (pix)')
        plt.title(kwargs.get('title'))
        # plt.savefig(kwargs.get('path')+'fit_ellipse/'+kwargs.get('figname')+'_fit.png')
        plt.show()
        print('eps =', isolist.eps[-1])
        print('pa =', isolist.pa[-1])
        # print('sma_max = ', isolist.sma[:])

        return isolist.eps[-1], isolist.pa[-1]  # получается разворот по внешнему эллипсу

    isolist = ellipse.fit_image(step=step)

    for iso in isolist:
        x, y, = iso.sampled_coordinates()
        plt.plot(x, y, color='red', lw=1, alpha=0.4)
    plt.show()
    print('eps =', isolist.eps[-1])
    print('pa =', isolist.pa[-1])
    # print('sma_max = ', isolist.sma[:])

    return isolist.eps[-1], isolist.pa[-1]  # получается разворот по внешнему эллипсу


def calc_sb(image, **kwargs):
    """
    image - image 2D array
    cat - string of catalog (e.g. r_cat[1].data.T[0]
    step - width of elliptical annulus
    f_max - maximal semimajor axis / sma_catalog
    """
    x0 = kwargs.get('x')
    y0 = kwargs.get('y')
    eps0 = kwargs.get('eps')
    theta0 = kwargs.get('theta')

    if kwargs.get('step'):
        step = kwargs.get('step')
    else:
        step = 5.5

    a_in = []
    a_out = []
    b_out = []
    a_in.append(step)
    a_out.append(a_in[-1]+step)
    b_out.append(a_out[-1] * np.sqrt(1 - eps0 ** 2))

    if kwargs.get('rmax'):
        maxsma = kwargs.get('rmax')

        while a_out[-1] < maxsma:
            a_in.append(a_in[-1] + step)
            a_out.append(a_out[-1] + step)
            b_out.append(a_out[-1] * np.sqrt(1 - eps0 ** 2))

        a_in, a_out, b_out = np.array([a_in, a_out, b_out])

        annulae = []
        for a_in_i, a_out_i, b_out_i in zip(a_in, a_out, b_out):
            annulae.append(EllipticalAnnulus((x0, y0), a_in_i, a_out_i, b_out_i, theta=theta0))

        table_aper = aperture_photometry(image, annulae)

        num_apers = len(table_aper.colnames) - 3

        intens = []
        for i in range(num_apers):
            intens.append(table_aper['aperture_sum_' + str(i)][0] / annulae[i].area())

        # print('number of apertures = ', len(intens))
        return (a_out + a_in) / 2., np.array(intens)

    f_max = 5.
    while a_out[-1] < sma0*f_max:
        a_in.append(a_in[-1]+step)
        a_out.append(a_out[-1]+step)
        b_out.append(a_out[-1]*np.sqrt(1-eps0**2))

    a_in, a_out, b_out = np.array([a_in, a_out, b_out])

    annulae = []
    for a_in_i, a_out_i, b_out_i in zip(a_in, a_out, b_out):
        annulae.append(EllipticalAnnulus((x0, y0), a_in_i, a_out_i, b_out_i, theta=theta0))

    table_aper = aperture_photometry(image, annulae)

    num_apers = len(table_aper.colnames) - 3

    intens = []
    for i in range(num_apers):
        intens.append(table_aper['aperture_sum_'+str(i)][0]/annulae[i].area())

    # print('number of apertures = ', len(intens))
    return (a_out+a_in)/2., np.array(intens)


def slit(image, step, width, centre, rmax, angle, **kwargs):

    plt.figure()
    plt.imshow(image, origin='lower', cmap='Greys')

    step_par = np.array([step*np.cos(angle), step*np.sin(angle)])
    step_per = np.array([step*np.cos(angle+np.pi/2.), step*np.sin(angle+np.pi/2.)])

    parallel = [centre]  # сюда я хочу записывать центры прямоугольников
    perpendicular = [centre]  # и сюда тоже

    i = 1
    dr = 0
    while dr < rmax:
        parallel.append(centre+i*step_par)
        parallel.append(centre-i*step_par)
        perpendicular.append(centre + i * step_per)
        perpendicular.append(centre - i * step_per)
        dr = np.sqrt(np.dot(i*step_par, i*step_par))
        i += 1

    parallel = np.array(parallel)
    perpendicular = np.array(perpendicular)
    ind = np.lexsort([parallel.T[0], parallel.T[1]])

    r_par = [parallel[i] for i in ind]
    r_per = [perpendicular[i] for i in ind]

    apertures_par = RectangularAperture(r_par, width, step, angle)
    apertures_per = RectangularAperture(r_per, width, step, angle+np.pi/2.)

    apertures_par.plot(color='green')
    apertures_per.plot(color='red')
    plt.title(kwargs.get('title')+'\n'+str(np.round(angle, 3)))
    # plt.savefig(kwargs.get('path')+'slit_image/'+kwargs.get('figname')+'_slitim.png')
    plt.show()

    table_par = aperture_photometry(image, apertures_par)
    table_per = aperture_photometry(image, apertures_per)

    area = step*width

    intense_par = [elem / area for elem in table_par['aperture_sum']]
    intense_per = [elem / area for elem in table_per['aperture_sum']]

    rad = np.array([k*step for k in range(-i+1, i, 1)])
    return [rad, intense_par], [rad, intense_per]


def unsharp_mask(image, **kwargs):  # пока не работает, забей
    if kwargs.get('size'):
        size = kwargs.get('size')
    else:
        size = 10
    image_med = median_filter(image, size=size)
    plt.figure()
    norm = ImageNormalize(stretch=LogStretch())
    plt.imshow(image, cmap='Greys_r', origin='lower') #, norm=norm)
    plt.show()
    plt.figure()
    plt.imshow(image_med, cmap='Greys_r', origin='lower') #, norm=norm)
    plt.show()
    image_res = gaussian_filter(image-image_med, sigma=2)
    plt.figure()
    plt.imshow(image_res, cmap='Greys_r', origin='lower') #, norm=norm)
    plt.show()
    return image_res


def calc_bkg(image, mask, **kwargs):
    """image - image 2D array
    mask - segmentation image
    kwargs:
    size - backfilter_size
    return:
    Background2D object"""

    if kwargs.get('size'):
        size = kwargs.get('size')
    else:
        size = int(np.shape(image)[0]/4)
    bkg = Background2D(image, (size, size), filter_size=(3, 3), mask=mask)
    # print('background', bkg.background_median)
    return bkg


def find_reg(r, sb, **kwargs):
    try:
        max_r = signal.argrelextrema(ynew, np.less)[0]  # magnitude!
        min_r = signal.argrelextrema(ynew, np.greater)[0]
        interval = range(2*min_r[0] - max_r[0], max_r[0], 1)
    except:
        print('no interval was found!')
        n = len(r)
        interval = range(int(n/4), int(n/2), 1)
    return interval


def find_parabola(r, sb, **kwargs):
    if kwargs.get('s'):
        s = kwargs.get('s')
    else:
        s = 0.3

    tck = splrep(r, sb, s=s)
    ynew = splev(r, tck, der=0)

    plt.figure()
    plt.plot(r*0.396, sb, color='darkred', lw=1, label='profile')

    if kwargs.get('grad'):
        fit_interval, approx_min = interval_grad(r, ynew)
        print('approx min from gradient analysis = ', r[approx_min]*0.396)
        plt.scatter(r[approx_min] * 0.396, sb[approx_min], color='darkmagenta', s=12, label='approx min')
    else:
        fit_interval = find_reg(r, ynew, s=kwargs.get('s'), path=kwargs.get('path'), figname=kwargs.get('figname'))

    fit_r = r[fit_interval]
    fit_sb = sb[fit_interval]
    p = np.poly1d(np.polyfit(fit_r*0.396, fit_sb, deg=2))

    plt.scatter(r[fit_interval]*0.396, sb[fit_interval], color='cyan', s=12, label='interval edges')
    plt.plot(r*0.396, ynew, color='darkmagenta', alpha=0.4, lw=3)
    # plt.title(kwargs.get('title'))
    plt.xlabel('r (arcsec)')
    plt.ylabel('$\mu \quad (mag\:arcsec^{-2})$')
    plt.legend()
    plt.gca().invert_yaxis()
    # plt.savefig(kwargs.get('path') + 'interval/' + kwargs.get('figname') + '_int.png')
    plt.show()

    return fit_r*0.396, p(fit_r*0.396)


def find_outer(image, centre, **kwargs):
    """ image = segmentation map with main object == 1"""

    print('find_outer centre', centre)
    idx_bg = np.where(image != 1)
    idx_main = np.where(image == 1)
    image[idx_main] = 100
    image[idx_bg] = 0

    plt.figure()
    plt.imshow(image, origin='lower')
    plt.show()

    r = [np.sqrt(np.dot(centre-np.array(idx_main).T[i], centre-np.array(idx_main).T[i])) for i in range(len(np.array(idx_main).T))]
    hist = np.histogram(r, bins=100, density=True)
    cum_hist = np.cumsum(hist[0])
    cum_hist = cum_hist/np.amax(cum_hist)
    rc = 0.5*(hist[1][1:] + hist[1][:-1])
    idx_max = np.searchsorted(cum_hist, 0.99)
    idx_min = np.searchsorted(cum_hist, 0.05)
    idx_q3 = np.searchsorted(cum_hist, 0.75)
    idx_q1 = np.searchsorted(cum_hist, 0.25)
    iqr = rc[idx_q3] - rc[idx_q1]
    r_max = rc[idx_max]
    r_min = rc[idx_min]

    FD_bin = 2*iqr/(len(r))**(1./3.)

    print('Freedman Diaconis bin = ', FD_bin)
    print('my width', rc[1]-rc[0])
    r_edges = np.arange(np.amin(r), np.amax(r), FD_bin)

    plt.figure()
    plt.hist(r, bins=r_edges, density=True, alpha=0.5, color='lightseagreen')
    # plt.hist(r, bins=100, density=True, alpha=0.5, color='lightseagreen')
    plt.axvline(r_max, color='red', label='$r_{max}$')
    plt.axvline(r_min, color='darkorange', label='$r_{min}$')
    # plt.fill_betweenx(hist[0], rc[idx_q1], rc[idx_q3], color='c', alpha=0.2)
    if kwargs.get('petro'):
        plt.axvline(kwargs.get('petro')/0.396, color='indigo', label='petro')
    if kwargs.get('petro50'):
        plt.axvline(kwargs.get('petro50')/0.396, color='green', label='petro50')
    plt.title(kwargs.get('title'))
    plt.xlabel('r (pix)')
    plt.legend()
    # plt.savefig(kwargs.get('path')+'rmax_hist/'+kwargs.get('figname')+'_rmax.png')
    plt.show()
    return r_max, r_min, FD_bin


def interval_grad(x, filt):  # надо отфильтрованный перпендикуляр или сглаженную кривую яркости
    grad = np.gradient(filt, x)
    grad2 = np.gradient(grad, x)

    idx_min = signal.argrelextrema(grad2, np.less)[0]
    print(idx_min)
    idx0 = np.sort(np.concatenate([signal.argrelextrema(grad, np.less)[0], signal.argrelextrema(grad, np.greater)[0]]))
    print(idx0)

    interval = np.arange(idx0[0], idx0[1], 1)
    plt.figure()
    plt.plot(x*0.396, grad)
    plt.plot(x*0.396, grad2)
    plt.show()

    return interval, idx_min[0]









