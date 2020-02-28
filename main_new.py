import mod_read
import mod_analysis
from mod_read import *
from mod_analysis import *
from contextlib import contextmanager
import importlib
import os
import umap
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import scipy.optimize as opt
from scipy.interpolate import UnivariateSpline, splrep, splev, sproot
from time import time
from matplotlib.gridspec import GridSpec
from scipy.signal import argrelextrema, savgol_filter

# table_path = '/media/mouse13/Seagate Expansion Drive/corotation/buta_gal/all_table_buta_rad_astrofyz.csv'
# im_path = '/media/mouse13/Seagate Expansion Drive/corotation/buta_gal/image'
# out_path = '/media/mouse13/Seagate Expansion Drive/corotation_code/data/newnew/'
# names = np.loadtxt('gal_names.txt', dtype='str')


table_path = '/media/mouse13/Seagate Expansion Drive/corotation/manga/dr14_zpt02_zpt06_lgmgt9_MANGA_barflag1.csv'
dirbase = '/media/mouse13/Seagate Expansion Drive/corotation/manga/'
im_path = '/media/mouse13/Seagate Expansion Drive/corotation/manga/'
out_path = '/media/mouse13/Seagate Expansion Drive/corotation/manga/pics/corot/'
out_table_name = out_path+'clear_ring_flag_l02'
# names = [elem.split('.')[0] for elem in os.listdir(im_path+'input/')]
# print(type(names[0]))
names = [elem.strip() for elem in open(im_path+'clear_ring.txt').readlines()]

print(names)
# images = make_images(names=names[:5], bands='all', types='all', path=im_path, SE=True, calibration=True, correction=True)
# images = make_images(names=names[10:20], bands=['r', 'g', 'z'], types=['seg', 'real', 'cat'], path=dirbase, path_table=table_path, manga=True)
# try other bands

@contextmanager
def figure(num=1, **kw):
    fig = plt.figure(num=num)
    yield fig
    if 'title' in kw:
        plt.title(kw.get('title'))
    if 'xlabel' in kw:
        plt.xlabel(kw.get('xlabel'))
    if 'ylabel' in kw:
        plt.xlabel(kw.get('ylabel'))
    if 'savename' in kw:
        plt.savefig(kw.get('savename'))
    plt.show()
    plt.close()

#%%
start = time()
dn = 10
n = len(names[:])
# print(n)
chunk = 0
for band, color in zip(['g', 'r', 'z', 'i', 'u'][:1], ['blue', 'r', 'g', 'darkorange', 'm'][:1]):
    fout = open(out_table_name + f'_{band}.csv', 'a')
    for chunk in range(0, n, dn):
        dc = min(chunk + dn, n)
        images = make_images(names=names[chunk:dc], bands=[band, ], types='all', path=im_path,
                             calibration=True, manga=True, path_table=table_path)
        idx_elong = np.where(
            np.array([images[i]['g']['cat'][1].data.T[0]['B_IMAGE']/images[i]['g']['cat'][1].data.T[0]['A_IMAGE']
             for i in range(len(images))]) < 0.7)
        # print(idx_elong)
        # print(np.array([images[i]['g']['cat'][1].data.T[0]['B_IMAGE']/images[i]['g']['cat'][1].data.T[0]['A_IMAGE']
        #      for i in range(len(images))]))
        for image in images[:]:
            # fig_orig = plt.figure()
            # size = int(np.shape(image[band]['real.mag'])[0]/2)
            # plt.imshow(image[band]['real.mag'], origin='lower', cmap='Greys', extent=(-size, size, -size, size))
            # plt.gca().set_aspect('equal')
            # plt.savefig(im_path+f"clear_ring/{str(image['objID'])}_{band}.png")
            # plt.close(fig_orig)

            dg_radii = []
            try:
                xc, yc = np.array([int(dim / 2) for dim in np.shape(image[band]['real.mag'])])
                fig = plt.figure(figsize=(16, 8))
                gs = GridSpec(nrows=2, ncols=4, figure=fig)
                ax1 = fig.add_subplot(gs[:, :2])
                ax2 = fig.add_subplot(gs[0, 2:])
                ax3 = fig.add_subplot(gs[1, 2:])
                calc_sb(image[band], error=True, step=1.)
                rad = image[band]['sb.rad.pix']
                radius = np.linspace(min(rad), max(rad), 500)
                sb = image[band]['sb.mag']
                sb_err = image[band]['sb.err.mag']
                conv_kernel = Gaussian1DKernel(stddev=max([2, 0.05*len(rad)]))
                sb_conv = convolve(sb, conv_kernel, boundary='extend')
                curvature = find_curvature(rad, savgol_filter(sb, int(len(rad)/10)+(1-int(int(len(rad)/10)%2)), 3))
                # curvature = find_curvature(rad, sb)
                ax1.imshow(image[band]['real.mag'], origin='lower', cmap='Greys',
                           norm=ImageNormalize(stretch=LinearStretch(slope=1.7)), extent=(-xc, xc, -yc, yc))
                ax2.scatter(rad, sb, color=color, s=1.)
                ax2.fill_between(image[band]['sb.rad.pix'], image[band]['sb.mag'] - image[band]['sb.err.mag'],
                                 image[band]['sb.mag'] + image[band]['sb.err.mag'], color=color, alpha=0.1)
                ax3.plot(rad, curvature, color='gold', lw=1., alpha=1.)
                ax3.axhline(0., color='k', lw=0.5)
                tck = splrep(rad, curvature)
                ynew = splev(radius, tck)
                ax3.plot(radius, ynew, color=color, lw=1., alpha=0.6, label='SG, wsize = 0.1*len, deg=3')
                roots = sproot(tck, mest=10)
                roots = np.hstack([0., roots])
                odd_flag = int(splev(0.5 * (roots[0] + roots[1]), tck) > 0)  # 0 if negative in first interval
                intervals = []
                arg_max_curv_rad = signal.argrelextrema(curvature, np.greater)
                max_curv_rad = np.append(rad[arg_max_curv_rad], rad[-1])
                for i in range(len(roots) - 1)[odd_flag:][::2]:
                    intervals.append([roots[i], roots[i+1]])
                    ax3.axvline(intervals[-1][0], color='gold', ls='-', alpha=0.3)
                    ax3.axvline(intervals[-1][1], color='gold', ls='-', alpha=0.3)

                for interval in intervals:
                    idxs_rad = np.where((rad <= interval[1]) & (rad >= interval[0]))
                    p = np.poly1d(np.polyfit(rad[idxs_rad], sb[idxs_rad], deg=2))
                    p1 = np.poly1d(np.polyfit([rad[idxs_rad][0], rad[idxs_rad][-1]], [sb[idxs_rad][0], sb[idxs_rad][-1]], deg=1))
                    # if max(abs(p1(rad[idxs_rad])-sb[idxs_rad]))>max(sb_err[idxs_rad]):
                    if (max(sb[idxs_rad])-min(sb[idxs_rad])) > np.max(image[band]['sb.err.mag'][idxs_rad]):
                        # ax2.plot(rad[idxs_rad], p(rad[idxs_rad]), color='k')
                        try:
                            rad_gap = -p[1]/(2*p[2])
                            rad_err = abs(rad_gap - rad[idxs_rad][np.argmax(sb[idxs_rad])])/rad_gap
                            ax2.plot(rad, p(rad), color='k', lw=0.3, label=np.round(rad_err, 3))
                            # print(rad_gap, rad[idxs_rad][np.argmax(sb[idxs_rad])])
                            ax2.axvline(rad_gap, color=color, alpha=0.7, lw=0.5)
                            ax2.axvline(rad[idxs_rad][np.argmax(sb[idxs_rad])], color='r', alpha=0.4, lw=0.5)
                            # if (rad_gap < interval[1]*1.1)&(rad_gap > 0.):
                            # if np.searchsorted(max_curv_rad, rad_gap) == np.searchsorted(max_curv_rad, interval[1]):
                            # if rad_gap < rad[-1]:
                            # print('ratio', rad_gap/rad[idxs_rad][np.argmax(sb[idxs_rad])])
                            if rad_err < 0.15:
                                # ax2.axvline(rad_gap, color=color, alpha=0.7)
                                # ax2.axvline(rad[idxs_rad][np.argmax(sb[idxs_rad])], color='r', alpha=0.4)
                                eps = np.sqrt(1 - (image[band]['cat'][1].data.T[0]['B_IMAGE'] /
                                                   image[band]['cat'][1].data.T[0]['A_IMAGE']) ** 2)
                                b = rad_gap * np.sqrt(1 - eps ** 2)
                                aper = EllipticalAperture([0., 0.], rad_gap, b, image[band]['cat'][1].data.T[0]['THETA_IMAGE']*np.pi/180.)
                                # aper = CircularAperture([0, 0], rad_gap)
                                aper.plot(axes=ax1, lw=0.3, color=color)
                                # aper = CircularAperture([xc, yc], rad[idxs_rad][np.argmax(sb[idxs_rad])])
                                # aper.plot(axes=ax1, lw=0.5, color='r', alpha=0.6)
                                dg_radii.append(rad_gap)
                            # else:
                            #     print('minimum beyond the interval')
                        except:
                            print(f'no minimum for this interval {interval}')
                    else:
                        print('surface brightness change is comparable to error')
                # try:
                #     I = fourier_harmonics(images[0]['g'], harmonics=[2, ], sig=2, plot=False)
                #     rad_f = np.linspace(rad[0], rad[-1], len(I[0]))
                #     ax3.plot(rad_f, I[0], label='2', color='k')
                #     min_f2 = argrelextrema(I[0], np.less)[0][0]
                #     CircularAperture([xc, yc], rad_f[min_f2]).plot(axes=ax1, lw=0.5, color='k', alpha=0.5)
                # except:
                #     print('no fourier')
                ax2.set_ylim(min(sb), max(sb))
                ax2.invert_yaxis()
                fig.legend()
                plt.suptitle('{}\nband: {}'.format(image['objID'], band))
                plt.tight_layout()
                # fig.show()
                fig.savefig(out_path+f"{band}/b2a_l07_{str(image['objID'])}_{band}.png")
                plt.close(fig)
                # fout.write(';'.join([image['objID'], str(image['ra']), str(image['dec']), str(int(len(dg_radii)>0)), str(len(dg_radii)), str(dg_radii), '\n']))
            # print(image['name'])
            except:
                print(image['objID'], band, 'none')
                # fout.write(';'.join([image['objID'], 'none', '\n']))
                pass
    fout.close()
print(time()-start)

#%%
# print(len(images))
# перестроить в эллиптических апертурах
#%%
plt.figure()
plt.imshow(images[0]['g']['seg'], origin='lower')
plt.show()
plt.close()
#%%
eps = np.sqrt(1 - (images[0]['g']['cat'][1].data.T[0]['B_IMAGE'] / images[0]['g']['cat'][1].data.T[0]['A_IMAGE']) ** 2)
geom_inp = EllipseGeometry(x0=xc, y0=yc, sma=20, eps=eps, pa=images[0]['g']['cat'][1].data.T[0]['THETA_IMAGE']*np.pi/180.)
aper_inp = EllipticalAperture((geom_inp.x0, geom_inp.y0), geom_inp.sma, geom_inp.sma*np.sqrt(1 - geom_inp.eps**2),
                                  geom_inp.pa)
#%%
with figure() as fig:
    plt.imshow(images[0]['g']['real.mag'], origin='lower', cmap='Greys_r')
    eps = np.sqrt(
        1 - (images[0]['g']['cat'][1].data.T[0]['B_IMAGE'] / images[0]['g']['cat'][1].data.T[0]['A_IMAGE']) ** 2)
    # print('b', b, image['cat'][1].data.T[0]['THETA_IMAGE'])
    b = rad_gap * np.sqrt(1 - eps ** 2)
    aper = EllipticalAperture([256, 256], rad_gap, b, images[0]['g']['cat'][1].data.T[0]['THETA_IMAGE']*np.pi/180.)
    # aper = CircularAperture([0, 0], rad_gap)
    aper.plot(lw=0.3, color=color)

    print(eps)
    print(images[0]['g']['cat'][1].data.T[0]['THETA_IMAGE']*np.pi/180.)

#%%
print(images[0]['g'].keys())
#%%
sb_ell = calc_sb(images[0]['g'], error=True, step=1.)
sb_circ = calc_sb(images[0]['g'], circ_aper = True, step=1.)
#%%
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

#%%
plt.figure()
plt.scatter(sb_ell[0], to_mag(sb_ell[1], 22.5, 1.), s=5.)
plt.scatter(sb_circ[0], to_mag(sb_circ[1], 22.5, 1.), s=5., color='crimson')
plt.gca().invert_yaxis()
plt.show()
plt.close()
