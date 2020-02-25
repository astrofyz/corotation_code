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
out_table_name = out_path+'ring_flag'
names = [elem.split('.')[0] for elem in os.listdir(im_path+'input/')]
# print(type(names[0]))
# names = [elem.strip() for elem in open(im_path+'bad_pics').readlines()]

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
dn = 8
n = len(names[:])
# print(n)
chunk = 0
for band, color in zip(['g', 'r', 'z', 'i', 'u'][:1], ['blue', 'r', 'g', 'darkorange', 'm'][:1]):
    fout = open(out_table_name + f'_{band}.csv', 'a')
    for chunk in range(0, n, dn):
        dc = min(chunk + dn, n)
        images = make_images(names=names[chunk:dc], bands=[band, ], types='all', path=im_path,
                             calibration=True, manga=True, path_table=table_path)
        for image in images[:]:
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
                curvature = find_curvature(rad, sb)
                # curvature = find_curvature(rad, savgol_filter(sb, 11, 3))
                # curvature_conv = find_curvature(rad, sb_conv)
                # curvature_conv1 = convolve(curvature, conv_kernel, boundary='extend')
                ax1.imshow(image[band]['real.mag'], origin='lower', cmap='Greys',
                           norm=ImageNormalize(stretch=LinearStretch(slope=1.7)))
                ax2.scatter(rad, sb, color=color, s=1.)
                ax2.plot(rad, sb_conv, color=color, lw=.7, alpha=0.5)
                # ax2.plot(rad, savgol_filter(sb, 11, 3), color='crimson', lw=.7, alpha=0.9)
                ax2.fill_between(image[band]['sb.rad.pix'], image[band]['sb.mag'] - image[band]['sb.err.mag'],
                                 image[band]['sb.mag'] + image[band]['sb.err.mag'], color=color, alpha=0.1)
                # ax3.plot(rad, curvature_conv, color='k', lw=1., alpha=0.6)
                # ax3.plot(rad, curvature_conv1, color='r', lw=1., alpha=0.6)
                ax3.plot(rad, curvature, color=color, lw=1., alpha=0.6)
                ax3.axhline(0., color='k', lw=0.5)
                # tck = splrep(rad, curvature_conv)
                # tck = splrep(rad, curvature+0.001*np.ones(len(curvature)))
                tck = splrep(rad, curvature)
                ynew = splev(radius, tck)
                ax3.plot(radius, ynew+0.001*np.ones(len(ynew)), color='gold', lw=1.)
                roots = sproot(tck, mest=10)
                roots = np.hstack([0., roots])
                odd_flag = int(splev(0.5 * (roots[0] + roots[1]), tck) > 0)  # 0 if negative in first interval
                intervals = []
                # arg_max_curv_rad = signal.argrelextrema(curvature_conv, np.greater)
                arg_max_curv_rad = signal.argrelextrema(curvature, np.greater)
                max_curv_rad = np.append(rad[arg_max_curv_rad], rad[-1])
                # print(max_curv_rad)
                for i in range(len(roots) - 1)[odd_flag:][::2]:
                    # ax2.axvline(roots[i], color=color, lw=0.3, alpha=0.7)
                    # ax2.axvline(roots[i+1], color=color, lw=0.3, alpha=0.7)
                    # try:
                    #     intervals.append([roots[i],
                    #                       rad[np.where((rad > roots[i+1])&(rad < roots[i+2]))][np.argmax(curvature_conv[np.where((rad > roots[i+1])&(rad < roots[i+2]))])]])
                    # except:
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
                        ax2.plot(rad, p(rad), color='k', lw=0.3)
                        try:
                            rad_gap = -p[1]/(2*p[2])
                            # if (rad_gap < interval[1]*1.1)&(rad_gap > 0.):
                            # if np.searchsorted(max_curv_rad, rad_gap) == np.searchsorted(max_curv_rad, interval[1]):
                            if rad_gap < rad[-1]:
                                ax2.axvline(rad_gap, color=color, alpha=0.7, label=np.round(rad_gap/interval[1], 3))
                            aper = CircularAperture([xc, yc], rad_gap)
                            aper.plot(axes=ax1, lw=0.5, color=color)
                            dg_radii.append(rad_gap)
                            # print('RAD_GAP/INT ', rad_gap/interval[1])
                            # else:
                            #     print('minimum on the edge')
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
                plt.suptitle('{}\nband: {}'.format(image['name'], band))
                plt.tight_layout()
                # fig.show()
                fig.savefig(out_path+f"{band}/new_all_f_{str(image['objID'])}_{band}.png")
                plt.close()
                fout.write(';'.join([image['objID'], str(int(len(dg_radii)>0)), str(dg_radii), '\n']))
            # print(image['name'])
            except:
                print(image['objID'], band, 'none')
                pass
    fout.close()
print(time()-start)

#%%



print(len(sb))
plt.figure(6)
# plt.scatter(rad, sb, s=3.)
# plt.plot(rad, savgol_filter(sb, 11, 3), color='crimson', lw=.7, alpha=0.9)
plt.plot(rad, savgol_filter(sb, 31, 3), color='navy', lw=.7, alpha=0.9)
plt.plot(rad, savgol_filter(sb, 21, 3), color='green', lw=.7, alpha=0.9)
# plt.plot(images[0]['g']['sb.rad.pix'][:], to_mag(images[0]['g']['sb'][:]), lw=2)
plt.plot(images[0]['g']['sb.rad.pix'][:], sb+2.5*np.log10(1.+images[0]['g']['sb.err.mag'][:]/images[0]['g']['sb']), lw=1, color='k')
plt.plot(images[0]['g']['sb.rad.pix'][:], sb-2.5*np.log10(1.+images[0]['g']['sb.err.mag'][:]/images[0]['g']['sb']), lw=1, color='k')
plt.show()
plt.close()

# plt.figure(7)
# plt.plot(images[0]['g']['sb.rad.pix'], images[0]['g']['sb.err']/images[0]['g']['sb'], lw=2)
# plt.show()
# plt.close()
#%%
from matplotlib.colors import Normalize
plt.figure()
plt.imshow(images[0]['g']['real.bg'], origin='lower', cmap='Greys_r')
plt.colorbar()
plt.show()

#%%
plt.figure()
plt.imshow(images[0]['g']['real.mag'], origin='lower', cmap='Greys_r')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(images[0]['g']['total_error'], origin='lower', cmap='Greys_r')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(images[0]['g']['total_error']/(images[0]['g']['real']*images[0]['g']['real.mag']), origin='lower', cmap='Greys_r',
           norm=Normalize(0., 3.4))
plt.colorbar()
plt.show()

bg_mag = calc_bkg(images[0]['g']['real.mag'], images[0]['g']['seg'])
error_mag = calc_total_error(images[0]['g']['real.mag'], bg_mag.background_rms, images[0]['g']['gain'])

plt.figure()
plt.imshow(error_mag, origin='lower', cmap='Greys_r')
plt.colorbar()
plt.show()
#%%
print(images[0]['g']['sb.err'])
print(images[0]['g']['sb'])

#%%
bg = calc_bkg(images[0]['r']['real'], images[0]['r']['seg'])
bg_mag = calc_bkg(images[0]['r']['real.mag'], images[0]['r']['seg'])
#%%
plt.figure()
plt.imshow(bg.background)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(bg_mag.background)
plt.colorbar()
plt.show()

#%%
errr = calc_total_error(images[0]['r']['real'], bg.background_rms, images[0]['r']['gain'])
errr_mag = calc_total_error(images[0]['r']['real.mag'], bg_mag.background_rms, images[0]['r']['gain'])

plt.figure()
plt.imshow(errr)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(errr_mag)
plt.colorbar()
plt.show()
#%%
# print(chunk, dc)
# importlib.reload(mod_analysis)
# I = fourier_harmonics(images[0]['g'], harmonics=[2, 4], sig=2)
# print(type(I), np.shape(I))
#%%
# tck = splrep(rad, curvature_conv)
# ynew = splev(radius, tck)
# with figure(1) as fig:
#     plt.scatter(rad, curvature_conv, marker='.')
#     plt.plot(radius, ynew)

# with figure(2) as fig:
#     plt.scatter(rad, sb-np.poly1d(fit1d)(rad), marker='.')

# with figure(3) as fig:
#     plt.scatter(rad, sb, marker='.')
#     plt.plot(rad, func(rad, *popt))

# with figure(4) as fig:
#     plt.scatter(rad, sb-func(rad, *popt), marker='.')

# with figure(5) as fig:
#     plt.scatter(rad, curvature, marker='.')
#     plt.plot(rad, np.zeros_like(rad), 'k')

# with figure(6) as fig:
#     plt.scatter(rad, sb_conv, marker='.')
#     plt.scatter(rad, sb, marker='.')

# with figure(7) as fig:
#     plt.scatter(rad, curvature_conv, marker='.')
#     plt.plot(rad, np.zeros_like(rad), 'k')




#%%
# importlib.reload(mod_read)
# print(images[0]['r']['eps'])
#%%
# img = images[0]['r']['real.mag']
# calc_slit(img, n_slit=40, convolve=True)
# from skimage.morphology import disk
# from skimage.filters import rank
#
# image = (img-np.min(img))/(np.max(img) - np.min(img))
#
# selem = disk(25)
# percentile_result = rank.mean_percentile(image, selem=selem, p0=.1, p1=.9)
# bilateral_result = rank.mean_bilateral(image, selem=selem, s0=500, s1=500)
# normal_result = rank.mean(image, selem=selem)
#
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
#                          sharex=True, sharey=True)
# ax = axes.ravel()
#
# titles = ['Original', 'Percentile mean', 'Bilateral mean', 'Local mean']
# imgs = [image, percentile_result, bilateral_result, normal_result]
# for n in range(0, len(imgs)):
#     ax[n].imshow(imgs[n], cmap=plt.cm.gray)
#     ax[n].set_title(titles[n])
#     ax[n].axis('off')
#
# plt.tight_layout()
# plt.show()

# use along with phoutils for rescaling?
#%%
# img = images[0]['r']
# img.plot_slits(n_slit=40)
# #%%
# img.keys()
# plt.figure()
# lefts0 = []
# rights0 = []
# lefts1 = []
# rights1 = []
# for i, slit in enumerate(img['slits']):
#     plt.plot(img['slits.rad.pix'], slit[0])
#     curv0 = find_curvature(img['slits.rad.pix'], slit[0])
#     left = signal.argrelextrema(abs(curv0), np.less)[0][0]
#     right = signal.argrelextrema(abs(curv0), np.less)[0][-1]
#     lefts0.append([left, img['slits.angle'][i]])
#     rights0.append([right, img['slits.angle'][i]+np.pi])
#     plt.axvline(img['slits.rad.pix'][left], color='y')
#     plt.axvline(img['slits.rad.pix'][right])
#     plt.plot(img['slits.rad.pix'], slit[1])
#     curv1 = find_curvature(img['slits.rad.pix'], slit[1])
#     left = signal.argrelextrema(abs(curv1), np.less)[0][0]
#     right = signal.argrelextrema(abs(curv1), np.less)[0][-1]
#     lefts1.append([left, img['slits.angle'][i]+np.pi/2.])
#     rights1.append([right, img['slits.angle'][i]+3.*np.pi/2.])
#     plt.axvline(img['slits.rad.pix'][left], color='red')
#     plt.axvline(img['slits.rad.pix'][right], color='g')
# plt.show()
# #%%
# sizem=2.
# fig = plt.figure()
# for left in lefts0:
#     plt.scatter(img['slits.rad.pix'][left[0]]*np.cos(left[1])+256, img['slits.rad.pix'][left[0]]*np.sin(left[1])+256, color='y', marker='.', s=sizem)
# for right in rights0:
#     plt.scatter(-img['slits.rad.pix'][right[0]]*np.sin(right[1])+256, -img['slits.rad.pix'][right[0]]*np.cos(right[1])+256, color='b', marker='.', s=sizem)
# for left in lefts1:
#     plt.scatter(img['slits.rad.pix'][left[0]]*np.cos(left[1])+256, img['slits.rad.pix'][left[0]]*np.sin(left[1])+256, color='r', marker='.', s=sizem)
# for right in rights1:
#     plt.scatter(img['slits.rad.pix'][right[0]]*np.sin(right[1])+256, img['slits.rad.pix'][right[0]]*np.cos(right[1])+256, color='g', marker='.', s=sizem)
# plt.xlim(0, 512)
# plt.ylim(0, 512)
# plt.gca().set_aspect('equal', adjustable='box')
# # ax[0].set_aspect('equal')
# plt.show()
# #%%
# from astropy.visualization import PowerStretch
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax2 = fig.add_subplot(1, 1, 1, projection='3d')
# theta, r = np.meshgrid(img['slits.angle'], img['slits.rad.pix'])
# ax2.plot_surface(r, theta, img['slits'][:, 0, :].T,
#                  linewidth=0, alpha=0.5, cmap='Blues',
#                  norm=ImageNormalize(stretch=PowerStretch(a=10)))
# ax2.plot_surface(r, theta, img['slits'][:, 1, :].T,
#                  linewidth=0, alpha=0.5, cmap='Reds',
#                  norm=ImageNormalize(stretch=PowerStretch(a=10)))
# ax2.set_zlim(bottom=ax2.get_zlim()[1], top=ax2.get_zlim()[0])
# ax2.view_init(elev=20, azim=90)
#
# plt.show()

#%%
# from skimage.morphology import disk
# from skimage.filters import rank
# from skimage import img_as_float
#
# image = img_as_float(img['real.mag'])
# selem = disk(img['petroRad'])
#
# percentile_result = rank.mean_percentile(image, selem=selem, p0=.1, p1=.9)
# bilateral_result = rank.mean_bilateral(image, selem=selem, s0=500, s1=500)
# normal_result = rank.mean(image, selem=selem)
#
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
#                          sharex=True, sharey=True)
# ax = axes.ravel()
#
# titles = ['Original', 'Percentile mean', 'Bilateral mean', 'Local mean']
# imgs = [image, percentile_result, bilateral_result, normal_result]
# for n in range(0, len(imgs)):
#     ax[n].imshow(imgs[n], cmap=plt.cm.gray)
#     ax[n].set_title(titles[n])
#     ax[n].axis('off')
#
# plt.tight_layout()
# plt.show()
#%%
#
# ax2.scatter(
#     point_x,
#     point_y,
#     point_z,
#     color='blue',
#     label='$|s - 1|<0.05$',
#     **scatter_settings
# )
#
# ax2.scatter(
#     point_x_s,
#     point_y_s,
#     point_z_s,
#     color='green',
#     label='$|s\' - 1|<0.05$',
#     **scatter_settings
# )

#%%


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

# calc_slit(images['r'], angle=images['r']['pa'], n_slit=20, convolve=True)
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


# from skimage.filters import unsharp_mask
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.measure import label
# from skimage import data
# from skimage import color
# from skimage.morphology import extrema
# from skimage import exposure
#
# res = unsharp_mask(images[0]['r']['real.mag'], radius=0.01, amount=3)
# img = images[0]['r']['real.mag']
# h = 5.
# h_maxima = extrema.h_maxima(img, h)
# label_h_maxima = label(h_maxima)
# overlay_h = color.label2rgb(label_h_maxima, img, alpha=0.7, bg_label=0,
#                             bg_color=None, colors=[(1, 0, 0)])
#
# fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
# axes[0].imshow(images[0]['r']['real.mag'], origin='lower', cmap='Greys',
#                        norm=ImageNormalize(stretch=LinearStretch(slope=1.7)))
# # axes[1].imshow(res, origin='lower', cmap='Greys')
# # print(overlay_h)
# axes[1].imshow(overlay_h[:,:,0], origin='lower', cmap='Greys', norm=ImageNormalize(vmin=10, vmax=40))
# # axes[1].imshow(overlay_h[:,:,1], origin='lower')
# # axes[1].imshow(overlay_h[:,:,2], origin='lower')
# plt.show()
# plt.close()
#
# print(overlay_h[:, :, 0].shape)
# # print(np.max(overlay_h), np.min(overlay_h))
#
# # print(img.shape)
# # print(np.max(img), np.min(img))
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy import ndimage as ndi
# from skimage.feature import shape_index
# from skimage.draw import circle
#
# image = images[0]['r']['real.mag']
# s = shape_index(image)
#
# # In this example we want to detect 'spherical caps',
# # so we threshold the shape index map to
# # find points which are 'spherical caps' (~1)
#
# target = 20
# delta = 2
#
# point_y, point_x = np.where(np.abs(s - target) < delta)
# point_z = image[point_y, point_x]
#
# # The shape index map relentlessly produces the shape, even that of noise.
# # In order to reduce the impact of noise, we apply a Gaussian filter to it,
# # and show the results once in
#
# s_smooth = ndi.gaussian_filter(s, sigma=10)
#
# point_y_s, point_x_s = np.where(np.abs(s_smooth - target) < delta)
# point_z_s = image[point_y_s, point_x_s]
#
# fig = plt.figure(figsize=(12, 4))
# ax1 = fig.add_subplot(1, 3, 1)
#
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.axis('off')
# ax1.set_title('Input image')
#
# scatter_settings = dict(alpha=0.75, s=10, linewidths=0)
#
# ax1.scatter(point_x, point_y, color='blue', **scatter_settings)
# ax1.scatter(point_x_s, point_y_s, color='green', **scatter_settings)
#
# ax2 = fig.add_subplot(1, 3, 2, projection='3d', sharex=ax1, sharey=ax1)
#
# x, y = np.meshgrid(
#     np.arange(0, image.shape[0], 1),
#     np.arange(0, image.shape[1], 1)
# )
#
# ax2.plot_surface(x, y, image, linewidth=0, alpha=0.5)
#
# ax2.scatter(
#     point_x,
#     point_y,
#     point_z,
#     color='blue',
#     label='$|s - 1|<0.05$',
#     **scatter_settings
# )
#
# ax2.scatter(
#     point_x_s,
#     point_y_s,
#     point_z_s,
#     color='green',
#     label='$|s\' - 1|<0.05$',
#     **scatter_settings
# )
#
# ax2.legend(loc='lower left')
#
# ax2.axis('off')
# ax2.set_title('3D visualization')
#
# ax3 = fig.add_subplot(1, 3, 3, sharex=ax1, sharey=ax1)
#
# ax3.imshow(s, cmap=plt.cm.gray)
# ax3.axis('off')
# ax3.set_title('Shape index, $\sigma=1$')
#
# fig.tight_layout()
#
# plt.show()

#%%
# for image in images[:]:
#     try:
#         for band in ['r', 'g', 'z', 'u', 'i']: #почему ошибки с plot false, что там происходит?
#             find_parabola(image[band], plot=False, band=band, savename=out_path+f"sb_check/sb_{str(image['objID'])}_{band}.png")
#
#         # calc_slit(image['z'], 40, convolve=True)  # возможно, вообще не надо
#         for band in ['z', 'g', 'r', 'u', 'i']:
#             if 'eps' not in image[band].keys():
#                 image[band]['eps'] = 0.
#
#         # plot surface brigtness profiles with fitted parabola
#         # with figure(xlabel='r (arcsec)', ylabel='$\mu[g, i, r, u, z] \quad (mag\:arcsec^{-2})$', savename=out_path+str(image['objID'])+'.png') as fig:
#         #     plt.title('{}\n ra={}; dec={}'.format(image['name'], np.round(image['ra'],3), np.round(image['dec'], 3)))
#         #     plt.gca().invert_yaxis()
#         #     for band, color in zip(['g', 'r', 'z', 'i', 'u'][:3], ['blue', 'r', 'g', 'gold', 'm'][:3]):
#         #         plt.plot(image[band]['sb.rad.pix']*0.396, image[band]['sb'], color=color,  label='{} : {}'''.format(band, np.round(image[band]['sb.rad.min'], 3)))
#         #         plt.fill_between(image[band]['sb.rad.pix']*0.396, image[band]['sb']-image[band]['sb.err'], image[band]['sb']+image[band]['sb.err'], color=color,  alpha=0.1)
#         #         plt.plot(image[band]['sb.rad.fit']*0.396, image[band]['sb.fit'], color='k')
#         #         plt.axvline(image[band]['sb.rad.min']*0.396, color=color)
#         #         # plt.vlines(kmeans.cluster_centers_.T[1]*0.396, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], label='kmeans')
#         #     plt.legend()
#
#         # find position angle of bar
#         # image['r'].plot_slits(n_slit=40, savename=out_path+str(image['objID14'])+'_slits.png')
#         # r_min_slit_0 = find_fancy_parabola(image=False, rad_pix=np.split(np.array(image['r']['slits.rad.pix'][:-1]), 2)[1],
#         #                                    sb_err=np.split(np.array(image['r']['slit.min.err'][:-1]), 2)[1],
#         #                                    r_max=image['r']['r.max.pix'],
#         #                                    sb=np.split(np.array(image['r']['slit.min'][:-1]), 2)[1])
#         #
#         # print(r_min_slit_0[-1])
#         # r_min_slit_1 = find_fancy_parabola(image=False, rad_pix=np.split(np.array(image['r']['slits.rad.pix'][:-1]), 2)[1],
#         #                              sb_err=np.split(np.array(image['r']['slit.max.err'][:-1]), 2)[1],
#         #                              r_max=image['r']['r.max.pix'],
#         #                              sb=np.split(np.array(image['r']['slit.max'][:-1]), 2)[1], plot=True)[-1]
#         #
#         # print(r_min_slit_1[-1])
#         # plot "bar"
#         # with figure(show=True) as fig:
#
#         # with figure(savename=out_path+'bar_'+str(image['objID'])+'.png') as fig:
#         #     plt.title('{}\n ra={}; dec={}; eps={}'.format(image['name'], np.round(image['ra'], 3), np.round(image['dec'], 3), np.round(image['z']['eps'], 3)))
#         #     plt.imshow(image['z']['real.mag'], origin='lower', cmap='Greys',
#         #                norm=ImageNormalize(stretch=LinearStretch(slope=1.7)))
#             # idx = np.argmax([sum(abs(row)) for row in image['r']['residuals']])  # перенести это в функцию
#             # idx_bar = np.argmax(abs(image['r']['residuals'][idx]))
#             # print(image['r']['slits.rad.pix'][idx_bar])
#             # xc, yc = np.array([int(dim / 2) for dim in np.shape(image['z']['real.mag'])])
#             # aper = CircularAperture([xc, yc], abs(image['r']['slits.rad.pix'][idx_bar]))
#             # aper.plot(lw=0.2, color='blue', label='max_resid')
#             # aper = CircularAperture([xc, yc], abs(image['r']['sb.rad.min']))
#             # aper.plot(lw=0.3, color='red', label='corot_r')
#             # aper = CircularAperture([xc, yc], abs(image['g']['sb.rad.min']))
#             # aper.plot(lw=0.3, color='blue', label='corot_g')
#             # aper = CircularAperture([xc, yc], abs(image['i']['sb.rad.min']))
#             # aper.plot(lw=0.3, color='gold', label='corot_i')
#             # aper = CircularAperture([xc, yc], abs(image['u']['sb.rad.min']))
#             # aper.plot(lw=0.3, color='green', label='corot_u')
#             # aper = CircularAperture([xc, yc], abs(image['z']['sb.rad.min']))
#             # aper.plot(lw=0.3, color='purple', label='corot_z')
#             # aper = CircularAperture([xc, yc], r_min_slit_0[-1])
#             # aper.plot(lw=0.2, color='red', label='perpendicular')
#             # plt.legend()
#     except:
#         print(image['objID'], 'none')
#         pass
