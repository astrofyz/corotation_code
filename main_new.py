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


start = time()
dn = 10
names = names[148:152]
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

        for image in images[:1]:
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
                print(roots)
                # odd_flag = int(splev(0.5 * (roots[0] + roots[1]), tck) > 0)  # 0 if negative in first interval
                intervals = []
                arg_max_curv_rad = signal.argrelextrema(curvature, np.greater)
                max_curv_rad = np.append(rad[arg_max_curv_rad], rad[-1])
                print(len(roots)-1)
                for i in range(len(roots) - 1):   # [odd_flag:][::2]:
                    if any(curvature[np.where((rad<roots[i+1])&(rad>roots[i]))] < 0):
                        intervals.append([roots[i], roots[i+1]])
                        ax3.axvline(intervals[-1][0], color='gold', ls='-', alpha=0.3)
                        ax3.axvline(intervals[-1][1], color='gold', ls='-', alpha=0.3)
                print(intervals)
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
                fig.show()
                # fig.savefig(out_path+f"{band}/b2a_err_l015_{str(image['objID'])}_{band}.png")
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
with figure() as fig:
    plt.plot(rad[:30], curvature[:30])
    plt.axhline(0.)

print(curvature[:30])
#%%
# dn = 10
# n = len(names[:])
# # print(n)
# chunk = 0
# for band, color in zip(['g', 'r', 'z', 'i', 'u'][:1], ['blue', 'r', 'g', 'darkorange', 'm'][:1]):
#     fout = open(out_table_name + f'_{band}.csv', 'a')
#     for chunk in range(0, n, dn):
#         dc = min(chunk + dn, n)
#         images = make_images(names=names[chunk:dc], bands=[band, ], types='all', path=im_path,
#                              calibration=True, manga=True, path_table=table_path)
#         idx_elong = np.where(
#             np.array([images[i]['g']['cat'][1].data.T[0]['B_IMAGE']/images[i]['g']['cat'][1].data.T[0]['A_IMAGE']
#              for i in range(len(images))]) < 0.7)
#         for image in images[:]:
#             fig_mask = plt.figure()
#             plt.imshow(image[band]['real.mag'], origin='lower', cmap='Greys', extent=(-256, 256, -256, 256))
#             ellipse = EllipticalAperture([0., 0.], image[band]['cat'][1].data.T[0]['A_IMAGE'],
#                                          image[band]['cat'][1].data.T[0]['B_IMAGE'],
#                                          image[band]['cat'][1].data.T[0]['THETA_IMAGE']*np.pi/180.)
#             ellipse.plot(axes=plt.gca(), lw=0.3, color='r')
#             # plt.savefig(im_path+f"SE_ellipses/{str(image['objID'])}_{band}.png")
#             plt.close(fig_mask)


#%%
# plt.figure()
# plt.scatter(sb_ell[0], to_mag(sb_ell[1], 22.5, 1.), s=5.)
# plt.scatter(sb_circ[0], to_mag(sb_circ[1], 22.5, 1.), s=5., color='crimson')
# plt.gca().invert_yaxis()
# plt.show()
# plt.close()
