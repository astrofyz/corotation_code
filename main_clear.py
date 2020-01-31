#%%
import mod_read
import mod_analysis
from mod_read import *
from mod_analysis import *
from contextlib import contextmanager
import os
import scipy.optimize as opt
from scipy.interpolate import UnivariateSpline, splrep, splev, sproot
from time import time
from matplotlib.gridspec import GridSpec
from scipy.signal import argrelextrema

# table_path = '/media/mouse13/Seagate Expansion Drive/corotation/buta_gal/all_table_buta_rad_astrofyz.csv'
# im_path = '/media/mouse13/Seagate Expansion Drive/corotation/buta_gal/image'
# out_path = '/media/mouse13/Seagate Expansion Drive/corotation_code/data/newnew/'
# names = np.loadtxt('gal_names.txt', dtype='str')


table_path = '/media/mouse13/Seagate Expansion Drive/corotation/manga/dr14_zpt02_zpt06_lgmgt9_MANGA_barflag1.csv'
dirbase = '/media/mouse13/Seagate Expansion Drive/corotation/manga/'
im_path = '/media/mouse13/Seagate Expansion Drive/corotation/manga/'
out_path = '/media/mouse13/Seagate Expansion Drive/corotation/manga/pics/corot/'
out_table_name = out_path+'ring_flag.csv'
names = [elem.split('.')[0] for elem in os.listdir(im_path+'input/')]

@contextmanager
def figure(**kw):
    fig = plt.figure()
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
dn = 2
n = len(names[:15])
columns = np.array([[f'R_{band}', f'IR_{band}', f'OR_{band}'] for band in ['g', 'r', 'z', 'i', 'u']]).flatten()
out_table = pd.DataFrame(columns=columns)
out_table.insert(0, 'objID', [])
# print(n)

# print(out_table)

chunk = 0
for chunk in range(0, n, dn):
    dc = min(chunk+dn, n)
    # for image in images[:]:
    # try:
    for band, color in zip(['g', 'r', 'z', 'i', 'u'][:3], ['blue', 'r', 'g', 'darkorange', 'm'][:3]):
        images = make_images(names=names[chunk:dc], bands=[band, ], types='all', path=im_path,
                             calibration=True, manga=True, path_table=table_path)
        print(len(images))
        print(images)
        # all_table.loc[all_table.objID == int(name), [prop_name]].values[0][0]
        for image in images:
            R_flag = False
            IR_flag = False
            OR_flag = False
            try:
                xc, yc = np.array([int(dim / 2) for dim in np.shape(image[band]['real.mag'])])
                fig = plt.figure(figsize=(16, 8))
                gs = GridSpec(nrows=2, ncols=4, figure=fig)
                ax1 = fig.add_subplot(gs[:, :2])
                ax2 = fig.add_subplot(gs[0, 2:])
                ax3 = fig.add_subplot(gs[1, 2:])
                calc_sb(image[band], error=True)
                rad = image[band]['sb.rad.pix']
                radius = np.linspace(min(rad), max(rad), 500)
                sb = image[band]['sb']
                sb_err = image[band]['sb.err']
                conv_kernel = Gaussian1DKernel(stddev=4. * np.sqrt(np.mean(sb_err)))
                sb_conv = convolve(sb, conv_kernel, boundary='extend')
                curvature = find_curvature(rad, sb)
                curvature_conv = find_curvature(rad, sb_conv)
                ax1.imshow(image[band]['real.mag'], origin='lower', cmap='Greys',
                           norm=ImageNormalize(stretch=LinearStretch(slope=1.7)))
                ax2.plot(rad, sb, color=color, lw=1.)
                ax2.plot(rad, sb_conv, color=color, lw=.7, alpha=0.5)
                ax2.fill_between(image[band]['sb.rad.pix'], image[band]['sb'] - image[band]['sb.err']/2.,
                                 image[band]['sb'] + image[band]['sb.err']/2., color=color, alpha=0.1)
                ax3.plot(rad, curvature_conv, color=color, lw=1., alpha=0.6)
                ax3.axhline(0., color='k', lw=0.5)
                tck = splrep(rad, curvature_conv)
                ynew = splev(radius, tck)
                ax3.plot(radius, ynew, color=color, lw=1.)
                roots = sproot(tck, mest=10)
                if len(roots) > 1:
                    odd_flag = int(splev(0.5 * (roots[0] + roots[1]), tck) > 0)  # 0 if negative in first interval
                    intervals = []
                    for i in range(len(roots) - 1)[odd_flag:][::2]:
                        ax2.axvline(roots[i], color=color, lw=0.3, alpha=0.1)
                        ax2.axvline(roots[i+1], color=color, lw=0.3, alpha=0.1)
                        intervals.append([roots[i], roots[i+1]])
                    for interval in intervals:
                        dark_gap_rad = []
                        idxs_rad = np.where((rad < interval[1]) & (rad > interval[0]))
                        p = np.poly1d(np.polyfit(rad[idxs_rad], sb[idxs_rad], deg=2))
                        ax2.plot(rad[idxs_rad], p(rad[idxs_rad]), color='k')
                        if (max(sb[idxs_rad])-min(sb[idxs_rad])) > np.mean(image[band]['sb.err'][idxs_rad]):
                            try:
                                rad_gap = opt.minimize_scalar(-p, method='Bounded',
                                                              bounds=[rad[idxs_rad][0], rad[idxs_rad][-1]]).x
                                ax2.axvline(rad_gap, color=color, alpha=0.7) #, label=np.round(p[2], 3))
                                aper = CircularAperture([xc, yc], rad_gap)
                                aper.plot(axes=ax1, lw=0.5, color=color)
                                R_flag = True
                                dark_gap_rad.append(rad_gap)
                            except:
                                print(f'no minimum for this interval {interval}')
                        else:
                            print('surface brightness change is comparable to error')
                try:
                    I = fourier_harmonics(images[0]['g'], harmonics=[2, ], sig=2, plot=False)
                    rad_f = np.linspace(rad[0], rad[-1], len(I[0]))
                    ax3.plot(rad_f, I[0], label='2', color='k')
                    min_f2 = argrelextrema(I[0], np.less)[0]
                    for rf2 in min_f2[:min([2, len(min_f2)])]:
                        CircularAperture([xc, yc], rad_f[rf2]).plot(axes=ax1, lw=0.5, color='k', alpha=0.5)
                except:
                    print('no fourier')
                ax2.invert_yaxis()
                fig.legend()
                plt.suptitle('{}\nband: {}'.format(image['objID'], band))
                plt.tight_layout()
                fig.show()
                fig.savefig(out_path+f"{band}/all_f_{str(image['objID'])}_{band}.png")
                plt.close()
            except:
                print(image['objID'], band, 'none')
                pass
print(time()-start)
