import mod_read
import mod_analysis
from mod_read import *
from mod_analysis import *
from contextlib import contextmanager
import importlib
import os
import umap
from time import time


# table_path = '/media/mouse13/Seagate Expansion Drive/corotation/buta_gal/all_table_buta_rad_astrofyz.csv'
# im_path = '/media/mouse13/Seagate Expansion Drive/corotation/buta_gal/image'
# out_path = '/media/mouse13/Seagate Expansion Drive/corotation_code/data/newnew/'
# names = np.loadtxt('gal_names.txt', dtype='str')


table_path = '/media/mouse13/Seagate Expansion Drive/corotation/manga/dr14_zpt02_zpt06_lgmgt9_MANGA_barflag.csv'
dirbase = '/media/mouse13/Seagate Expansion Drive/corotation/manga/'
im_path = '/media/mouse13/Seagate Expansion Drive/corotation/manga/input/'
out_path = '/media/mouse13/Seagate Expansion Drive/corotation/manga/pics/corot/'
names = [elem.split('.')[0] for elem in os.listdir(im_path)]

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
dn = 10
n = len(names)
for chunk in range(0, n, dn):
    dc = min(chunk+dn, n)
    # images = make_images(names=names[26:], bands='all', types='all', path=im_path, SE=True, calibration=True, correction=True)
    images = make_images(names=names[chunk:dc], bands=['r', 'g', 'z'], types=['seg', 'real', 'cat'], path=dirbase,
                         path_table=table_path, manga=True)
    for image in images:
        try:
            for band in ['z', 'g', 'r', 'i', 'u'][:3]:
                find_parabola(image[band], plot=True, band=band,
                              savename=out_path + f"sb_check/sb_{str(image['objID'])}_{band}.png")

            calc_slit(image['z'], 40, convolve=True)  # возможно, вообще не надо
            for band in ['z', 'g', 'r', 'i', 'u'][:3]:
                if 'eps' not in image[band].keys():
                    image[band]['eps'] = 0.

            # plot surface brigtness profiles with fitted parabola
            with figure(xlabel='r (arcsec)', ylabel='$\mu[g, i, r, u, z] \quad (mag\:arcsec^{-2})$',
                        savename=out_path + str(image['objID']) + '.png') as fig:
                plt.title('{}\n ra={}; dec={}'.format(image['name'], np.round(image['ra'], 3), np.round(image['dec'], 3)))
                plt.gca().invert_yaxis()
                for band, color in zip(['g', 'r', 'z', 'i', 'u'][:3], ['blue', 'r', 'g', 'gold', 'm'][:3]):
                    plt.plot(image[band]['sb.rad.pix'] * 0.396, image[band]['sb'], color=color,
                             label='{} : {}'''.format(band, np.round(image[band]['sb.rad.min'], 3)))
                    plt.fill_between(image[band]['sb.rad.pix'] * 0.396, image[band]['sb'] - image[band]['sb.err'],
                                     image[band]['sb'] + image[band]['sb.err'], color=color, alpha=0.1)
                    plt.plot(image[band]['sb.rad.fit'] * 0.396, image[band]['sb.fit'], color='k')
                    plt.axvline(image[band]['sb.rad.min'] * 0.396, color=color)
                plt.legend()

            with figure(savename=out_path + 'bar_' + str(image['objID']) + '.png') as fig:
                plt.title('{}\n ra={}; dec={}; eps={}'.format(image['name'], np.round(image['ra'], 3),
                                                              np.round(image['dec'], 3), np.round(image['z']['eps'], 3)))
                plt.imshow(image['z']['real.mag'], origin='lower', cmap='Greys',
                           norm=ImageNormalize(stretch=LinearStretch(slope=1.7)))

                xc, yc = np.array([int(dim / 2) for dim in np.shape(image['z']['real.mag'])])
                aper = CircularAperture([xc, yc], abs(image['r']['sb.rad.min']))
                aper.plot(lw=0.3, color='red', label='corot_r')
                aper = CircularAperture([xc, yc], abs(image['g']['sb.rad.min']))
                aper.plot(lw=0.3, color='blue', label='corot_g')
                # aper = CircularAperture([xc, yc], abs(image['i']['sb.rad.min']))
                # aper.plot(lw=0.3, color='gold', label='corot_i')
                # aper = CircularAperture([xc, yc], abs(image['u']['sb.rad.min']))
                # aper.plot(lw=0.3, color='green', label='corot_u')
                aper = CircularAperture([xc, yc], abs(image['z']['sb.rad.min']))
                aper.plot(lw=0.3, color='purple', label='corot_z')
                plt.legend()
        except:
            print(image['objID'], 'none')
            pass
print(time()-start)