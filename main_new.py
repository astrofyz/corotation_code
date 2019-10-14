from mod_read import *
from mod_analysis import *
from contextlib import contextmanager

table_path = '/media/mouse13/My Passport/corotation/buta_gal/all_table_buta_rad_astrofyz.csv'
im_path = '/media/mouse13/My Passport/corotation/buta_gal/image'
out_path = '/media/mouse13/My Passport/corotation_code/data/newnew/'

names = np.loadtxt('gal_names.txt', dtype='str')
# print(names)

images = make_images(names=names[:2], bands='all', types='all', path=im_path)

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


find_parabola(images[0]['r'], plot=True)
calc_sb(images[0]['r'], error=True)
calc_slit(images[0]['r'], n_slit=40)
# print(images[0]['r']['slit.max'], images[0]['r']['angle.max'])

# print(np.split(images[0]['r']['slit.max'][:-1], 2))

# print(len(images[0]['r']['sb.err']))
# sb_err = interp1d(images[0]['r']['sb'], images[0])
r_min_slit = find_parabola(image=False, rad_pix=np.split(np.array(images[0]['r']['slits.rad.pix'][:-1]), 2)[1],
              sb_err=np.split(np.array(images[0]['r']['slit.max.err'][:-1]), 2)[1],
              r_max=images[0]['r']['r.max.pix'], sb=np.split(np.array(images[0]['r']['slit.max'][:-1]), 2)[1], plot=True)[-1]

print(images[0]['r']['sb.rad.min'])
print(r_min_slit)

r_min_slit = find_parabola(image=False, rad_pix=np.split(np.array(images[0]['r']['slits.rad.pix'][:-1]), 2)[1],
              sb_err=np.split(np.array(images[0]['r']['slit.min.err'][:-1]), 2)[1],
              r_max=images[0]['r']['r.max.pix'], sb=np.split(np.array(images[0]['r']['slit.min'][:-1]), 2)[1], plot=True)[-1]

print(r_min_slit)


# for image in images:
#     try:
#         print('lul')
#         for band in ['g', 'i', 'r', 'u', 'z']:
#             find_parabola(image[band])
#         # with figure(xlabel='r (arcsec)', ylabel='$\mu[g, i, r, u, z] \quad (mag\:arcsec^{-2})$', savename=out_path+str(image['objid14'])+'.png') as fig:
#         with figure(xlabel='r (arcsec)', ylabel='$\mu[g, i, r, u, z] \quad (mag\:arcsec^{-2})$', show=True) as fig:
#             print(out_path+image['name']+'.png')
#             plt.title('{}\n ra={}; dec={}'.format(image['name'], np.round(image['ra'],3), np.round(image['dec'], 3)))
#             plt.gca().invert_yaxis()
#             for band, color in zip(['g', 'i', 'r', 'u', 'z'], ['blue', 'gold', 'r', 'm', 'g']):
#                 plt.plot(image[band]['sb.rad.pix']*0.396, image[band]['sb'], color=color,  label='{} : {}'''.format(band, np.round(image[band]['sb.rad.min'], 3)))
#                 plt.fill_between(image[band]['sb.rad.pix']*0.396, image[band]['sb']-image[band]['sb.err'], image[band]['sb']+image[band]['sb.err'], color=color,  alpha=0.2)
#                 plt.plot(image[band]['sb.rad.fit']*0.396, image[band]['sb.fit'], color='k')
#                 plt.axvline(image[band]['sb.rad.min']*0.396, color=color)
#             plt.legend()
#         image['r'].plot_slits(n_slit=40, savename=out_path+str(image['objid14'])+'_slits.png')
#         idx = np.argmax([sum(abs(row)) for row in image['r']['residuals']])  # перенести это в функцию
#         print(image['name'])
#         print(image['r']['pa'])
#         print(image['r']['slits.angle'][idx])
#
#         with figure(show=True) as fig:
#         # with figure(savename=out_path+'bar_'+str(image['objid14'])+'.png') as fig:
#             plt.title('{}\n ra={}; dec={}'.format(image['name'], np.round(image['ra'], 3), np.round(image['dec'], 3)))
#             plt.imshow(image['r']['real.mag'], origin='lower', cmap='Greys',
#                        norm=ImageNormalize(stretch=LinearStretch()))
#             idx_bar = np.argmax(abs(image['r']['residuals'][idx]))
#             print(image['r']['slits.rad.pix'][idx_bar])
#             xc, yc = np.array([int(dim / 2) for dim in np.shape(image['r']['real.mag'])])
#             aper = CircularAperture([xc, yc], abs(image['r']['slits.rad.pix'][idx_bar]))
#             aper.plot(lw=0.2, color='blue')
#     except:
#         print(image['objid14'], 'none')
#         pass




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
