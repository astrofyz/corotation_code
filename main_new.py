import mod_read
import mod_analysis
from mod_read import *
from mod_analysis import *
from contextlib import contextmanager
import importlib

#%%
table_path = '/media/mouse13/My Passport/corotation/buta_gal/all_table_buta_rad_astrofyz.csv'
im_path = '/media/mouse13/My Passport/corotation/buta_gal/image'
out_path = '/media/mouse13/My Passport/corotation_code/data/newnew/'

names = np.loadtxt('gal_names.txt', dtype='str')
# print(names)

images = make_images(names=names[0:2], bands='all', types='all', path=im_path)

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

#%%
images = [images]
# print(len(images))

#%%
importlib.reload(mod_analysis)
for image in images[0]:
    print(image.keys())
    # try:
        for band in ['g', 'i', 'r', 'u', 'z']:
            find_parabola(image[band])

        calc_slit(image['r'], 40, convolve=True)
        # plot surface brigtness profiles with fitted parabola
        # with figure(xlabel='r (arcsec)', ylabel='$\mu[g, i, r, u, z] \quad (mag\:arcsec^{-2})$', savename=out_path+str(image['objid14'])+'.png') as fig:
        #     plt.title('{}\n ra={}; dec={}'.format(image['name'], np.round(image['ra'],3), np.round(image['dec'], 3)))
        #     plt.gca().invert_yaxis()
        #     for band, color in zip(['g', 'i', 'r', 'u', 'z'], ['blue', 'gold', 'r', 'm', 'g']):
        #         plt.plot(image[band]['sb.rad.pix']*0.396, image[band]['sb'], color=color,  label='{} : {}'''.format(band, np.round(image[band]['sb.rad.min'], 3)))
        #         plt.fill_between(image[band]['sb.rad.pix']*0.396, image[band]['sb']-image[band]['sb.err'], image[band]['sb']+image[band]['sb.err'], color=color,  alpha=0.2)
        #         plt.plot(image[band]['sb.rad.fit']*0.396, image[band]['sb.fit'], color='k')
        #         plt.axvline(image[band]['sb.rad.min']*0.396, color=color)
        #     plt.legend()
        # find position angle of bar
        # image['r'].plot_slits(n_slit=40, savename=out_path+str(image['objid14'])+'_slits.png')

        # r_min_slit_0 = find_fancy_parabola(image=False, rad_pix=np.split(np.array(image['r']['slits.rad.pix'][:-1]), 2)[1],
        #                                    sb_err=np.split(np.array(image['r']['slit.min.err'][:-1]), 2)[1],
        #                                    r_max=image['r']['r.max.pix'],
        #                                    sb=np.split(np.array(image['r']['slit.min'][:-1]), 2)[1])
        #
        # print(r_min_slit_0[-1])

        # r_min_slit_1 = find_fancy_parabola(image=False, rad_pix=np.split(np.array(image['r']['slits.rad.pix'][:-1]), 2)[1],
        #                              sb_err=np.split(np.array(image['r']['slit.max.err'][:-1]), 2)[1],
        #                              r_max=image['r']['r.max.pix'],
        #                              sb=np.split(np.array(image['r']['slit.max'][:-1]), 2)[1], plot=True)[-1]
        #
        # print(r_min_slit_1[-1])

        # plot "bar"
        # with figure(show=True) as fig:
        with figure(savename=out_path+'bar_'+str(image['objid14'])+'.png') as fig:
            plt.title('{}\n ra={}; dec={}'.format(image['name'], np.round(image['ra'], 3), np.round(image['dec'], 3)))
            plt.imshow(image['r']['real.mag'], origin='lower', cmap='Greys',
                       norm=ImageNormalize(stretch=LinearStretch(slope=1.7)))
            idx = np.argmax([sum(abs(row)) for row in image['r']['residuals']])  # перенести это в функцию
            idx_bar = np.argmax(abs(image['r']['residuals'][idx]))
            # print(image['r']['slits.rad.pix'][idx_bar])
            xc, yc = np.array([int(dim / 2) for dim in np.shape(image['r']['real.mag'])])
            aper = CircularAperture([xc, yc], abs(image['r']['slits.rad.pix'][idx_bar]))
            aper.plot(lw=0.2, color='blue', label='max_resid')
            aper = CircularAperture([xc, yc], abs(image['r']['sb.rad.min']))
            aper.plot(lw=0.2, color='red', label='corot_r')
            aper = CircularAperture([xc, yc], r_min_slit_0[-1])
            aper.plot(lw=0.2, color='red', label='perpendicular')
            plt.legend()
    # except:
    #     print(image['objid14'], 'none')
    #     pass

#%%
importlib.reload(mod_read)
img = images[0][0]['r']
calc_slit(img, n_slit=40, convolve=True)

#%%
plt.figure()
lefts0 = []
rights0 = []
lefts1 = []
rights1 = []
for i, slit in enumerate(img['slits']):
    plt.plot(img['slits.rad.pix'], slit[0])
    curv0 = find_curvature(img['slits.rad.pix'], slit[0])
    left = signal.argrelextrema(abs(curv0), np.less)[0][0]
    right = signal.argrelextrema(abs(curv0), np.less)[0][-1]
    lefts0.append([left, img['slits.angle'][i]])
    rights0.append([right, img['slits.angle'][i]+np.pi])
    plt.axvline(img['slits.rad.pix'][left], color='y')
    plt.axvline(img['slits.rad.pix'][right])
    plt.plot(img['slits.rad.pix'], slit[1])
    curv1 = find_curvature(img['slits.rad.pix'], slit[1])
    left = signal.argrelextrema(abs(curv1), np.less)[0][0]
    right = signal.argrelextrema(abs(curv1), np.less)[0][-1]
    lefts1.append([left, img['slits.angle'][i]+np.pi/2.])
    rights1.append([right, img['slits.angle'][i]+3.*np.pi/2.])
    plt.axvline(img['slits.rad.pix'][left], color='red')
    plt.axvline(img['slits.rad.pix'][right], color='g')
plt.show()
#%%
sizem=2.
fig = plt.figure()
for left in lefts0:
    plt.scatter(img['slits.rad.pix'][left[0]]*np.cos(left[1])+256, img['slits.rad.pix'][left[0]]*np.sin(left[1])+256, color='y', marker='.', s=sizem)
for right in rights0:
    plt.scatter(-img['slits.rad.pix'][right[0]]*np.sin(right[1])+256, -img['slits.rad.pix'][right[0]]*np.cos(right[1])+256, color='b', marker='.', s=sizem)
for left in lefts1:
    plt.scatter(img['slits.rad.pix'][left[0]]*np.cos(left[1])+256, img['slits.rad.pix'][left[0]]*np.sin(left[1])+256, color='r', marker='.', s=sizem)
for right in rights1:
    plt.scatter(img['slits.rad.pix'][right[0]]*np.sin(right[1])+256, img['slits.rad.pix'][right[0]]*np.cos(right[1])+256, color='g', marker='.', s=sizem)
plt.xlim(0, 512)
plt.ylim(0, 512)
plt.gca().set_aspect('equal', adjustable='box')
# ax[0].set_aspect('equal')
plt.show()
#%%
from astropy.visualization import PowerStretch
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1, projection='3d')
theta, r = np.meshgrid(img['slits.angle'], img['slits.rad.pix'])
ax2.plot_surface(r, theta, img['slits'][:, 0, :].T,
                 linewidth=0, alpha=0.5, cmap='Blues',
                 norm=ImageNormalize(stretch=PowerStretch(a=10)))
ax2.plot_surface(r, theta, img['slits'][:, 1, :].T,
                 linewidth=0, alpha=0.5, cmap='Reds',
                 norm=ImageNormalize(stretch=PowerStretch(a=10)))
ax2.set_zlim(bottom=ax2.get_zlim()[1], top=ax2.get_zlim()[0])
ax2.view_init(elev=20, azim=90)

plt.show()

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
