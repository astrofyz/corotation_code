import matplotlib.pyplot as plt
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.isophote import EllipseGeometry, Ellipse
from photutils import EllipticalAperture
from prep_images import *
import pandas as pd

all_table = pd.read_csv('../corotation/clear_outer/all_table.csv')

# print(all_table.columns)

# print(all_table.loc[all_table.objid14==1237651539800293493, ['sdss', 'ring', 'tt']])

# gal_name = '1237648720167174259'
# seeing_giruz = np.array([1.764125, 1.500433, 1.50123, 1.875232, 1.51328])  # никаких отличий после свёртки
gal_name = '1237651539800293493'
seeing_giruz = np.array([1.57333, 1.463924, 1.388818, 1.685696, 1.571356])
r_obj, r_aper, r_cat, r_real = read_images(gal_name, type=['obj', 'aper', 'cat', 'real'], band='r', path = '../corotation/clear_outer')
g_real, u_real, i_real, z_real = read_images(gal_name, type=['real'], band=['g', 'u', 'i', 'z'], path = '../corotation/clear_outer')
r_seg, g_seg = read_images(gal_name, type=['seg'], band=['r', 'g'], path = '../corotation/clear_outer')

# zp_r = zeropoint(name=[gal_name], band=['r'], table=all_table)[0][0]

# bkg = calc_bkg(r_real[0].data, r_seg[0].data)

# norm = ImageNormalize(stretch=SqrtStretch())
# plt.figure()
# plt.imshow(bkg.background, origin='lower', cmap='Greys_r', norm=norm)
# plt.show()
#
# print(np.mean(bkg.background), np.median(bkg.background), np.std(bkg.background))

# r_bg = r_real[0].data - bkg.background

# r_obj_mag = to_mag(image=r_bg, zp=zp_r) #, mask=[r_seg[0].data])  # почему вдруг сломалось?

# idxs = np.where(r_seg[0].data == 1)
# idxs_fin = np.isfinite(r_obj_mag[idxs])
# print(np.amin(r_obj_mag[idxs][idxs_fin]))

# plt.figure()
# norm = ImageNormalize(stretch=LogStretch())
# plt.imshow(r_obj_mag, origin='lower', cmap='Greys')
# plt.show()

# average_sb(image = r_real[0].data, cat=r_cat[1].data.T[0])
# r, mu = average_sb(image=r_obj_mag, cat=r_cat[1].data.T[0])
#
# plt.figure()
# plt.plot((r*0.396)[:350], mu[:350])
# plt.ylim(mu.max(), mu.min())
# plt.show()

# print(mu)

# print(r_cat[0])  # read header. header['%tag'] - read specific tag from header
# print(r_cat[1].columns)  # column names
# print(type(r_cat[1].data['X_IMAGE']))
# print(r_cat[1].data['NUMBER'])  # reading catalog; or specific field

# print r_hdu[0].data, np.shape(r_hdu[0].data) # read data (image)

print(ellipse_fit(cat=r_cat[1].data.T[0], image=r_real[0].data, step=0.5))

sma, intens = calc_sb(image=r_real[0].data, cat=r_cat[1].data.T[0], step=0.7)

plt.figure()
norm = ImageNormalize(stretch=LogStretch())
plt.imshow(r_real[0].data, norm=norm, origin='lower', cmap='Greys_r')
plt.show()

plt.figure()
plt.plot(sma, intens)
# plt.ylim(mu.max(), mu.min())
plt.show()


# plt.figure()
# # norm = ImageNormalize(stretch=LogStretch())
# plt.imshow(r_seg[0].data) #, norm=norm, origin='lower', cmap='Greys_r')
# plt.show()

# нужна какая-нибудь хрень, чтобы выбирать из каталога нужный объект, например, вместе с именем давать координаты и
# искать ближайшие
# look through columns:
# for i in (r_cat[1].data['NUMBER']):
#     print(r_cat[1].data['NUMBER'][i-1], r_cat[1].data['X_IMAGE'][i-1], r_cat[1].data['Y_IMAGE'][i-1],
#           r_cat[1].data['A_IMAGE'][i-1], r_cat[1].data['B_IMAGE'][i-1], r_cat[1].data['THETA_IMAGE'][i-1])

# num = 0  # number of object in catalog (from SE)
#
# angle = r_cat[1].data['THETA_IMAGE'][num]
# sx = 1.
# sy = r_cat[1].data['A_IMAGE'][num]/r_cat[1].data['B_IMAGE'][num]
#
#
# r_obj_rs = rotate_and_scale(r_real[0].data, -angle, sx, sy)  # rotate and scale image

# plt.figure()
# norm = ImageNormalize(stretch=LogStretch())
# plt.imshow(r_obj_rs, norm=norm, origin='lower', cmap='Greys_r')
# plt.show()

# что мне вообще нужно от этого эллипса в итоге?! возможно, как раз то, что я беру из SE и использую выше
# в sma должен лежать какой-нибудь из радиусов.

# plt.figure('r_norm')
# norm = ImageNormalize(stretch=LogStretch())
# plt.imshow(r_real[0].data, norm=norm, origin='lower', cmap='Greys_r')
#
# geometry = EllipseGeometry(x0=r_cat[1].data['X_IMAGE'][num], y0=r_cat[1].data['Y_IMAGE'][num],
#                            sma=r_cat[1].data['A_IMAGE'][num], eps=(1-r_cat[1].data['B_IMAGE'][num]/r_cat[1].data['A_IMAGE'][num]),
#                            pa=-r_cat[1].data['THETA_IMAGE'][num]*np.pi/180.)
#
# aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma, geometry.sma*(1 - geometry.eps), geometry.pa)
# aper.plot(color='green')  # initial ellipse guess
#
# ellipse = Ellipse(r_real[0].data, geometry)
# isolist = ellipse.fit_image()
# print(isolist)

# smas = np.linspace(3, 60, 4)
# for sma in smas:
#     iso = isolist.get_closest(sma)
#     print(iso.x0, iso.y0, iso.eps, iso.sma, iso.pa, iso.npix_e, iso.ndata)
#     x, y, = iso.sampled_coordinates()
#     plt.plot(x, y, color='red', lw=1, alpha=0.3)
#     angle1 = iso.pa
# plt.show()
#
# r_obj_rs1 = rotate_and_scale(r_obj[0].data, -angle1, sx, sy)  # obtain rotated and scaled image

# plt.figure()
# norm = ImageNormalize(stretch=LogStretch())
# plt.imshow(r_obj_rs1, norm=norm, origin='lower', cmap='Greys_r')
# plt.show()

# fwhm_max = max(seeing_giruz)
#
# i_obj_fwhm = common_FWHM(i_obj[0].data, seeing_giruz[1], fwhm_max)
#
# plt.figure()
# norm = ImageNormalize(stretch=LogStretch())
# plt.imshow(i_obj[0].data, norm=norm, origin='lower', cmap='Greys_r')
# plt.show()
#
# plt.figure()
# norm = ImageNormalize(stretch=LogStretch())
# plt.imshow(i_obj_fwhm, norm=norm, origin='lower', cmap='Greys_r')
# plt.show()

# после фита эллипсом разворачивается лучше


# bad fitting especially for rings. how to obtain apertures from SE?

# plt.figure('g')
# plt.imshow(g_hdu[0].data, norm=colors.LogNorm(), cmap='binary')

# plt.figure('i-g')
# plt.imshow(g_hdu[0].data-i_hdu[0].data, norm=colors.LogNorm(), cmap='binary')

# plt.figure('g-u')
# plt.imshow(g_hdu[0].data-u_hdu[0].data, norm=colors.LogNorm(), cmap='binary')
# plt.show()

# plt.figure('z-g')
# plt.imshow(z_hdu[0].data-g_hdu[0].data, norm=colors.LogNorm(), cmap='binary')
# plt.show()
#
# mask = np.zeros(np.shape(g_obj[0].data))
# mask1 = np.zeros(np.shape(g_obj[0].data))
# top = np.array([111, 10])
# bottom = np.array([68, 121])
#
# lvec = np.sqrt(np.dot(top - bottom, top - bottom))
# xstep = abs(top[0] - bottom[0]) / lvec
# ystep = abs(top[1] - bottom[1]) / lvec
# for i in range(int(lvec)):
#     x = int(top[0] - i * xstep)
#     y = int(top[1] + i * ystep)
#     mask[y, x] = 1.

# # slower method (971 mu_s vs 2.24 ms)
# x = top[0]
# y = top[1]
# while (x > bottom[0] and y < bottom[1]):
# 	mask1[int(np.round(y)),int(np.round(x))] = 1.
# 	deltax = abs(x - bottom[0])
# 	deltay = abs(y - bottom[1])
# 	deltar = np.sqrt(np.dot(np.array([x,y])-bottom, np.array([x,y])-bottom))
# 	cos_alpha = deltay/deltar
# 	sin_alpha = deltax/deltar
# 	y = y+cos_alpha
# 	x = x-sin_alpha

# plt.figure('mask')
# plt.imshow(mask)
# plt.show()

# idxs = np.where(mask > 0)  # indices of the cut
# id_max = np.argmax(g_obj[0].data[idxs])  # index of the brightest element along the cut
#
# center = np.array(list(zip(idxs[0], idxs[1]))[id_max])  # brightest pixel along the cut
#
# radius = [np.sqrt(np.sum((pix - center) ** 2)) for pix in
#           np.array(list(zip(idxs[0], idxs[1]))[id_max:])]  # range of radii in pixels along the cut
#
# # plt.figure('cut')
# # plt.scatter(radius, g_obj[0].data[(idxs[0][id_max:], idxs[1][id_max:])], marker='.', color='midnightblue')
# # plt.show()
#
#
# # estimating the background
# from astropy.stats import mad_std
#
# bg_sigma_real = mad_std(r_real[0].data)
# print('real bg ', bg_sigma_real)
#
# bg_sigma_obj = mad_std(r_obj[0].data)
# print('obj bg ', bg_sigma_obj)


