from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.isophote import EllipseGeometry
from photutils import EllipticalAperture
from photutils.isophote import Ellipse


r_obj = fits.open('/home/mouse13/corotation/clear_outer/se_frames/r/obj/r1237651226783711239-objects.fits')
g_obj = fits.open('/home/mouse13/corotation/clear_outer/se_frames/g/obj/g1237651226783711239-objects.fits')
i_obj = fits.open('/home/mouse13/corotation/clear_outer/se_frames/i/obj/i1237651226783711239-objects.fits')
u_obj = fits.open('/home/mouse13/corotation/clear_outer/se_frames/u/obj/u1237651226783711239-objects.fits')
z_obj = fits.open('/home/mouse13/corotation/clear_outer/se_frames/z/obj/z1237651226783711239-objects.fits')
r_real = fits.open('/home/mouse13/corotation/clear_outer/r/stamps/r1237651226783711239.fits')
r_aper = fits.open('/home/mouse13/corotation/clear_outer/se_frames/r/aper/r1237651226783711239-apertures.fits')

# print(r_aper[0].header)  #read header. header['%tag'] - read specific tag from header
# print r_hdu[0].data, np.shape(r_hdu[0].data) # read data (image)

# plt.figure('r')
# plt.imshow(r_real[0].data, norm=colors.LogNorm(), cmap='Greys_r', origin='lower') # compare with ds9, rotated?
# plt.show()

# astropy norm looks nicer
plt.figure('r_norm')
norm = ImageNormalize(stretch=LogStretch())
plt.imshow(r_real[0].data, norm=norm, origin='lower', cmap='Greys_r')

geometry = EllipseGeometry(x0=87, y0=68, sma=15, eps=0.3, pa=102.*np.pi/180.)
aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma, geometry.sma*(1 - geometry.eps), geometry.pa)
aper.plot(color='green')

ellipse = Ellipse(r_real[0].data, geometry)
isolist = ellipse.fit_image()
print(isolist.eps)

smas = np.linspace(3, 20, 3)
for sma in smas:
    iso = isolist.get_closest(sma)
    x, y, = iso.sampled_coordinates()
    plt.plot(x, y, color='white', lw=1)
plt.show()

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

mask = np.zeros(np.shape(g_obj[0].data))
mask1 = np.zeros(np.shape(g_obj[0].data))
top = np.array([111, 10])
bottom = np.array([68, 121])

lvec = np.sqrt(np.dot(top - bottom, top - bottom))
xstep = abs(top[0] - bottom[0]) / lvec
ystep = abs(top[1] - bottom[1]) / lvec
for i in range(int(lvec)):
    x = int(top[0] - i * xstep)
    y = int(top[1] + i * ystep)
    mask[y, x] = 1.

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

idxs = np.where(mask > 0)  # indices of the cut
id_max = np.argmax(g_obj[0].data[idxs])  # index of the brightest element along the cut

center = np.array(list(zip(idxs[0], idxs[1]))[id_max])  # brightest pixel along the cut

radius = [np.sqrt(np.sum((pix - center) ** 2)) for pix in
          np.array(list(zip(idxs[0], idxs[1]))[id_max:])]  # range of radii in pixels along the cut

# plt.figure('cut')
# plt.scatter(radius, g_obj[0].data[(idxs[0][id_max:], idxs[1][id_max:])], marker='.', color='midnightblue')
# plt.show()


# estimating the background
from astropy.stats import mad_std

bg_sigma_real = mad_std(r_real[0].data)
print('real bg ', bg_sigma_real)

bg_sigma_obj = mad_std(r_obj[0].data)
print('obj bg ', bg_sigma_obj)
