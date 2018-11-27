from prep_images import *
import pandas as pd
from scipy.interpolate import splrep, splev
from scipy.ndimage import shift
from astropy.wcs import wcs

# all_table = pd.read_csv('../corotation/clear_outer/all_table1.csv')
all_table = pd.read_csv('../corotation/buta_gal/all_table_buta_astrofyz.csv')

path = '../corotation/buta_gal/image'

# print(all_table.columns)

# gal_name = '1237651539800293493'
gal_name = '588007004191326250'

seeing_giruz = all_table.loc[all_table.objid14 == int(gal_name),
                             ['seeing_g', 'seeing_i', 'seeing_r', 'seeing_u', 'seeing_z']].values[0]

r_obj, r_aper, r_cat, r_real, r_seg = read_images(gal_name, type=['obj', 'aper', 'cat', 'real', 'seg'],
                                                  band='r', path=path)
g_real, u_real, i_real, z_real = read_images(gal_name, type=['real'], band=['g', 'u', 'i', 'z'], path=path)
g_seg, u_seg, i_seg, z_seg = read_images(gal_name, type=['seg'], band=['g', 'u', 'i', 'z'], path=path)
g_cat, u_cat, i_cat, z_cat = read_images(gal_name, type=['cat'], band=['g', 'u', 'i', 'z'], path=path)

w = wcs.WCS(r_real[0].header)
ra_real, dec_real = all_table.loc[all_table.objid14 == int(gal_name), ['ra', 'dec']].values[0]
x_real, y_real = w.wcs_world2pix(ra_real, dec_real, 1)

mask_r = main_obj(cat=r_cat, mask=r_seg[0].data, xy=[x_real, y_real])
mask_g = main_obj(cat=g_cat, mask=g_seg[0].data, xy=[x_real, y_real])
mask_i = main_obj(cat=i_cat, mask=i_seg[0].data, xy=[x_real, y_real])
mask_u = main_obj(cat=u_cat, mask=u_seg[0].data, xy=[x_real, y_real])
mask_z = main_obj(cat=z_cat, mask=z_seg[0].data, xy=[x_real, y_real])

giruz_fwhm = []
max_seeing = max(seeing_giruz)
for im, fwhm in zip([g_real[0].data, i_real[0].data, r_real[0].data, u_real[0].data, z_real[0].data], seeing_giruz):
    if fwhm != max_seeing:
        giruz_fwhm.append(common_FWHM(im, fwhm, max_seeing))
    else:
        giruz_fwhm.append(im)

zp_g, zp_i, zp_r, zp_u, zp_z = zeropoint(name=[gal_name], band=['g', 'i', 'r', 'u', 'z'], table=all_table)[0]

bkg_r = calc_bkg(r_real[0].data, r_seg[0].data)
bkg_i = calc_bkg(i_real[0].data, i_seg[0].data)
bkg_u = calc_bkg(u_real[0].data, u_seg[0].data)
bkg_g = calc_bkg(g_real[0].data, g_seg[0].data)
bkg_z = calc_bkg(z_real[0].data, z_seg[0].data)

real_bg_r = r_real[0].data - bkg_r.background
real_bg_g = g_real[0].data - bkg_g.background
real_bg_u = u_real[0].data - bkg_u.background
real_bg_i = i_real[0].data - bkg_i.background
real_bg_z = z_real[0].data - bkg_z.background

vmin = np.amin(abs(real_bg_r))
vmax = np.amax(abs(real_bg_r))

real_mag_r = to_mag(image=real_bg_r, zp=zp_r)
real_mag_g = to_mag(image=real_bg_g, zp=zp_g)
real_mag_u = to_mag(image=real_bg_u, zp=zp_u)
real_mag_i = to_mag(image=real_bg_i, zp=zp_i)
real_mag_z = to_mag(image=real_bg_z, zp=zp_z)

eps, pa = ellipse_fit(cat=r_cat[1].data.T[0], image=r_real[0].data, f=5, step=0.4)
# eps_g, pa_g = ellipse_fit(cat=g_cat[1].data.T[0], image=g_real[0].data)
# eps_u, pa_u = ellipse_fit(cat=u_cat[1].data.T[0], image=u_real[0].data)
# eps_i, pa_i = ellipse_fit(cat=i_cat[1].data.T[0], image=i_real[0].data)
# eps_z, pa_z = ellipse_fit(cat=z_cat[1].data.T[0], image=z_real[0].data)

sma_pix_r, sb_r = calc_sb(real_mag_r, r_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=4., eps=eps)
sma_pix_g, sb_g = calc_sb(real_mag_g, g_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=4., eps=eps)
sma_pix_u, sb_u = calc_sb(real_mag_u, u_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=4., eps=eps)
sma_pix_i, sb_i = calc_sb(real_mag_i, i_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=4., eps=eps)
sma_pix_z, sb_z = calc_sb(real_mag_z, z_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=4., eps=eps)
sma_pix_g_i, sb_g_i = calc_sb(real_mag_g-real_mag_i, r_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0],
                              step=0.4, f_max=4., eps=eps)
sma_pix_r_i, sb_r_i = calc_sb(real_mag_r-real_mag_i, r_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0],
                              step=0.4, f_max=4., eps=eps)
sma_pix_g_r, sb_g_r = calc_sb(real_mag_g-real_mag_r, r_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0],
                              step=0.4, f_max=4., eps=eps)
sma_pix_u_g, sb_u_g = calc_sb(real_mag_u-real_mag_g, r_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0],
                              step=0.4, f_max=4., eps=eps)

bg_mag = calc_bkg(real_mag_r, mask_r).background_median

# mag_max = np.amax(sb_r)
# mag_min = np.amin(sb_r)

mag_max = np.amax(np.concatenate([sb_r, sb_i, sb_g, sb_z, sb_u]))
mag_min = np.amin(np.concatenate([sb_r, sb_i, sb_g, sb_z, sb_u]))

f, (a_all, a_gi, a_ri, a_gr, a_ug) = plt.subplots(5, 1, gridspec_kw={'height_ratios': [8, 1, 1, 1, 1]}, sharex=True,
                                                  figsize=(8, 10))

title_name, title_ra, title_dec = all_table.loc[all_table.objid14 == int(gal_name), ['name', 'ra', 'dec']].values[0]

a_all.set_title(f"{title_name} \nra={title_ra}, dec={title_dec}")

from scipy.signal import argrelextrema

# min_r = argrelextrema(sb_r, np.less)[0]
# max_r = argrelextrema(sb_r, np.greater)[0]
#
# print('rel max', max_r)
# print('rel min', min_r)
# print(sb_r[max_r])

a_all.plot(sma_pix_r*0.396, sb_r, label='r', color='red')
a_all.plot(sma_pix_g*0.396, sb_g, label='g', color='blue')
a_all.plot(sma_pix_u*0.396, sb_u, label='u', color='m')
a_all.plot(sma_pix_i*0.396, sb_i, label='i', color='gold')
a_all.plot(sma_pix_z*0.396, sb_z, label='z', color='g')

tck = splrep(sma_pix_r, sb_r, s=0.3)
ynew = splev(sma_pix_r, tck, der=0)
a_all.scatter(sma_pix_r*0.396, ynew, marker='.', color='orange')
a_all.axhline(bg_mag, label='bg_r')

max_r = argrelextrema(ynew, np.less)[0]  # magnitude!
min_r = argrelextrema(ynew, np.greater)[0]

print('rel max', max_r)
print('rel min', min_r)
print(sb_r[max_r])

a_all.scatter(sma_pix_r[max_r]*0.396, sb_r[max_r], color='slateblue', label='max')
a_all.scatter(sma_pix_r[min_r]*0.396, sb_r[min_r], color='olive', label='min')

a_all.set_ylim(mag_max, mag_min)
a_all.legend()
a_all.set_ylabel('$\mu[u,g,r,i] \quad (mag\:arcsec^{-2})$')

a_gi.plot(sma_pix_g_i*0.396, sb_g_i)
a_gi.set_ylabel('$g-i$')

a_ri.plot(sma_pix_r_i*0.396, sb_r_i)
a_ri.set_ylabel('$r-i$')

a_gr.plot(sma_pix_g_r*0.396, sb_g_r)
a_gr.set_ylabel('$g-r$')

a_ug.plot(sma_pix_u_g*0.396, sb_u_g)
a_ug.set_ylabel('$u-g$')
a_ug.set_xlabel('r (arcsec)')
plt.show()

real_mag_r_sh = shift(real_mag_r, [256-y_real, 256-x_real], mode='nearest')

rot_r = rotate_and_scale(real_mag_r_sh, angle=pa, sx=1., sy=1.)
# print('max and min values for rotated and original images')
# print(np.amax(rot_r), np.amax(real_mag_r))
# print(np.amin(rot_r), np.amin(real_mag_r))

vmin_mag = zp_r-2.5*np.log10(vmin/53.907)
vmax_mag = zp_r-2.5*np.log10(vmax/53.907)

par = slit(real_mag_r_sh, .7, 2.5, [256,256], 65, pa)[0]
per = slit(real_mag_r_sh, .7, 2.5, [256,256], 65, pa)[1]

plt.figure(figsize=(10, 7))
plt.plot(par[0]*0.396, par[1], label='parallel')
plt.plot(per[0]*0.396, per[1], label='perpendicular')
plt.axhline(bg_mag)
plt.gca().invert_yaxis()
plt.legend()
plt.show()

par = slit(rot_r, .7, 2.5, [256, 256], 65, 0.)[0]
per = slit(rot_r, .7, 2.5, [256, 256], 65, 0.)[1]

plt.figure(figsize=(10, 7))
plt.plot(par[0]*0.396, par[1], label='parallel')
plt.plot(per[0]*0.396, per[1], label='perpendicular')
plt.axhline(bg_mag)
plt.gca().invert_yaxis()
plt.legend()
plt.show()

plt.figure()
plt.imshow(real_mag_r, origin='lower', cmap='Greys')
plt.show()

plt.figure()
plt.imshow(rot_r, origin='lower', cmap='Greys')
plt.show()
