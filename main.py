from prep_images import *
import pandas as pd

all_table = pd.read_csv('../corotation/clear_outer/all_table1.csv')

# print(all_table.columns)

gal_name = '1237651539800293493'
seeing_giruz = all_table.loc[all_table.objid14==int(gal_name), ['seeing_g', 'seeing_i', 'seeing_r', 'seeing_u', 'seeing_z']].values[0]

r_obj, r_aper, r_cat, r_real, r_seg = read_images(gal_name, type=['obj', 'aper', 'cat', 'real', 'seg'], band='r', path = '../corotation/clear_outer')
g_real, u_real, i_real, z_real = read_images(gal_name, type=['real'], band=['g', 'u', 'i', 'z'], path = '../corotation/clear_outer')
g_seg, u_seg, i_seg, z_seg = read_images(gal_name, type=['seg'], band=['g', 'u', 'i', 'z'], path = '../corotation/clear_outer')
g_cat, u_cat, i_cat, z_cat = read_images(gal_name, type=['cat'], band=['g', 'u', 'i', 'z'], path = '../corotation/clear_outer')

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

real_mag_r = to_mag(image=real_bg_r, zp=zp_r)
real_mag_g = to_mag(image=real_bg_g, zp=zp_g)
real_mag_u = to_mag(image=real_bg_u, zp=zp_u)
real_mag_i = to_mag(image=real_bg_i, zp=zp_i)
real_mag_z = to_mag(image=real_bg_z, zp=zp_z)

print(np.amax(real_mag_r))

eps, pa = ellipse_fit(cat=r_cat[1].data.T[0], image=r_real[0].data)
# eps_g, pa_g = ellipse_fit(cat=g_cat[1].data.T[0], image=g_real[0].data)
# eps_u, pa_u = ellipse_fit(cat=u_cat[1].data.T[0], image=u_real[0].data)
# eps_i, pa_i = ellipse_fit(cat=i_cat[1].data.T[0], image=i_real[0].data)
# eps_z, pa_z = ellipse_fit(cat=z_cat[1].data.T[0], image=z_real[0].data)

sma_pix_r, sb_r = calc_sb(real_mag_r, r_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=2.5, eps=np.sqrt(1 - (r_cat[1].data['B_IMAGE'][0] / r_cat[1].data['A_IMAGE'][0])**2))
sma_pix_g, sb_g = calc_sb(real_mag_g, g_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=2.5, eps=np.sqrt(1 - (r_cat[1].data['B_IMAGE'][0] / r_cat[1].data['A_IMAGE'][0])**2))
sma_pix_u, sb_u = calc_sb(real_mag_u, u_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=2.5, eps=np.sqrt(1 - (r_cat[1].data['B_IMAGE'][0] / r_cat[1].data['A_IMAGE'][0])**2))
sma_pix_i, sb_i = calc_sb(real_mag_i, i_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=2.5, eps=np.sqrt(1 - (r_cat[1].data['B_IMAGE'][0] / r_cat[1].data['A_IMAGE'][0])**2))
sma_pix_z, sb_z = calc_sb(real_mag_z, z_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=2.5, eps=np.sqrt(1 - (r_cat[1].data['B_IMAGE'][0] / r_cat[1].data['A_IMAGE'][0])**2))
sma_pix_g_i, sb_g_i = calc_sb(real_mag_g-real_mag_i, r_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=2.5, eps=np.sqrt(1 - (r_cat[1].data['B_IMAGE'][0] / r_cat[1].data['A_IMAGE'][0])**2))
sma_pix_r_i, sb_r_i = calc_sb(real_mag_r-real_mag_i, r_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=2.5, eps=np.sqrt(1 - (r_cat[1].data['B_IMAGE'][0] / r_cat[1].data['A_IMAGE'][0])**2))
sma_pix_g_r, sb_g_r = calc_sb(real_mag_g-real_mag_r, r_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=2.5, eps=np.sqrt(1 - (r_cat[1].data['B_IMAGE'][0] / r_cat[1].data['A_IMAGE'][0])**2))
sma_pix_u_g, sb_u_g = calc_sb(real_mag_u-real_mag_g, r_cat[1].data.T[0], angle=pa, sma=r_cat[1].data['A_IMAGE'][0], step=0.4,
                          f_max=2.5, eps=np.sqrt(1 - (r_cat[1].data['B_IMAGE'][0] / r_cat[1].data['A_IMAGE'][0])**2))


mag_max = np.amax(np.concatenate([sb_r, sb_u, sb_i, sb_g, sb_z]))
mag_min = np.amin(np.concatenate([sb_r, sb_u, sb_i, sb_g, sb_z]))


f, (a_all, a_gi, a_ri, a_gr, a_ug) = plt.subplots(5, 1, gridspec_kw = {'height_ratios': [8, 1, 1, 1, 1]}, sharex=True, figsize=(8, 10))

# plt.figure()
title_name, title_ra, title_dec = all_table.loc[all_table.objid14==int(gal_name), ['sdss', 'ra', 'dec']].values[0]
# print(title_str)
a_all.set_title(f"{title_name} \nra={title_ra}, dec={title_dec}")
a_all.plot(sma_pix_r*0.396, sb_r, label='r', color='red')
a_all.plot(sma_pix_g*0.396, sb_g, label='g', color='blue')
a_all.plot(sma_pix_u*0.396, sb_u, label='u', color='m')
a_all.plot(sma_pix_i*0.396, sb_i, label='i', color='gold')
a_all.plot(sma_pix_z*0.396, sb_z, label='z', color='g')
a_all.set_ylim(mag_max, mag_min)
a_all.legend()
a_all.set_ylabel('$\mu[u,g,r,i] \quad (mag\:arcsec^{-2})$')
# a_all.set_xlabel('r (arcsec)')
# plt.show()

# plt.figure()
a_gi.plot(sma_pix_g_i*0.396, sb_g_i)
# plt.ylim(sb_g_i.max(), sb_g_i.min())
a_gi.set_ylabel('$g-i$')
# a_gi.set_xlabel('r (arcsec)')
# plt.show()

# plt.figure()
a_ri.plot(sma_pix_r_i*0.396, sb_r_i)
# plt.ylim(sb_r_i.max(), sb_r_i.min())
a_ri.set_ylabel('$r-i$')
# a_ri.set_xlabel('r (arcsec)')
# plt.show()

# plt.figure()
a_gr.plot(sma_pix_r*0.396, sb_g_r)
# plt.ylim(sb_g_r.max(), sb_g_r.min())
a_gr.set_ylabel('$g-r$')
# a_gr.set_xlabel('r (arcsec)')
# plt.show()

# plt.figure()
a_ug.plot(sma_pix_u_g*0.396, sb_u_g)
# plt.ylim(sb_u_g.max(), sb_u_g.min())
a_ug.set_ylabel('$u-g$')
a_ug.set_xlabel('r (arcsec)')
plt.show()

sy = all_table.loc[all_table.objid14==int(gal_name), ['ba']].values[0][0]
print(sy)
r_rot = rotate_and_scale(real_bg_r, angle=pa*np.pi/180., sx=1., sy=1.)

plt.figure()
norm = ImageNormalize(stretch=LogStretch())
plt.imshow(r_rot, norm=norm, origin='lower', cmap='Greys')
plt.show()

# norm = ImageNormalize(stretch=SqrtStretch())
plt.figure()
plt.imshow(real_mag_r, origin='lower', cmap='Greys_r')
plt.show()


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

