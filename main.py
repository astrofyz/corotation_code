from prep_images import *
import pandas as pd
from scipy.interpolate import splrep, splev
from scipy.ndimage import shift
from astropy.wcs import wcs
from scipy.signal import argrelextrema
import scipy.signal as signal
import csv
import numpy.ma as ma

# all_table = pd.read_csv('../corotation/clear_outer/all_table1.csv')
# all_table = pd.read_csv('/media/mouse13/My Passport/corotation/buta_gal/all_table_buta_rad_astrofyz.csv')
all_table = pd.read_csv('../corotation/buta_gal/all_table_buta_rad_astrofyz.csv')

path = '../corotation/buta_gal/image'
out_path = '/home/mouse13/corotation_code/data/'

# path = '/media/mouse13/My Passport/corotation/buta_gal/image'
# out_path = '/media/mouse13/My Passport/corotation_code/data/'

# print(all_table.columns)

# gal_name = '1237651539800293493'
# gal_name = '588007004191326250'
# gal_name = '587742551759257682'
# gal_name = '587732771864182806'
# gal_name = '587724648720826467'
# gal_name = '588848898849112176'
# gal_name = '587739707948204093'
# gal_name = '587735349636300832'
gal_name = '588011124118585393'
# gal_name = '587737827288809605'
# gal_name = '587741490906398723'
# gal_name = '587732048403824840'
# gal_name = '587738946131132437'
# gal_name = '587736584429306061'
# gal_name = '587729150383095831'
# gal_name = '587729150383161562'
# gal_name = '587741490893684878'
# gal_name = '587736804008722435'
# gal_name = '588017566556225638'
# gal_name = '587726033334632485'
# gal_name = '588017990689751059'

title_name, title_ra, title_dec = all_table.loc[all_table.objid14 == int(gal_name), ['name', 'ra', 'dec']].values[0]
title = f"{title_name} \nra={title_ra}, dec={title_dec}"

### READING FILES ######################################################################################################
r_obj, r_cat, r_real, r_seg = read_images(gal_name, type=['obj', 'cat', 'real', 'seg'], band='r', path=path)
g_real, u_real, i_real, z_real = read_images(gal_name, type=['real'], band=['g', 'u', 'i', 'z'], path=path)
g_seg, u_seg, i_seg, z_seg = read_images(gal_name, type=['seg'], band=['g', 'u', 'i', 'z'], path=path)
g_cat, u_cat, i_cat, z_cat = read_images(gal_name, type=['cat'], band=['g', 'u', 'i', 'z'], path=path)

seeing_g, seeing_i, seeing_r, seeing_u, seeing_z = all_table.loc[all_table.objid14 == int(gal_name),
                                                 ['seeing_g', 'seeing_i', 'seeing_r', 'seeing_u', 'seeing_z']].values[0]
zp_g, zp_i, zp_r, zp_u, zp_z = zeropoint(name=[gal_name], band=['g', 'i', 'r', 'u', 'z'], table=all_table)[0]
petro_r, petro50_r = all_table.loc[all_table.objid14 == int(gal_name), ['petroRad_r', 'petroR50_r']].values[0]

### COORD FROM WCS #####################################################################################################
w = wcs.WCS(r_real[0].header)
ra_real, dec_real = all_table.loc[all_table.objid14 == int(gal_name), ['ra', 'dec']].values[0]
x_real, y_real = w.wcs_world2pix(ra_real, dec_real, 1)
print('coords = ', x_real, y_real)

xc, yc = [int(dim/2) for dim in np.shape(r_real[0].data)]

### CORRECT MASK #######################################################################################################
mask_r = main_obj(cat=r_cat, mask=r_seg[0].data, xy=[x_real, y_real])
mask_g = main_obj(cat=g_cat, mask=g_seg[0].data, xy=[x_real, y_real])
mask_i = main_obj(cat=i_cat, mask=i_seg[0].data, xy=[x_real, y_real])
mask_u = main_obj(cat=u_cat, mask=u_seg[0].data, xy=[x_real, y_real])
mask_z = main_obj(cat=z_cat, mask=z_seg[0].data, xy=[x_real, y_real])

### SHIFT TO THE CENTER ################################################################################################
r_mask_sh = shift(mask_r, [yc-y_real, xc-x_real], mode='nearest')
g_mask_sh = shift(mask_g, [yc-y_real, xc-x_real], mode='nearest')
u_mask_sh = shift(mask_u, [yc-y_real, xc-x_real], mode='nearest')
i_mask_sh = shift(mask_i, [yc-y_real, xc-x_real], mode='nearest')
z_mask_sh = shift(mask_z, [yc-y_real, xc-x_real], mode='nearest')

r_real_sh = shift(r_real[0].data, [yc-y_real, xc-x_real], mode='nearest')
g_real_sh = shift(g_real[0].data, [yc-y_real, xc-x_real], mode='nearest')
i_real_sh = shift(i_real[0].data, [yc-y_real, xc-x_real], mode='nearest')
u_real_sh = shift(u_real[0].data, [yc-y_real, xc-x_real], mode='nearest')
z_real_sh = shift(z_real[0].data, [yc-y_real, xc-x_real], mode='nearest')

### BACKGROUND AND CONVERSION TO MAG ###################################################################################
bkg_r = calc_bkg(r_real_sh, shift(r_seg[0].data, [yc-y_real, xc-x_real], mode='nearest'))
bkg_i = calc_bkg(i_real_sh, shift(i_seg[0].data, [yc-y_real, xc-x_real], mode='nearest'))
bkg_u = calc_bkg(u_real_sh, shift(u_seg[0].data, [yc-y_real, xc-x_real], mode='nearest'))
bkg_g = calc_bkg(g_real_sh, shift(g_seg[0].data, [yc-y_real, xc-x_real], mode='nearest'))
bkg_z = calc_bkg(z_real_sh, shift(z_seg[0].data, [yc-y_real, xc-x_real], mode='nearest'))

real_bg_r = r_real_sh - bkg_r.background
real_bg_g = g_real_sh - bkg_g.background
real_bg_u = u_real_sh - bkg_u.background
real_bg_i = i_real_sh - bkg_i.background
real_bg_z = z_real_sh - bkg_z.background

real_mag_r = to_mag(image=real_bg_r, zp=zp_r)
real_mag_g = to_mag(image=real_bg_g, zp=zp_g)
real_mag_u = to_mag(image=real_bg_u, zp=zp_u)
real_mag_i = to_mag(image=real_bg_i, zp=zp_i)
real_mag_z = to_mag(image=real_bg_z, zp=zp_z)

### CORRECT FWHM #######################################################################################################
giruz_fwhm = []
max_seeing = max([seeing_g, seeing_i, seeing_r, seeing_u, seeing_z])
for im, fwhm in zip([g_real[0].data, i_real[0].data, r_real[0].data, u_real[0].data, z_real[0].data],
                    [seeing_g, seeing_i, seeing_r, seeing_u, seeing_z]):
    if fwhm != max_seeing:
        giruz_fwhm.append(common_FWHM(im, fwhm, max_seeing))
    else:
        giruz_fwhm.append(im)

### MAX_RADIUS #########################################################################################################
r_max, r_min, step_FD = find_outer(r_mask_sh, [xc, yc], title=title, figname=gal_name, path=out_path,
                                   petro=petro_r, petro50=petro50_r)
r_max = r_max*1.3
r_min = r_min
print('r_max = ', r_max, '(pix)', r_max*0.396,  '(arcsec)')
print('r_min = ', r_min, '(pix)', r_min*0.396, '(arcsec)')

### FITTING ELLIPSE ####################################################################################################

eps = 0
pa = np.pi

image_for_fit = ma.masked_array(r_real_sh, mask=np.ones_like(r_mask_sh)-r_mask_sh)
try:  # нужен ли step?
    eps, pa = ellipse_fit(image=r_real_sh, x=xc, y=yc, rmax=r_max,
                      eps=np.sqrt(1-(r_cat[1].data.T[0]['B_IMAGE']/r_cat[1].data.T[0]['A_IMAGE'])**2),
                      theta=r_cat[1].data.T[0]['THETA_IMAGE'], step=0.4, rmin=petro_r,
                      title=title, figname=gal_name, path=out_path)
except:
    try:
        eps, pa = ellipse_fit(image=image_for_fit, x=256, y=256, fflag=0.1,
                              eps=np.sqrt(1 - (r_cat[1].data.T[0]['B_IMAGE'] / r_cat[1].data.T[0]['A_IMAGE']) ** 2),
                              theta=r_cat[1].data.T[0]['THETA_IMAGE'], step=0.2, maxgerr=0.7, rmin=petro50_r,
                              title=title, figname=gal_name, path=out_path)
    except:
        print('No fit neither with petroRad nor petro50')
#         # здесь должна быть ошибка

### тут надо подумать, можно ли сделать это лучше - флаг, максимальный радиус, минимальный радиус

step = step_FD

sma_pix_r, sb_r = calc_sb(real_mag_r, step=step, rmax=r_max, x=xc, y=xc, eps=0.,
                          sma=r_cat[1].data.T[0]['X_IMAGE'], theta=r_cat[1].data.T[0]['THETA_IMAGE'])

sma_pix_g, sb_g = calc_sb(real_mag_g, step=step, rmax=r_max, x=xc, y=yc, eps=0.,
                          sma=g_cat[1].data.T[0]['X_IMAGE'], theta=g_cat[1].data.T[0]['THETA_IMAGE'])

sma_pix_u, sb_u = calc_sb(real_mag_u, step=step, rmax=r_max, x=xc, y=yc, eps=0.,
                          sma=u_cat[1].data.T[0]['X_IMAGE'], theta=u_cat[1].data.T[0]['THETA_IMAGE'])

sma_pix_i, sb_i = calc_sb(real_mag_i, step=step, rmax=r_max, x=xc, y=yc, eps=0.,
                          sma=i_cat[1].data.T[0]['X_IMAGE'], theta=i_cat[1].data.T[0]['THETA_IMAGE'])

sma_pix_z, sb_z = calc_sb(real_mag_z, step=step, rmax=r_max, x=xc, y=yc, eps=0.,
                          sma=z_cat[1].data.T[0]['X_IMAGE'], theta=z_cat[1].data.T[0]['THETA_IMAGE'])

sma_pix_g_i, sb_g_i = calc_sb(real_mag_g-real_mag_i, step=step, rmax=r_max, x=xc, y=yc, eps=0.,
                              sma=g_cat[1].data.T[0]['X_IMAGE'], theta=g_cat[1].data.T[0]['THETA_IMAGE'])

sma_pix_r_i, sb_r_i = calc_sb(real_mag_r-real_mag_i, step=step, rmax=r_max, x=xc, y=yc, eps=0.,
                              sma=r_cat[1].data.T[0]['X_IMAGE'], theta=r_cat[1].data.T[0]['THETA_IMAGE'])

sma_pix_g_r, sb_g_r = calc_sb(real_mag_g-real_mag_r, step=step, rmax=r_max, x=xc, y=yc, eps=0.,
                              sma=r_cat[1].data.T[0]['X_IMAGE'], theta=r_cat[1].data.T[0]['THETA_IMAGE'])

sma_pix_u_g, sb_u_g = calc_sb(real_mag_u-real_mag_g, step=step, rmax=r_max, x=xc, y=yc, eps=0.,
                              sma=g_cat[1].data.T[0]['X_IMAGE'], theta=g_cat[1].data.T[0]['THETA_IMAGE'])

bg_mag = calc_bkg(real_mag_r, mask_r).background_median
print('number of apertures', len(sb_r))

mag_max = np.amax(np.concatenate([sb_r, sb_i, sb_g, sb_z, sb_u]))
mag_min = np.amin(np.concatenate([sb_r, sb_i, sb_g, sb_z, sb_u]))

par_r = find_parabola(sma_pix_r, sb_r, s=0.1, path=out_path, figname=gal_name, grad=True, smooth=0.03)
par_g = find_parabola(sma_pix_g, sb_g, s=0.1, path=out_path, figname=gal_name, grad=True, smooth=0.03)
par_i = find_parabola(sma_pix_i, sb_i, s=0.3, path=out_path, figname=gal_name, grad=True, smooth=0.03)
par_z = find_parabola(sma_pix_z, sb_z, s=0.1, path=out_path, figname=gal_name, grad=True, smooth=0.03)

rad_r = par_r[0][np.argmax(par_r[1])]
rad_g = par_g[0][np.argmax(par_g[1])]
rad_i = par_i[0][np.argmax(par_i[1])]
rad_z = par_z[0][np.argmax(par_z[1])]

print('radii ', rad_r, rad_g, rad_i, rad_z)

f, (a_all, a_gi, a_ri, a_gr, a_ug) = plt.subplots(5, 1, gridspec_kw={'height_ratios': [8, 1, 1, 1, 1]}, sharex=True,
                                                  figsize=(8, 10))

a_all.set_title(title)

a_all.plot(sma_pix_r*0.396, sb_r, color='red',  label='r  '+str(np.round(rad_r, 3)))
a_all.plot(sma_pix_g*0.396, sb_g, color='blue', label='g  '+str(np.round(rad_g, 3)))
a_all.plot(sma_pix_i*0.396, sb_i, color='gold', label='i  '+str(np.round(rad_i, 3)))
a_all.plot(sma_pix_z*0.396, sb_z, color='g',    label='z  '+str(np.round(rad_z, 3)))
a_all.plot(sma_pix_u*0.396, sb_u, label='u', color='m')

a_all.plot(par_r[0], par_r[1], color='k')
a_all.plot(par_g[0], par_g[1], color='k')
a_all.plot(par_z[0], par_z[1], color='k')
a_all.plot(par_i[0], par_i[1], color='k')

a_all.axvline(par_r[0][np.argmax(par_r[1])], color='maroon')
a_all.axvline(par_g[0][np.argmax(par_g[1])], color='navy')
a_all.axvline(par_i[0][np.argmax(par_i[1])], color='sienna')
a_all.axvline(par_z[0][np.argmax(par_z[1])], color='darkgreen')

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
plt.savefig(out_path+'sb_profile/'+gal_name+'_sb_prof.png')
plt.show()

par, per = slit(real_mag_r, .7, 2.5, [256, 256], r_max, pa, title=title, figname=gal_name, path=out_path)

N = 2
Wn = 0.1
b, a = signal.butter(N, Wn)
par_filt = signal.filtfilt(b, a, par[1], padlen=150)
per_filt = signal.filtfilt(b, a, per[1], padlen=150)

plt.figure(figsize=(6, 4))
plt.plot(par[0]*0.396, par[1], color='skyblue', lw=5, alpha=0.5, label='parallel')
plt.plot(par[0]*0.396, par_filt, color='navy', label=str(N)+' ; '+str(Wn))
plt.plot(per[0]*0.396, per[1], color='lightsalmon', lw=5, alpha=0.5, label='perpendicular')
plt.plot(per[0]*0.396, per_filt, color='crimson')
plt.gca().invert_yaxis()
plt.title(title)
plt.legend()
plt.xlabel('r (arcsec')
plt.ylabel('$\mu[r] \quad (mag\:arcsec^{-2})$')
plt.savefig(out_path+'slit/'+gal_name+'_slit.png')
plt.show()


rot_sca_r = rotate_and_scale(real_mag_r, pa, sx=1., sy=1./np.sqrt(1-eps**2))

plt.figure()
plt.imshow(rot_sca_r, origin='lower', cmap='Greys')
plt.title(title)
plt.savefig(out_path+'rot_scale_image/' + gal_name + '_rs.png')
plt.show()

print('hey')
with open(out_path+'result.csv', 'a', newline='') as csvfile:
    res_writer = csv.writer(csvfile, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    res_writer.writerow(['name : ' + title_name])
    res_writer.writerow(['r_max : ' + str(np.round(r_max, 5))])
    res_writer.writerow(['x_real, y_real : ' + str(np.round(x_real, 3)) + ' ' + str(np.round(y_real, 3))])

    res_writer.writerow(['eps : ' + str(np.round(eps, 5)) + '  ' + str(np.round(np.sqrt(1-eps**2), 3))])
    res_writer.writerow(['PA : ' + str(np.round(pa, 5)) + '  ' + str(np.round(pa*180./np.pi, 3))])
    res_writer.writerow(['number of apertures : '+str(len(sb_r))])

    res_writer.writerow(['corot_rad_r : ' + str(np.round(rad_r, 5))])
    res_writer.writerow(['corot_rad_g : ' + str(np.round(rad_g, 5))])
    res_writer.writerow(['corot_rad_i : ' + str(np.round(rad_i, 5))])
    res_writer.writerow(['corot_rad_z : ' + str(np.round(rad_z, 5))])
    res_writer.writerow(['corot_rad : ' + str(np.round(np.mean([rad_r, rad_g, rad_i, rad_z]), 3)) + '+-'
                         + str(np.round(np.std([rad_r, rad_g, rad_i, rad_z]), 3))])
    res_writer.writerow([title_name, str(step_FD)])
    res_writer.writerow(['....................................................................'])
    csvfile.close()


# unsharp_mask(real_mag_r_sh)

# len_par = int(len(par[0])/2+1)
# print(par[0][-len_par:], len(par[0]))
# grad_par = np.gradient(par_filt[-len_par:], par[0][-len_par:])
# print(grad_par)
#
# plt.figure()
# plt.plot(par[0][-len_par:]*0.396, grad_par)
# idx_min = argrelextrema(grad_par, np.less)
# plt.scatter(par[0][-len_par:][idx_min]*0.396, grad_par[idx_min], color='orange')
# print(idx_min)
# print(par[0][-len_par:][idx_min]*0.396)
# plt.show()
#
# plt.figure()
# plt.scatter(par[0][-len_par:]*0.396, par_filt[-len_par:], color='navy', lw=3)
#
# dlt = 3.5
# for x, y, df in zip(par[0][-len_par:], par_filt[-len_par:], grad_par):
#     xr = np.linspace(x-dlt, x+dlt, 10)
#     yr = y + df*(xr-x)
#     plt.plot(xr*0.396, yr, lw=1, linestyle='dashed', color='orange')
# plt.scatter(par[0][-len_par:][idx_min]*0.396, par_filt[-len_par:][idx_min], color='red')
# plt.gca().invert_yaxis()
# plt.show()

