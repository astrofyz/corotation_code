from astropy.modeling.models import Sersic2D, Disk2D
from prep_images import *
import pandas as pd
from scipy.ndimage import shift
from astropy.wcs import wcs
from astropy.convolution import Gaussian1DKernel, convolve

def bar_model():
    x,y = np.meshgrid(np.arange(512), np.arange(512))

    mod = Sersic2D(amplitude=5, r_eff=15, n=4, x_0=256, y_0=256,
                   ellip=.5, theta=1)

    mod_disk = Disk2D(amplitude=0.5, x_0=256, y_0=256, R_0=150)
    img = mod(x, y) + mod_disk(x,y)
    log_img = np.log10(img)

    plt.figure()
    plt.imshow(log_img, origin='lower', interpolation='nearest',
               vmin=-1, vmax=2)
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    cbar.set_label('Log Brightness', rotation=270, labelpad=25)
    cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
    plt.show()

    return log_img



all_table = pd.read_csv('/media/mouse13/My Passport/corotation/buta_gal/all_table_buta_rad_astrofyz.csv')
path = '/media/mouse13/My Passport/corotation/buta_gal/image'
out_path = '/media/mouse13/My Passport/corotation_code/data/check_fourier/'

# gal_name = '587738946131132437'
# gal_name = '588017566556225638'
# gal_name = '587732048403824840'
# gal_name = '587741490906398723'
# gal_name = '587736804008722435'
# gal_name = '588848898849112176'
gal_name = '588011124118585393'
# gal_name = '587741490893684878'
# gal_name = '587739707948204093'
# gal_name = '588007004191326250'
# gal_name = '587732771864182806'
# gal_name = '587736584429306061'
# gal_name = '587729150383161562'
# gal_name = '587742551759257682'
# gal_name = '587724648720826467'
# gal_name = '587735349636300832'
# gal_name = '587737827288809605'
# gal_name = '587729150383095831'
# gal_name = '588017990689751059'

title_name, title_ra, title_dec = all_table.loc[all_table.objid14 == int(gal_name), ['name', 'ra', 'dec']].values[0]
title = f"{title_name} \nra={title_ra}, dec={title_dec}"

r_obj, r_cat, r_real, r_seg = read_images(gal_name, type=['obj', 'cat', 'real', 'seg'], band='r', path=path)
seeing_g, seeing_i, seeing_r, seeing_u, seeing_z = all_table.loc[all_table.objid14 == int(gal_name),
                                                 ['seeing_g', 'seeing_i', 'seeing_r', 'seeing_u', 'seeing_z']].values[0]
zp_g, zp_i, zp_r, zp_u, zp_z = zeropoint(name=[gal_name], band=['g', 'i', 'r', 'u', 'z'], table=all_table)[0]
petro_r, petro50_r = all_table.loc[all_table.objid14 == int(gal_name), ['petroRad_r', 'petroR50_r']].values[0]

w = wcs.WCS(r_real[0].header)
ra_real, dec_real = all_table.loc[all_table.objid14 == int(gal_name), ['ra', 'dec']].values[0]
x_real, y_real = w.wcs_world2pix(ra_real, dec_real, 1)

xc, yc = [int(dim/2) for dim in np.shape(r_real[0].data)]

mask_r = main_obj(cat=r_cat, mask=r_seg[0].data, xy=[x_real, y_real])
r_mask_sh = shift(mask_r, [yc-y_real, xc-x_real], mode='nearest')
r_real_sh = shift(r_real[0].data, [yc-y_real, xc-x_real], mode='nearest')
bkg_r = calc_bkg(r_real_sh, shift(r_seg[0].data, [yc-y_real, xc-x_real], mode='nearest'))
real_bg_r = r_real_sh - bkg_r.background
conv_rms = Gaussian1DKernel(stddev=bkg_r.background_rms_median)
real_mag_r = to_mag(image=real_bg_r, zp=zp_r)

r_max, r_min, step_FD = find_outer(r_mask_sh, [xc, yc], title=title, figname=gal_name, path=out_path,
                                   petro=petro_r, petro50=petro50_r)
r_max = r_max*1.3
r_min = r_min

step = 1.2
width = 3.5
par, per = slit(real_mag_r, 1.2, 3.5, [256, 256], r_max, 0., title=title, figname=gal_name, path=out_path)

# попробуем посчитать вычеты по углу:
pa_space = np.linspace(0, np.pi/2., 10)

residual, residual_conv = mult_slit(real_mag_r, pa_space, int(len(par[0])/2), r_max, step, width, title=title, figname=gal_name,
                                    path=out_path, conv=conv_rms, dir=out_path+'slit_im_resid/'+gal_name+'/')

idx = np.argmax([sum(abs(row)) for row in residual_conv])
if sum(residual_conv[idx]) > 0:
    angle_max = pa_space[idx]
else:
    angle_max = (pa_space[idx] + np.pi/2.)

plt.figure()
plt.title('residuals convolved')
for i in range(len(pa_space)):
    plt.plot(par[0][:int(len(par[0])/2)]*0.396, residual_conv[i], label=np.round(pa_space[i], 2))
plt.legend()
# plt.savefig(out_path+'slit_im_resid/'+gal_name+'/'+'res_conv_'+gal_name+'.png', dpi=92)
plt.show()

bar = bar_model()
residual_mod, residual_conv_mod = mult_slit(bar, pa_space, int(len(par[0])/2), r_max, step, width, title=title, figname=gal_name,
                                            path=out_path, conv=conv_rms, dir=out_path+'slit_im_resid/'+gal_name+'/')

idx = np.argmax([sum(abs(row)) for row in residual_conv_mod])
if sum(residual_conv_mod[idx]) > 0:
    angle_max = pa_space[idx]
else:
    angle_max = (pa_space[idx] + np.pi/2.)

plt.figure()
plt.title('residuals convolved model')
for i in range(len(pa_space)):
    plt.plot(par[0][:int(len(par[0])/2)]*0.396, residual_conv_mod[i], label=np.round(pa_space[i], 2))
plt.legend()
# plt.savefig(out_path+'slit_im_resid/'+gal_name+'/'+'res_conv_'+gal_name+'.png', dpi=92)
plt.show()
