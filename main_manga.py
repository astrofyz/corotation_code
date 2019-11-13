#%%
import mod_read
import mod_analysis
from mod_read import *
from mod_analysis import *
from contextlib import contextmanager
import importlib
import os
import matplotlib
# matplotlib.use('Agg')


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


path_table = '/media/mouse13/My Passport/corotation/manga/dr14_zpt02_zpt06_lgmgt9_MANGA_barflag.cat'
dirbase = '/media/mouse13/My Passport/corotation/manga/'
im_path = '/media/mouse13/My Passport/corotation/manga/input/'
out_path = '/media/mouse13/My Passport/corotation/manga/pics/'

names = [elem.split('.')[0] for elem in os.listdir(im_path)]

# print(len(names))

#
# all_table = pd.read_table(path_table, sep=' ')
# all_table.loc[all_table.objID == int(names[0]), ['ra', 'dec']].values[0][0]

# # RUN SExtractor........................................................................................................
# # dirbase_correct = dirbase.replace("\\ ", " ")
# os.system(f'touch {dirbase}makeSEscript.sh')
# fn = open(f'{dirbase}makeSEscript.sh', 'w')
# fn.write('#!bin/bash\n')
# for name in names:
#     fn.write(f'./sextractFrameCustom -d se_input/ input/{name}.fits\n')
# fn.close()
# os.chdir(dirbase)
# os.system('sh makeSEscript.sh')
# #.......................................................................................................................

# from time import time
# start = time()
# dn = 20
# n = len(names)
# for chunk in range(0, n, dn):
#     dc = min(chunk+dn, n)
#     images = make_images(names=names[chunk:dc], bands=['z'], types=['seg', 'real', 'cat'], path=dirbase, path_table=path_table, manga=True)
#     for i in range(len(images)):
#         calc_slit(images[i]['z'], 60, convolve=True, mask=True, petro=True)  #перестать делать это кейвордами
#         images[i]['z'].plot_slits(n_slit=60, rotate=True, savename=out_path+'petro_'+str(images[i]['objID'])+'.png')
#
# print(time() - start)
images = make_images(names=names[0:2], bands=['z'], types=['seg', 'real', 'cat'], path=dirbase, path_table=path_table, manga=True)
importlib.reload(mod_read)
calc_slit(images[1]['z'], 60, convolve=True, mask=True)  #перестать делать это кейвордами
images[1]['z'].plot_slits(n_slit=60, cut=True, savename=out_path+'ringrad_'+str(images[1]['objID'])+'.png')

#%%
# main_obj_mask = main_obj(images[1]['z']['cat'], images[1]['z']['seg'], xy=[256, 256])
# with figure() as fig:
#     img_rot = rotate_and_scale(images[0]['z']['real'], images[0]['z']['angle.max'])
#     plt.imshow(img_rot, origin='lower', norm=ImageNormalize(stretch=LogStretch()))
for i in range(2):
    with figure() as fig:
        plt.imshow(images[i]['z']['real'], origin='lower', norm=ImageNormalize(stretch=LogStretch()))


#%%
# print(images[0]['z']['residuals'])
idx = np.argmax([sum(abs(row)) for row in images[1]['z']['residuals']])
print(images[1]['z']['residuals'][idx])

#%%
print(images[1]['z']['bg'].background_median)
print(np.mean(images[1]['z']['total_error']))
#%%
residual = images[1]['z']['residuals'][idx]
rad = images[1]['z']['slits.rad.pix']
err = np.mean(images[1]['z']['total_error'])

from scipy.optimize import minimize_scalar

minimize_scalar(lambda x: abs(interp1d(rad, residual)(x) - err), method='Bounded', bounds=[min(rad), max(rad)])

#%%
# def fourier_harmonics(image, harmonics=[1, 2, 3, 4], sig=5, plot=False, **kw):
#     image_work = np.zeros_like(image['real'])
#     image_work[:] = image['real'][:]
#     value = np.sqrt(((image_work.shape[0] / 2.0) ** 2.0) + ((image_work.shape[1] / 2.0) ** 2.0))
#
#     polar_image = cv2.linearPolar(image_work, (image_work.shape[0] / 2, image_work.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)
#     # print(type(polar_image), np.shape(polar_image))
#
#     # norm = ImageNormalize(stretch=LogStretch())
#     plt.figure()
#     plt.imshow(polar_image, origin='lower', cmap='Greys')
#     ticks = np.linspace(0, image_work.shape[1], 10)  # y or x len in case of non-square image?
#     plt.yticks(ticks, [str(np.round(tick * 2. * np.pi / image_work.shape[1], 1)) for tick in ticks])
#     plt.show()
#
#     # r_range = np.linspace(0, nx, 50)
#     # phi_range = np.linspace(0, 2 * np.pi, 150)
#
#     if all(['r.' not in key.lower() for key in image.keys()]):
#         if ('seg' in image.keys()) & ('petro' not in kw):  #change centered to without .center
#             try:
#                 image.prop(['r.max.pix', 'r.min.pix', 'FD'], data=find_outer(image['seg'])[1:])
#             except:
#                 image['r.max.pix'] = image['petroR90'] * 3
#         elif ('seg' not in image.keys()) or ('petro' in kw):
#             image['r.max.pix'] = image['petroR90']  # or petroR90 * 2.; check .prop()
#
#     len_I = int(image['r.max.pix'])
#     I = np.zeros((len(harmonics), len_I))
#
#     j = 0
#     for r in range(sig, len_I-sig):
#         # data_r = polar_image[:, r]
#         data_r = [np.mean(row) for row in polar_image[:, r-sig:r+sig]]
#         data_fft = fft.dct(data_r)
#         i = 0
#         for harmonic in harmonics:
#             I[i][j] = abs(data_fft[harmonic])/abs(data_fft[0])
#             i += 1
#         j += 1
#         # if r == 40:
#         #     freq = fft.fftfreq(len(data_r), 1. / len(data_r))
#         #     nx = image.shape[0]
#         #     plt.figure()
#         #     plt.plot(np.linspace(0, nx, nx) * 2. * np.pi / nx, polar_image[:, r])
#         #     plt.plot(np.linspace(0, nx, nx) * 2. * np.pi / nx, 1. / nx * sum(
#         #         [data_fft[i] * np.cos(freq[i] * np.linspace(0, nx, nx) * np.pi / nx) for i in range(len(data_fft))]))
#         #     plt.show()
#
#     if 'plot':
#         plt.figure()
#         for i in range(len(harmonics)):
#             plt.plot(np.linspace(0, len_I, len_I)*0.396, I[i], label=harmonics[i])
#         plt.legend()
#         if 'savename' in kw:
#             plt.savefig(kw['savename'])
#         plt.show()
#         plt.close()
#     image.prop('fourier.harm', data=I)
#     return I

# fourier_harmonics(images[1]['z'], [2, 4], plot=True)