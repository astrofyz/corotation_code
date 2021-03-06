#%%
import mod_read
import mod_analysis
from mod_read import *
from mod_analysis import *
from contextlib import contextmanager
import importlib
import os
import matplotlib
matplotlib.use('Agg')


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
#%%
print(len(names))
#%%
#
# all_table = pd.read_table(path_table, sep=' ')
# all_table.loc[all_table.objID == int(names[0]), ['ra', 'dec']].values[0][0]
#%%
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
#%%
from time import time
start = time()
dn = 20
n = len(names)
# n = 5
for chunk in range(0, n, dn):
    dc = min(chunk+dn, n)
    images = make_images(names=names[chunk:dc], bands=['z'], types=['seg', 'real', 'cat'], path=dirbase, path_table=path_table, manga=True)
    for i in range(len(images)):
        calc_slit(images[i]['z'], 60, convolve=True, mask=True, petro=True)  #перестать делать это кейвордами
        images[i]['z'].plot_slits(n_slit=60, rotate=True, savename=out_path+'petro_'+str(images[i]['objID'])+'.png')

print(time() - start)

#%%
print(images)
#%%
main_obj_mask = main_obj(images['z']['cat'], images['z']['seg'], xy=[256, 256])

#%%
with figure() as fig:
    img_rot = rotate_and_scale(images['z']['real'], images['z']['angle.max'])
    plt.imshow(img_rot, origin='lower', norm=ImageNormalize(stretch=LogStretch()))

#%%
for i in range(1):
    with figure() as fig:
        plt.imshow(images[i]['z']['bg'].background, origin='lower')

