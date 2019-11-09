#%%
import mod_read
import mod_analysis
from mod_read import *
from mod_analysis import *
from contextlib import contextmanager
import importlib
import os

path_table = '/media/mouse13/My Passport/corotation/manga/dr14_zpt02_zpt06_lgmgt9_MANGA_barflag.cat'
im_path = '/media/mouse13/My Passport/corotation/manga/input/'
out_path = '/media/mouse13/My Passport/corotation/manga/pics/'

names = [elem.split('.')[0] for elem in os.listdir(im_path)]
#
# all_table = pd.read_table(path_table, sep=' ')
# all_table.loc[all_table.objID == int(names[0]), ['ra', 'dec']].values[0][0]

#%%
images = make_images(names=names[10:], bands='all', types='all', path=im_path, path_table=path_table, manga=True)

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


#%%
for i in range(len(images)):
    calc_slit(images[i]['r'], 40, convolve=True)
    images[i]['r'].plot_slits(n_slit=40)

#%%
with figure() as fig:
    plt.imshow(images[0]['r']['real'], origin='lower', cmap='Greys', norm=ImageNormalize(stretch=LogStretch()))

#%%
print(images[0]['r']['real'])