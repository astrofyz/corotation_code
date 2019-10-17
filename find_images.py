from astroquery.sdss import SDSS
from astropy import coordinates as coords
import numpy as np
import astropy.units as u
import pandas as pd

fn = open('/media/mouse13/My Passport/corotation/buta_gal/simbad.dat')
df_init = pd.read_csv(fn, sep=';', usecols=[3, 4], names=['name', 'coord1'])
fn.close()

#%%
for i in range(len(df_init)):
    pos = coords.SkyCoord(df_init.iloc[i]['coord1'], frame='icrs', unit=u.deg)
    # print(pos)
    xid = SDSS.query_region(coordinates=pos, radius=0.5*u.arcmin)
    try:
        print(xid[0])
    except:
        print('object not found')

# FIXME: look for running bash scripts https://stackoverflow.com/questions/13745648/running-bash-script-from-within-python