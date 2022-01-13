#!/usr/bin/env python3

import numpy as n
import matplotlib.pyplot as plt
import astropy.io.fits as fio
import imageio
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
import glob
import re
from astropy.time import Time
from astropy.time import TimeDelta

from astropy.coordinates import SkyCoord


def file_name_to_datetime(fname,dt=60.0):
    """
    Read date from file name. assume integrated over dt seconds starting at file print time
    """
    res=re.search(".*(....)_(..)_(..)_(..)_(..)_(..)_(...)_(......).mp4.*",fname)
    # '2010-01-01T00:00:00'
    year=res.group(1)
    month=res.group(2)
    day=res.group(3)
    hour=res.group(4)
    mins=res.group(5)
    sec=res.group(6)    

    dt = TimeDelta(dt/2.0,format="sec")
    t0 = Time("%s-%s-%sT%s:%s:%s"%(year,month,day,hour,mins,sec), format='isot', scale='utc') + dt
    return(t0)

def detection_azel(imfile="tests/2022_01_09_22_08_02_000_011331.mp4.png",
                   fname="tests/2022_01_09_22_08_02_000_011331.mp4.corr",
                   plot=False):
    
    obs=EarthLocation(lon=19.22454409315662,height=77.3,lat=69.5861167101982)
    t0=file_name_to_datetime(imfile)
    aa_frame = AltAz(obstime=t0, location=obs)


    a=fio.open(fname)
    corr=a[1].data

    
    if plot:
        I=imageio.imread(imfile)
        plt.title(imfile)
        plt.imshow(I)

    azs=[]
    els=[]
    xs=[]
    ys=[]    
        
    for l in corr:
        print("RA %1.2f DEC %1.2f"%(l["index_ra"],l["index_dec"]))

        if l["match_weight"] > 0.0:
            if plot:
                plt.scatter(l[0]-1.0,l[1]-1.0,s=100,facecolors='none',edgecolors='white')
            
            c = SkyCoord(ra=l["index_ra"]*u.degree, dec=l["index_dec"]*u.degree, frame='icrs')
            altaz=c.transform_to(aa_frame)
            taz=altaz.az/u.deg
            tel=altaz.alt/u.deg
            azs.append(taz)
            els.append(tel)
            xs.append(l[0]-1.0)
            ys.append(l[1]-1.0)
        
            ras=Angle(l["index_ra"],u.deg).to_string(unit=u.hour)
            decs=Angle(l["index_dec"],u.deg).to_string(unit=u.deg,sep=('deg', 'm', 's'))

            if plot:
                plt.text(l[0]+10.0,l[1]+2.0,"%1.2f,%1.2f"%(taz,tel),color="white",alpha=0.3)
    if plot:
        plt.tight_layout()
        plt.show()
    

    return(n.array(xs),n.array(ys),n.array(azs),n.array(els))


    


    
if __name__ == "__main__":
    fl=glob.glob("tests/*.corr")
    fl.sort()
    for f in fl:
        prefix=re.search("(.*).corr",f).group(1)
        imfile="%s.png"%(prefix)
        print(imfile)
        print(f)
        plot_xyl(imfile,f)
