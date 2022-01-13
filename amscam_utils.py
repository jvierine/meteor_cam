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

import solve_video as sv


#def detection_azel(imfile="tests/2022_01_09_22_08_02_000_011331.mp4.png",
#                   fname="tests/2022_01_09_22_08_02_000_011331.mp4.corr",
#                   t0=Time(0,format="unix"),
#                   obs=EarthLocation(lon=19.22454409315662,height=77.3,lat=69.5861167101982),
#                   plot=False):

def get_obs_loc():
    return(EarthLocation(lon=19.22454409315662,height=77.3,lat=69.5861167101982))

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

def file_name_to_cam_id(fname):
    """
    Read date from file name. assume integrated over dt seconds starting at file print time
    """
    res=re.search(".*(....)_(..)_(..)_(..)_(..)_(..)_(...)_(......).mp4.*",fname)
    cam_id=res.group(8)    
    return(cam_id)



if __name__ == "__main__":
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    fl=glob.glob("tests/*.mp4")
    fl.sort()
    obs=get_obs_loc()
    
    for fi in range(rank,len(fl),size):
        f=fl[fi]
        print("rank %d file %s"%(rank,f))
        t0 = file_name_to_datetime(f)
        cam_id = file_name_to_cam_id(f)
        
        fname=sv.img_mean(fname=f,
                          t0=t0,
                          obs=obs,
                          plot=True)
        