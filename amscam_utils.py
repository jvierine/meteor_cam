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
import os

conf = {
    "cal_video_dir":"./tests/",
    "image_width":1920,
    "image_height":1080,

    "locations":{"AMS133": {"lon":19.22454409315662,"height":77.3,"lat":69.5861167101982},
                 "AMS132": {"lon":17.081673,"height":21.0,"lat":69.054463},
                 "AMS131": {"lon":18.938883907338976,"height":104.0,"lat":69.66154987956081},
                 }
}

#def detection_azel(imfile="tests/2022_01_09_22_08_02_000_011331.mp4.png",
#                   fname="tests/2022_01_09_22_08_02_000_011331.mp4.corr",
#                   t0=Time(0,format="unix"),
#                   obs=EarthLocation(lon=19.22454409315662,height=77.3,lat=69.5861167101982),
#                   plot=False):

def get_obs_loc(cam_id="011331"):
    obs_id = "AMS%s"%(cam_id[2:5])
    lat=conf["locations"][obs_id]["lat"]
    lon=conf["locations"][obs_id]["lon"]
    height=conf["locations"][obs_id]["height"]    
    return(EarthLocation(lon=lon,height=height,lat=lat))

def get_solved_videos(cam_id):
    fl=glob.glob("%s/*_%s.azel.h5"%(conf["cal_video_dir"],cam_id))
    fl.sort()
    return(fl)

def mask(I,cam_id):
    mask_fname="masks/mask_%s.png"%(cam_id)
    print(mask_fname)
    if os.path.exists(mask_fname):
        mask=n.array(imageio.imread(mask_fname),dtype=n.float32)/255.0
#        plt.imshow(I*mask[:,:,0])
 #       plt.colorbar()
  #      plt.show()
        return(I*(mask[:,:,0]))
    else:
        print("no mask found")
        return(I)

def get_cameras(station_id="AMS133"):
    """
    Look in calibration folder and determine what cameras exist based on file names
    """
    stat_num=station_id[3:6]
    cam_ids=[]
    for i in range(1,8):
        cam="01%s%d"%(stat_num,i)
        fl=glob.glob("%s/2*%s.mp4"%(conf["cal_video_dir"],cam))
        if len(fl)>0:
            print("found file for %s"%(cam))
            cam_ids.append(cam)

#    fl=glob.glob("%s/2*.mp4"%(conf["cal_video_dir"]))
 #   fl.sort()
  #  cam_ids=[]
   # for f in fl:
    #    cam_id=file_name_to_cam_id(f)
     #   if cam_id not in cam_ids:
      #      cam_ids.append(cam_id)
#    cam_ids.sort()
    return(cam_ids)
            
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

def file_name_to_obs(fname):
    cam_id=file_name_to_cam_id(fname)
    return(get_obs_loc(cam_id="011331"))

def file_name_to_cam_id(fname):
    """
    Read date from file name. assume integrated over dt seconds starting at file print time
    """
    res=re.search(".*(....)_(..)_(..)_(..)_(..)_(..)_(...)_(......).mp4.*",fname)
    cam_id=res.group(8)    
    return(cam_id)

if __name__ == "__main__":
    cam_ids=get_cameras()
    print(cam_ids)
