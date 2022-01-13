#!/usr/bin/env python3
'''
 Go through all HD video files in folder and do a plate solve
 splitting up the plate into smaller sections.
'''
import cv2
import numpy as n
import matplotlib.pyplot as plt
import glob
import sys
import imageio

import os
import re
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
import h5py

import astrometry_utils as ah
import lens_model as lm

def img_mean(fname="full_59.mp4",
             t0=Time(0,format="unix"),
             obs=EarthLocation(lon=19.22454409315662,height=77.3,lat=69.5861167101982),
             scale=1.0,
             solve=False,
             n_blocks_x=3,
             n_blocks_y=4,
             blur_width=5,
             plot=False):

    prefix=re.search("(.*).(...)",fname).group(1)
        
    cap = cv2.VideoCapture(fname)
    ret,frame0 = cap.read()
    avg=n.array(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY),dtype=n.float32)
    w=avg.shape[0]
    h=avg.shape[1]

    need_to_resize=False
    new_dim=(1920,1080)
    if h != 1920:
        print("resizing SD to HD size")
        need_to_resize=True
        
        avg = cv2.resize(avg, new_dim)        
        w=1080
        h=1920
    
    n_avg=1.0
    idx=0
    while(1):
        ret,frame0 = cap.read()
        if not ret:
            break

        frame = n.array(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY),dtype=n.float32)
        frame = cv2.resize(frame, new_dim)
        frame = cv2.blur(frame,(blur_width,blur_width))
#        print(n_avg)
#        avg=n.maximum(frame,avg)
        avg+=frame
        n_avg+=1.0
    cap.release()
    avg=avg/n_avg
    avg=avg-n.min(avg)
    avg=255.0*avg/n.max(avg)

    avg[avg>255.0]=255.0
    avg[avg<0]=0

     #   apply pre-distortion to make it easier to find stars    
     #   avg=lm.img_spherical_to_gnomic(n.array(avg,dtype=n.uint8), out_fname="%s.orig.png"%(fname),f_g=1.1, f_s=1.6,plot=False)

    imageio.imwrite("%s.orig.png"%(fname), n.array(avg,dtype=n.uint8))

    dx=int(n.floor(avg.shape[0]/n_blocks_x))
    xstep=int(dx/2)
    dy=int(n.floor(avg.shape[1]/n_blocks_y))
    ystep=int(dy/2)

    all_xs=n.array([])
    all_ys=n.array([])
    all_azs=n.array([])
    all_els=n.array([])
    all_wgts=n.array([])
    all_flxs=n.array([])            
    
    for i in range(2*n_blocks_x-1):
        for j in range(2*n_blocks_y-1):
            tox=n.min([avg.shape[0],i*xstep+dx])
            toy=n.min([avg.shape[1],j*ystep+dy])                
            BI=n.copy(avg[ (i*xstep):tox, (j*ystep):toy ])
            
            block_fname="%s.%d.%d.png"%(fname,i,j)
            print(block_fname)
            imageio.imwrite(block_fname, n.array(BI,dtype=n.uint8))
            det_file=ah.solve_field(block_fname)
            
            if det_file != None:
                xs,ys,azs,els,wgts,flxs=ah.detection_azel(block_fname,det_file,t0,obs,plot=False)
                all_xs=n.concatenate((all_xs,xs+j*ystep))
                all_ys=n.concatenate((all_ys,ys+i*xstep))
                all_azs=n.concatenate((all_azs,azs))
                all_els=n.concatenate((all_els,els))
                all_wgts=n.concatenate((all_wgts,wgts))
                all_flxs=n.concatenate((all_flxs,flxs))                                

    out_fname="%s.azel.h5"%(prefix)                
    ho=h5py.File("%s.azel.h5"%(prefix),"w")
    ho["weigth"]=all_wgts
    ho["flux"]=all_flxs    
    ho["x_pix"]=all_xs
    ho["y_pix"]=all_ys
    ho["az_deg"]=all_azs
    ho["el_deg"]=all_els
    ho["t0_unix"]=t0.unix
    ho["lat_deg"]=float(obs.lat/u.deg)
    ho["lon_deg"]=float(obs.lon/u.deg)
    ho["height_m"]=float(obs.height/u.m)
    ho.close()
    if plot:
        plt.scatter(all_xs,all_ys,s=100,facecolors='none',edgecolors='white')
        plt.title(fname)
        plt.imshow(avg,vmax=64)
        plt.tight_layout()
        plt.savefig("%s.solved.png"%(prefix))
        plt.clf()
        plt.close()
#        
 #       plt.show()
        
        plt.subplot(121)
        plt.scatter(all_xs,all_ys,c=all_els,s=20,vmin=0,vmax=90)
        plt.xlim([0,1920])
        plt.ylim([0,1080])        
        plt.colorbar()
        plt.subplot(122)       
        plt.scatter(all_xs,all_ys,c=all_azs,s=20,vmin=0,vmax=360)
        plt.xlim([0,1920])
        plt.ylim([0,1080])                
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("%s.azel.png"%(prefix))
        plt.clf()
        plt.close()
#        plt.show()
    

    return(out_fname)


if __name__ == "__main__":
    # SD video
    fname=img_mean(fname="tests/2022_01_10_00_04_00_000_011331.mp4",
                   t0=Time("2022-01-10T00:04:00",format="isot"),
                   obs=EarthLocation(lon=19.22454409315662,height=77.3,lat=69.5861167101982),
                   plot=True)
    
    
    
    
    fname=img_mean(fname="tests/2022_01_09_22_08_02_000_011331.mp4",
                   t0=Time("2022-01-09T22:08:02",format="isot"),
                   obs=EarthLocation(lon=19.22454409315662,height=77.3,lat=69.5861167101982),
                   plot=True)



    
#    img_mean(fname="tests/2022_01_10_02_15_00_000_011335.mp4")    
 #   img_mean(fname="tests/2022_01_10_02_34_00_000_011331.mp4")
#    img_mean(fname="tests/2022_01_10_01_08_00_000_011332.mp4")    
 #   img_mean(fname="tests/2022_01_10_01_08_00_000_011333.mp4")
  #  img_mean(fname="tests/2022_01_10_15_00_01_000_011334.mp4")
 #   img_mean(fname="tests/2022_01_10_15_21_00_000_011336.mp4")
  #  img_mean(fname="tests/2022_01_10_15_20_01_000_011337.mp4")
