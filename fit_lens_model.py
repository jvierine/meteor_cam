#!/usr/bin/env python3

import amscam_utils as au
import lens_model as lm

import numpy as n
import matplotlib.pyplot as plt

import h5py
import glob
import imageio
import bright_stars as bsc

# Import libraries
from mpl_toolkits import mplot3d

from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.time import TimeDelta
from astropy.coordinates import SkyCoord
import os

import cv2


import polycal as pcal



def plot_fisheye_bounds(image_width=1080):

    image = n.zeros([1080,1920,3],dtype=n.float32)

    colors = [ [1.0,0,0],[0,1,0], [0,0,1] ] 
    
    focal_l = 1.5*(image_width/2.0)/n.pi

    xg=n.arange(au.conf["image_width"]) - au.conf["image_width"]/2.0 + 0.5
    yg=n.arange(au.conf["image_height"]) - au.conf["image_height"]/2.0 + 0.5
    xgg,ygg=n.meshgrid(xg,yg)
    
    data = {}

#    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (image_width,image_width))

    cam_ids=["011331","011332","011333","011334","011335","011336","011337"]

    n_cams = len(cam_ids)

    I = n.zeros([image_width,image_width,3],dtype=n.float32)

    for cam_id in cam_ids:
        pc=pcal.get_polycal(cam_id=cam_id)
        
        # positions per pixel
        p_n, p_e, p_u = pc.get_neu(xgg,ygg)

        # normalize
        norm=n.sqrt(p_n**2.0 + p_e**2.0 + p_u**2.0)
        p_n=p_n/norm
        p_e=p_e/norm
        p_u=p_u/norm        

        theta = n.arccos(p_u)
        az = n.arctan2(p_e, p_n)

        R = focal_l*theta
        xp = n.array(n.round(R*n.cos(az) + image_width/2.0),dtype=n.int)
        yp = n.array(n.round(R*n.sin(az) + image_width/2.0),dtype=n.int)

        cam_data={"cal": pc, "xp":xp, "yp":yp}
        
        data[cam_id]=cam_data
        
    cams=data.keys()
    new_dim=(1920,1080)

    for ci in range(len(cam_ids)):
        cam_id=cam_ids[ci]
        print(cam_id)
        xp=data[cam_id]["xp"]
        yp=data[cam_id]["yp"]

        for col in range(3):
#            image[0:10,:,col]=colors[ci%len(colors)][col]
 #           image[:,0:10,col]=colors[ci%len(colors)][col]
  #          image[(1080-10):1080,:,col]=colors[ci%len(colors)][col]
   #         image[:,(1920-10):1920,col]=colors[ci%len(colors)][col]
            image[:,:,col]=colors[ci%len(colors)][col]
#            image[:,0:10,col]=colors[ci%len(colors)][col]
 #           image[(1080-10):1080,:,col]=colors[ci%len(colors)][col]
  #          image[:,(1920-10):1920,col]=colors[ci%len(colors)][col]

        I[xp,yp,:] += image
        
    plt.imshow(I)
    plt.show()



def plot_sky_sphere(station_id="AMS133"):
    """
    show the field of views of the cameras as an approxmate grid
    """

    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    cam_ids=au.get_cameras(station_id=station_id)

    xg=n.linspace(-au.conf["image_width"]/2.0,au.conf["image_width"]/2.0,
                  num=20)
    yg=n.linspace(-au.conf["image_height"]/2.0,au.conf["image_height"]/2.0,
                  num=20)
    xgg,ygg=n.meshgrid(xg,yg)
    
    for cam in cam_ids:
        try:
            pc=pcal.get_polycal(cam)
            print("on-axis az %1.3f el %1.3f"%(pc.on_axis_az, pc.on_axis_el))
            print(pc.azel_to_xy(pc.on_axis_az,pc.on_axis_el))
        
            model_n, model_e, model_u = pc.get_neu(xgg,ygg)

            m_norm=n.sqrt(model_n**2.0 + model_e**2.0 + model_u**2.0)
            model_n = model_n/m_norm
            model_e = model_e/m_norm
            model_u = model_u/m_norm        

            ax.scatter3D(model_n.flatten(),model_e.flatten(),model_u.flatten(),label=cam)
        except:
            print("not enough calibration points found yet")
            pass
    ax.set_xlabel("North")
    ax.set_ylabel("East")
    ax.set_zlabel("Up")        
  #  # show plot
    plt.legend(title="camera id")
    
    plt.show()
    plt.close()
        



    

    


def find_yale_matches(station_id="AMS133"):
    import star_finder as sfm
    
    cam_ids=au.get_cameras(station_id=station_id)
    
    for cam in cam_ids:

        pcal_found=False
        try:
            pc=pcal.get_polycal(cam_id=cam,model_order=2,astrometry=False)
            pcal_found=True
        except:
            pass
            

        if pcal_found:
            yale_x = []
            yale_y = []
            yale_az = []
            yale_el = []

            x_resid = []
            y_resid = []
            
#            pc=pcal.polycal(fname=cal_fname,model_order=1)
            
            sf=sfm.star_finder(pc=pc)            

            fl = glob.glob("tests/*%s.mp4.orig.png"%(cam))
            for f in fl:
                I_orig=imageio.imread(f)
                I=au.mask(I_orig,cam)
                t0 = au.file_name_to_datetime(f)
                obs = au.get_obs_loc(cam)
                
                x,y,az,el=sf.find_bright_stars_in_image(t0,obs,N_stars=500)
                
                n_stars=len(x)
                
                found_xs=[]
                found_ys=[]
                true_xs=[]
                true_ys=[]
                
                for si in range(n_stars):
                    x0=x[si]
                    y0=y[si]
                    print(x0)
                    print(y0)
                    minx=n.max([0,x0-40])
                    miny=n.max([0,y0-40])
                    maxx=n.min([x0+40,pc.image_width])
                    maxy=n.min([y0+40,pc.image_height])

#                    plt.imshow(I[miny:maxy,minx:maxx])  
                    sources=sfm.detect_stars(I[miny:maxy,minx:maxx])
                    if sources != None:
                        xc=sources["xcentroid"]
                        yc=sources["ycentroid"]

                        truex=x0-minx
                        truey=y0-miny

                        di=n.argmin( (truex-xc)**2.0 + (truey-yc)**2.0 )

                        if n.abs(truex - xc[di]) < 5.0 and n.abs(truey-yc[di]) < 5.0:
                            resid = n.sqrt( (truex-xc[di])**2.0 + (truey-yc[di])**2.0)
                            
                            x_resid.append(truex-xc[di])
                            y_resid.append(truey-yc[di])                            
                            
                            print("found nearby star resid %1.2f"%(resid))
                            found_xs.append(x[si])
                            found_ys.append(y[si])
                            true_xs.append(xc[di]+minx)
                            true_ys.append(yc[di]+miny)
                            
                            yale_x.append(xc[di]+minx)
                            yale_y.append(yc[di]+miny)
                            yale_az.append(az[si])
                            yale_el.append(el[si]) 
                            #                               plt.scatter(xc[di],yc[di],s=100,facecolors='none',edgecolors='red')
                            
                            #                  plt.axhline(y0-miny,color="red")
                            #                 plt.axvline(x0-minx,color="red")                    
                            #                    plt.scatter([x0-minx],[y0-miny],s=100,facecolors='none',edgecolors='red')                        
                            #                plt.colorbar()
                            #               plt.show()
                            

                plt.figure(figsize=(1.5*8,6*1.5))
                plt.imshow(I_orig,vmax=128)
                plt.title(t0)

                # plt.scatter(found_xs,found_ys,s=100,facecolors='none',edgecolors='white')
                # all stars searched
                plt.scatter(x,y,s=100,facecolors='none',edgecolors='red')
                # found positions
                plt.scatter(true_xs,true_ys,s=100,facecolors='none',edgecolors='yellow')                
                plt.tight_layout()
                print("saving yale matches %s"%(f))
#                plt.show()
                plt.savefig("%s.yale.png"%(f))
#                plt.clf()
 #               plt.close()
#                plt.show()
            # we now have
            # more stars that are found using the course cal. let's save them
            out_cal_fname="calibrations/yale_%s.h5"%(cam)            
            print("saving yale cal %s resid %1.2f %1.2f"%(out_cal_fname,n.median(n.abs(x_resid)),n.median(n.abs(y_resid))))
            
            ho=h5py.File(out_cal_fname,"w")
            ho["x"]=yale_x
            ho["y"]=yale_y
            ho["az"]=yale_az
            ho["el"]=yale_el            
            ho["image_width"]=pc.image_width
            ho["image_height"]=pc.image_height
            ho["cam_id"]=pc.cam_id
            ho["x_resid"]=x_resid
            ho["y_resid"]=y_resid            
            ho.close()



if __name__ == "__main__":
    plot_fisheye_bounds(image_width=1080)
#    plot_sky_sphere(station_id="AMS133")
 #   find_yale_matches()
            
