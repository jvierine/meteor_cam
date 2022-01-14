#!/usr/bin/env python3

import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob

import cv2
import re
import os

import amscam_utils as au
import fit_lens_model as lm
import imageio

def find_files(dirname="./fisheye_videos"):

    # 2022_01_09_21_11
    fl=glob.glob("%s/2022_01_09_21_11_*.mp4"%(dirname))
    
    fl.sort()
    for f in fl:
        print(f)
        cam_id=au.file_name_to_cam_id(f)
        t0 = au.file_name_to_datetime(f,dt=60.0)
        print(cam_id)
        print(t0)
    return(fl)



def find_pics():

    fl=glob.glob("fisheye_pics/2022_01_*/*011331*.jpg")
    fl.sort()

    files=[]
    
    for f in fl:

        g=re.search("(.*)/AMS133_(......)_(................).jpg",f)
        dirn=g.group(1)
        cam_id=g.group(2)
        date=g.group(3)
        trys = ["%s/AMS133_011331_%s.jpg"%(dirn,date),
                "%s/AMS133_011332_%s.jpg"%(dirn,date),
                "%s/AMS133_011333_%s.jpg"%(dirn,date),
                "%s/AMS133_011334_%s.jpg"%(dirn,date),
                "%s/AMS133_011335_%s.jpg"%(dirn,date),
                "%s/AMS133_011336_%s.jpg"%(dirn,date),
                "%s/AMS133_011337_%s.jpg"%(dirn,date),
                ]
        found_all=True
        for t in trys:
            if not os.path.exists(t):
                found_all=False
        if found_all:
            files.append(trys)

    return(files)
  


#print(files)
#exit(0)



def fisheye_pics(image_width=2*1080):

    files=find_pics()
    
    focal_l = 1.5*(image_width/2.0)/n.pi

    # video frame
    I = n.zeros([image_width,image_width,3],dtype=n.float32)
    N = n.zeros([image_width,image_width,3],dtype=n.float32)    

    xg=n.linspace(-1,1,num=au.conf["image_width"])
    y_max=au.conf["image_height"]/au.conf["image_width"]
    yg=n.linspace(-y_max,y_max,num=au.conf["image_height"])
    xgg,ygg=n.meshgrid(xg,yg)
    
    data = {}

    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (image_width,image_width))
    

    cam_ids=["011331","011332","011333","011334","011335","011336","011337"]
    for cam_id in cam_ids:
#        cam_id=au.file_name_to_cam_id(f)
        
        pc=lm.get_polycal(cam_id=cam_id)
        
        # positions per pixel
        p_n, p_e, p_u = pc.get_neu(xgg,ygg)

        # normalize
        norm=n.sqrt(p_n**2.0 + p_e**2.0 + p_u**2.0)
        p_n=p_n/norm
        p_e=p_e/norm
        p_u=p_u/norm        

        theta = n.arccos(p_u)
        az = n.arctan2(p_e, p_n)

#        R = focal_l*n.sin(theta)
        R = focal_l*theta
        xp = n.array(n.round(R*n.cos(az) + image_width/2.0),dtype=n.int)
        yp = n.array(n.round(R*n.sin(az) + image_width/2.0),dtype=n.int)

        #        plt.pcolormesh(xp)
        #       plt.colorbar()
        #      plt.show()
        
        #        print(yp.shape)
        #       print(xp.shape)        

        cam_data={"cal": pc, "xp":xp, "yp":yp}
        
        data[cam_id]=cam_data
        
        #        print(data[cam_id])
        
        
        ##    def get_neu(self,x,y):
        #      n_m=forward_polymodel3(x,y,self.npar)
        #     e_m=forward_polymodel3(x,y,self.epar)
        #    u_m=forward_polymodel3(x,y,self.upar)
        #   return(n_m,e_m,u_m)
        

    cams=data.keys()
    new_dim=(1920,1080)
    
    for idx in range(len(files)):
        N[:,:,:]=0.0
        I[:,:,:]=0.0
        fl = files[idx]
        for ci in range(len(cam_ids)):
            frame0 = imageio.imread(fl[ci])

            cam_id=cam_ids[ci]
            print(cam_id)
#            ret,frame0=data[cam_id]["cap"].read()
            frame = cv2.resize(frame0, new_dim)
            plt.imshow(frame)

#            print(frame.shape)
            xp=data[cam_id]["xp"]
            yp=data[cam_id]["yp"]            
            I[xp,yp,:]+=n.array(frame,dtype=n.float32)/255.0
            N[xp,yp,:]+=1.0#n.array(frame,dtype=n.float32)/255.0
        N[N < 1.0]=1.0
        I=I/N
        cv2.imshow("frame",I)
        out.write(n.array(I*255.0,dtype=n.uint8))

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    out.release()
#        plt.imshow(I)
 #       plt.show()
        
#        ret,frame0 = cap.read()
 #       if not ret:
  #          break

   #     frame = n.array(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY),dtype=n.float32)
    #    
     #   frame = cv2.blur(frame,(blur_width,blur_width))

        
    # we need to go from NEU coordinates to x,y




fisheye_pics()


def fisheye_video(file_list,image_width=1080):
    # Orthographic
    # r=f sin theta
    
    # full fisheye with pixel units on the image plane
#    focal_l = 1075/2.0
    focal_l = 1.8*(image_width/2.0)/n.pi

    # video frame
    I = n.zeros([image_width,image_width,3],dtype=n.float32)
    N = n.zeros([image_width,image_width,3],dtype=n.float32)    

    xg=n.linspace(-1,1,num=au.conf["image_width"])
    y_max=au.conf["image_height"]/au.conf["image_width"]
    yg=n.linspace(-y_max,y_max,num=au.conf["image_height"])
    xgg,ygg=n.meshgrid(xg,yg)
    
    data = {}
    file_list.sort()
    for f in file_list:
        cam_id=au.file_name_to_cam_id(f)
        cap=cv2.VideoCapture(f)
        
        pc=lm.get_polycal(cam_id=cam_id)
        
        # positions per pixel
        p_n, p_e, p_u = pc.get_neu(xgg,ygg)

        # normalize
        norm=n.sqrt(p_n**2.0 + p_e**2.0 + p_u**2.0)
        p_n=p_n/norm
        p_e=p_e/norm
        p_u=p_u/norm        

        theta = n.arccos(p_u)
        az = n.arctan2(p_e, p_n)

#        R = focal_l*n.sin(theta)
        R = focal_l*theta
        xp = n.array(n.round(R*n.cos(az) + image_width/2.0),dtype=n.int)
        yp = n.array(n.round(R*n.sin(az) + image_width/2.0),dtype=n.int)

        #        plt.pcolormesh(xp)
        #       plt.colorbar()
        #      plt.show()
        
        #        print(yp.shape)
        #       print(xp.shape)        

        cam_data={"cap":cap, "cal": pc, "xp":xp, "yp":yp}
        
        data[cam_id]=cam_data
        
        #        print(data[cam_id])
        
        
        ##    def get_neu(self,x,y):
        #      n_m=forward_polymodel3(x,y,self.npar)
        #     e_m=forward_polymodel3(x,y,self.epar)
        #    u_m=forward_polymodel3(x,y,self.upar)
        #   return(n_m,e_m,u_m)
        

    cams=data.keys()
    new_dim=(1920,1080)
    while(1):
        N[:,:,:]=0.0
        I[:,:,:]=0.0
        for cam_id in cams:
            print(cam_id)
            ret,frame0=data[cam_id]["cap"].read()
            frame = cv2.resize(frame0, new_dim)

            print(frame.shape)
            xp=data[cam_id]["xp"]
            yp=data[cam_id]["yp"]            
            I[xp,yp,:]+=n.array(frame,dtype=n.float32)/255.0
            N[xp,yp,:]+=1.0#n.array(frame,dtype=n.float32)/255.0
        N[N < 1.0]=1.0
        I=I/N
        cv2.imshow("frame",I)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
#        plt.imshow(I)
 #       plt.show()
        
#        ret,frame0 = cap.read()
 #       if not ret:
  #          break

   #     frame = n.array(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY),dtype=n.float32)
    #    
     #   frame = cv2.blur(frame,(blur_width,blur_width))

        
    # we need to go from NEU coordinates to x,y


    I[xp,yp,:]=im


    





fl=find_files()
fisheye_video(fl)        
