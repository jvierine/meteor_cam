import cv2
import numpy as n
import matplotlib.pyplot as plt
import glob
import sys
import imageio

from skimage.measure import block_reduce
import os
import re

#from skimage.measure import compare_ssim as ssim
from mpi4py import MPI
comm = MPI.COMM_WORLD    


def img_var_est(fname="full_59.mp4",
                cam_0=1490,
                cam_1=1536,
                dec=4,
                plot=True,
                median_len=30):
    
    cap = cv2.VideoCapture(fname)
    ret,frame0 = cap.read()
    frame0[cam_0:cam_1,:,:]=0
    prev=n.array(block_reduce(cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY),block_size=(dec,dec),func=n.max),dtype=n.float32)

    w=prev.shape[0]
    h=prev.shape[1]
    stdest=n.zeros([w,h],dtype=n.float32)
    H=n.zeros([median_len,w,h],dtype=n.float32)    
    n_avg=0.0
    idx=0
    while(1):
        ret,frame0 = cap.read()
        if not ret:
            break
        frame0[cam_0:cam_1,:,:]=0                    
        gray = n.array(block_reduce(cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY),block_size=(dec,dec),func=n.max),dtype=n.float32)
        prev=gray
        H[idx,:,:]=gray
        idx+=1
#        print(idx)
        if idx == median_len:
            stdest+=n.var(H,axis=0)+1.0
            n_avg+=1.0
            idx=0
    cap.release()
    stdest=n.sqrt(stdest/n_avg)
    if plot:
        plt.imshow(stdest,vmin=0,vmax=10.0)
        plt.colorbar()
        plt.show()
    return(stdest)

def diff_img_stack(fname="full_59.mp4",
                   odir="/scratch/data/juha/meteor",
                   cam_0=1490,
                   cam_1=1536,
                   max_skip=2,        # maximum frame number difference
                   max_vel=40,        # at least one pixel and at most 50 pixel per frame movement.
                   min_vel=0.1,       # this many pixels per frame
                   min_n_frames=4,
                   dec=4,
                   history_len=1000,
                   debug_print=False,
                   n_frames=15):
    
    os.system("mkdir -p %s"%(odir))
    stdest=img_var_est(fname=fname,
                       cam_0=cam_0,
                       cam_1=cam_1,
                       dec=dec,
                       plot=False,
                       median_len=30)

    det_frames = []
    det_pos = []
    det_snr = []    

    scale=None
    
    cap = cv2.VideoCapture(fname)
    ret,frame0 = cap.read()
    frame0[cam_0:cam_1,:,:]=0
    history=n.zeros(history_len)

    gray = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
    prev=n.array(block_reduce(gray,block_size=(dec,dec),func=n.max),dtype=n.float32)
    frame_num=1

    prev_det_frame=0
    prev_det_xi=-1
    prev_det_yi=-1

    det_idx=0

    while(1):
        ret,frame0 = cap.read()
        if ret:
            frame_orig=n.copy(frame0)
            frame0[cam_0:cam_1,:,:]=0
            gray = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
            gray2=n.array(block_reduce(gray,block_size=(dec,dec),func=n.max),dtype=n.float32)
            diff_im = gray2-prev

            scaled_diff=diff_im/stdest/n.sqrt(2.0)

            
            max_diff=n.max(scaled_diff)

            # difference image maximum at least 3 stdev
            if max_diff > 8.0:
                frame_r=block_reduce(n.array(frame_orig,dtype=n.float32),block_size=(2,2,1),func=n.mean)

                if scale == None:
                    fr2=n.copy(frame_r)
                    fs=n.sort(fr2.flatten())
                    max_img_val=fs[int(0.95*len(fs))]
                    scale=n.min([255.0/max_img_val,4.0])
                frame_r=frame_r*scale
                frame_r[frame_r > 255.0]=255.0
                frame_r=n.array(frame_r,dtype=n.uint8)

                
                p=re.search(".*/(cam.)/(........)/(..)/(full_...mp4)",fname)
                cam_str=p.group(1)
                yr_str=p.group(2)
                hr_str=p.group(3)
                fn_str=p.group(4)
                #%04d.jpg
                odirname="%s/%s/%s/%s"%(odir,cam_str,yr_str,hr_str)
                os.system("mkdir -p %s"%(odirname))
                ofname="%s/%s-%04d.jpg"%(odirname,fn_str,frame_num)
                print("saving %s"%(ofname))
                cv2.imwrite(ofname, frame_r)

                # what is the x and y index of this
                idx=n.argmax(diff_im)
                xi,yi=n.unravel_index(idx,diff_im.shape)

                # value of pixel
                max_val=scaled_diff[xi,yi]

                print("possible det %d %d,%d snr %1.2f scale %1.2f"%(frame_num,xi,yi,max_val,scale))
                det_frames.append(frame_num)
                det_pos.append((xi,yi))
                det_snr.append(max_val)
                det_idx+=1

            prev=gray2
            frame_num+=1

        else:
            break
    cap.release()
    return(det_frames,det_pos,det_snr)


if __name__ == "__main__":
#    fl=glob.glob("examples/*.mp4")    
    fl=glob.glob("/var/www/kaira/juha/meteor/cam*/20201213/*/*.mp4")
    fl.sort()

    for fi in range(comm.rank,len(fl),comm.size):
        f=fl[fi]
        print(f)
        diff_img_stack(fname=f,
                       odir="/scratch/data/juha/meteor")
#        det_frames,det_pos=diff_img_stack(fname=f)

        

