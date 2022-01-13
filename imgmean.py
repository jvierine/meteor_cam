import cv2
import numpy as n
import matplotlib.pyplot as plt
import glob
import sys
import imageio

import os
import re

import astrometry_help as ah

def img_mean(fname="full_59.mp4",
             scale=1.0,
             solve=False,
             n_blocks_x=3,
             n_blocks_y=4,
             blur_width=5,
             plot=True):
    print(fname)

    
    cap = cv2.VideoCapture(fname)
    ret,frame0 = cap.read()
    avg=n.array(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY),dtype=n.float32)
    w=avg.shape[0]
    h=avg.shape[1]
    n_avg=1.0
    idx=0
    while(1):
        ret,frame0 = cap.read()
        if not ret:
            break

        frame = n.array(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY),dtype=n.float32)
        frame = cv2.blur(frame,(blur_width,blur_width))
#        print(n_avg)
#        avg=n.maximum(frame,avg)
        avg+=frame
        n_avg+=1.0
    cap.release()
    avg=avg/n_avg
    avg=avg-n.min(avg)
    avg=255.0*avg/n.max(avg)
    if plot:
        avg[avg>255.0]=255.0
        avg[avg<0]=0

        imageio.imwrite("%s.orig.png"%(fname), n.array(avg,dtype=n.uint8))

        dx=int(n.floor(avg.shape[0]/n_blocks_x))
        xstep=int(dx/2)
        dy=int(n.floor(avg.shape[1]/n_blocks_y))
        ystep=int(dy/2)

        all_xs=n.array([])
        all_ys=n.array([])
        all_azs=n.array([])
        all_els=n.array([])        
        
        for i in range(2*n_blocks_x-1):
            for j in range(2*n_blocks_y-1):
                tox=n.min([avg.shape[0],i*xstep+dx])
                toy=n.min([avg.shape[1],j*ystep+dy])                
                BI=n.copy(avg[ (i*xstep):tox, (j*ystep):toy ])

                block_fname="%s.%d.%d.png"%(fname,i,j)
                print(block_fname)
                imageio.imwrite(block_fname, n.array(BI,dtype=n.uint8))
                det_file=solve_field(block_fname)
                if det_file != None:
                    xs,ys,azs,els=ah.detection_azel(block_fname,det_file,plot=False)
                    all_xs=n.concatenate((all_xs,xs+j*ystep))
                    all_ys=n.concatenate((all_ys,ys+i*xstep))
                    all_azs=n.concatenate((all_azs,azs))
                    all_els=n.concatenate((all_els,els))
                    
        plt.scatter(all_xs,all_ys,s=100,facecolors='none',edgecolors='white')
        plt.title(fname)
        plt.imshow(avg,vmax=64)
        plt.show()

        plt.subplot(121)
        plt.scatter(all_xs,all_ys,c=all_els,s=20)
        plt.colorbar()
        plt.subplot(122)       
        plt.scatter(all_xs,all_ys,c=all_azs,s=20)
        plt.colorbar()
        plt.show()



def solve_field(fname):
    cmd="solve-field  %s --overwrite -d 1-40 --scale-low 3 --scale-high 40  --verbose  --crpix-center --plot-scale 4.0 -S %s"%(fname,"%s.solved"%(fname))
    os.system(cmd)
    if os.path.exists("%s.solved"%(fname)):
        print("Success!")
        prefix=re.search("(.*).png",fname).group(1)
        detection_file="%s.corr"%(prefix)
        return(detection_file)
    else:
        return(None)


if __name__ == "__main__":
    img_mean(fname="tests/2022_01_09_22_08_02_000_011331.mp4",solve=True)    
    img_mean(fname="tests/2022_01_10_02_15_00_000_011335.mp4",solve=True)    

    img_mean(fname="tests/2022_01_10_02_34_00_000_011331.mp4",solve=True)
    
    img_mean(fname="tests/2022_01_10_01_08_00_000_011332.mp4",solve=True)    
    img_mean(fname="tests/2022_01_10_01_08_00_000_011333.mp4",solve=True)
    img_mean(fname="tests/2022_01_10_15_00_01_000_011334.mp4",solve=True)
    img_mean(fname="tests/2022_01_10_15_21_00_000_011336.mp4",solve=True)

    img_mean(fname="tests/2022_01_10_15_20_01_000_011337.mp4",solve=True)
