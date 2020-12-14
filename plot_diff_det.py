import cv2
import numpy as n
import matplotlib.pyplot as plt
import glob
import re
import time

def animate_mp4s(mp4fname,max_im,out):
    fl2=glob.glob("%s-*.jpg"%(mp4fname))    
    fl2.sort()
    print(fl2)
    if len(fl2)>1:
        #        max_im=cv2.imread(fl2[0])
        #       max_im[:,:,:]=0.0
        
        idxs=[]
        for f2 in fl2:
            idx=int(re.search(".*/full_...mp4-(....).jpg",f2).group(1))
            idxs.append(idx)
            #        print(idxs)
        diffs=n.diff(idxs)
        pairs=[]
        for di,d in enumerate(diffs):
            if d == 1:
                pairs.append(di)
                pairs.append(di+1)
        gidx=n.unique(pairs)
        if len(gidx)>2:
            for gi in gidx:
                im=cv2.imread(fl2[gi])
                d_im0=cv2.resize(im,(1024,768))
                max_im[738:768,:,:]=0        
                max_im=n.maximum(max_im,d_im0)
                out.write(max_im)
                cv2.imshow("window",max_im)
                cv2.waitKey(100)
            if len(gidx)>0:
                cv2.waitKey(1000)
    return(max_im)



cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#fourcc = cv2.cv.CV_FOURCC(*'XVID')
codec = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('output.avi',codec, 5.0, (1024,768))

dl=glob.glob("/data0/cam*")
dl.sort()
print(dl)
for d in dl:
    max_im=n.zeros([768,1024,3],dtype=n.uint8)

    fl = glob.glob("%s/21/full*.mp4"%(d))
    fl.sort()
    for f in fl:
        print(f)
        max_im=animate_mp4s(f,max_im,out)
        
out.release()
