import cv2
import numpy as n
import matplotlib.pyplot as plt
import glob

def diff_img_stack(fname="full_44.mp4",use_diff=False,diff_img=None):
    cap = cv2.VideoCapture('full_44.mp4')
    ret,frame0 = cap.read()

    if not use_diff:
        diff_img=n.zeros([frame0.shape[0],frame0.shape[1]],dtype=n.float32)
        
    frame_bw=n.zeros([frame0.shape[0],frame0.shape[1]],dtype=n.float32)
    
    prev=None
    first=True
    frame_num=0
    while(1):
        ret,frame0 = cap.read()
        frame0=n.array(frame0,dtype=n.float32)
        if not ret:
            break
        frame_num+=1
        print("frame %d"%(frame_num))
        
        frame_bw=frame0[:,:,0]+frame0[:,:,1]+frame0[:,:,2]
        if not first:
            diff_img = n.maximum( frame_bw-prev,diff_img)
        else:
            first=False

        prev=frame_bw
    video_capture.release()
        
    return(diff_img)

if __name__ == "__main__":
    fl=glob.glob("%s/*.mp4"%(sys.argv[1]))
    
    diff_img=diff_img_stack(fname=fl[0],use_diff=False)
    for f in fl[1:len(fl)]:
        diff_img=diff_img_stack(fname=f,use_diff=True,diff_img=diff_img)
        plt.imshow(diff_img,cmap="gray")
        plt.colorbar()
        plt.show()

    

