#!/usr/bin/env python3

import amscam_utils as au
import numpy as n
from mpi4py import MPI
import glob
import solve_video as sv

import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == "__main__":


    
    if len(sys.argv) == 2:
        fl=glob.glob(sys.argv[1])
    else:
        fl=glob.glob("%s/*.mp4"%(au.conf["cal_video_dir"]))
    fl.sort()
    
    
    for fi in range(rank,len(fl),size):
        f=fl[fi]
        print("rank %d file %s"%(rank,f))
        t0 = au.file_name_to_datetime(f)
        cam_id = au.file_name_to_cam_id(f)
        obs=au.get_obs_loc(cam_id)
        
        fname=sv.img_mean(fname=f,
                          t0=t0,
                          obs=obs,
                          plot=True)
        
