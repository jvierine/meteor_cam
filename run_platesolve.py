#!/usr/bin/env python3

import amscam_utils as au
import numpy as n
from mpi4py import MPI
import glob
import solve_video as sv

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == "__main__":

    fl=glob.glob("tests/*.mp4")
    fl.sort()
    obs=au.get_obs_loc()
    
    for fi in range(rank,len(fl),size):
        f=fl[fi]
        print("rank %d file %s"%(rank,f))
        t0 = au.file_name_to_datetime(f)
        cam_id = au.file_name_to_cam_id(f)
        
        fname=sv.img_mean(fname=f,
                          t0=t0,
                          obs=obs,
                          plot=True)
        
