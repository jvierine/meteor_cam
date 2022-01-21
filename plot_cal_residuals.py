#!/usr/bin/env python3

import numpy as n
import h5py
import matplotlib.pyplot as plt

import glob


fl=glob.glob("calibrations/yale*.h5")
fl.sort()
print(fl)

for f in fl:
    h=h5py.File(f,"r")
    print(h.keys())
    xr=h["x_resid"][()]
    yr=h["y_resid"][()]
    xstd=n.median(n.abs(xr))
    ystd=n.median(n.abs(yr))

    plt.subplot(211)
    plt.hist(xr,bins=20)
    plt.axvline(xstd)
    plt.xlabel("x residual (pixels)")
    plt.title("std = %1.2f"%(xstd))
    plt.subplot(212)
    plt.hist(yr,bins=20)
    plt.title("std = %1.2f"%(ystd))
    plt.axvline(ystd)
    plt.xlabel("y residual (pixels)")
    plt.show()
    
        
