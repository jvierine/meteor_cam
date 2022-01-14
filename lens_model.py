#!/usr/bin/env python3

import numpy as n
import matplotlib.pyplot as plt
import imageio as iio

def pixel_units_to_normalized(x,y,w=1920,h=1080):
    """
    go from pixel coordinates in the range 0..(w-1) on the horizontal axis to
    normalized pixel coordinates in range [-1,1] on the horizontal axis 
    horizontal width must be greater than vertical width!
    """
    mean_x=w/2.0
    mean_y=h/2.0    
    xn=(x-mean_x)/mean_x
    yn=(y-mean_y)/mean_x
    return(xn,yn)

def normalized_to_pixel(x,y,w=1920,h=1080):
    """
    go from  normalized pixel coordinates in range [-1,1] on the horizontal axis 
    to pixel coordinates in the range 0..(w-1) on the horizontal axis
    horizontal width must be greater than vertical width!
    """
    
    mean_x=w/2.0
    mean_y=h/2.0

    xp = x*mean_x + mean_x
    yp = y*mean_x + mean_y
    return(xp,yp)

def gnomic_to_stereographic(x,y,f_g=1.1,f_s=1.6):
    """
    these use normalized pixel coordinates in range [-1,1] on the horizontal axis
    horizontal width must be greater than vertical width!
    """
    
    R_g = n.sqrt(x**2.0 + y**2.0)
    theta = n.arctan(R_g/f_g)
    R_s = f_s*n.sin(theta)
    alpha=n.arctan2(y,x)
    return(R_s*n.cos(alpha),R_s*n.sin(alpha))

def stereographic_to_gnomic(x,y,f_g=1.1,f_s=1.6):
    """
    these use normalized pixel coordinates in range [-1,1] on the horizontal axis
    horizontal width must be greater than vertical width!
    """
    R_s = n.sqrt(x**2.0 + y**2.0)
    theta = n.arcsin(R_s/f_s)
    R_g = f_g*n.tan(theta)
    alpha=n.arctan2(y,x)
    return(R_g*n.cos(alpha),R_g*n.sin(alpha))
    

def img_spherical_to_gnomic(I, out_fname="tmp.png",f_g=1.1, f_s=1.6,plot=False):
    """
    focal length is in units of pixels, with the maximum x (horizontal) pixel value being the scale.
    the width of the image has to be larger than the height!
    horizontal width must be greater than vertical width!
    """
    # width = shape[1]
    # height = shape[0]
    dim_in=(I.shape[1],I.shape[0])
    dim_out=dim_in

    
    # pixel units 
    inx = n.arange(dim_in[0])
    inx = inx - n.mean(inx)
    iny = n.arange(dim_in[1])
    iny = iny - n.mean(iny)

    max_x = n.max(inx)
    max_y = n.max(iny)    
    min_x = n.min(inx)
    min_y = n.min(iny)    
    inxm,inym=n.meshgrid(inx/max_x,iny/max_x)
    outxm,outym=n.meshgrid(inx/max_x,iny/max_x)    

    # gnomic radius
    R_g = n.sqrt(outxm**2.0 + outym**2.0)
    theta=n.arctan(R_g/f_g)
    R_s = f_s*n.sin(theta)
    alpha=n.arctan2(outym,outxm)

    x_s = R_s*n.cos(alpha)
    y_s = R_s*n.sin(alpha)
    
    xidx=n.array(n.round(x_s*max_x - min_x),dtype=n.int64)
    yidx=n.array(n.round(y_s*max_x - min_y ),dtype=n.int64)

    bidx=n.where( (xidx < 0) | (xidx> (dim_in[0]-1)) | (yidx < 0) | (yidx > (dim_in[1]-1)) )
    yidx[bidx]=0
    xidx[bidx]=0

#    print(I.shape)
 #   print(n.max(yidx))
  #  print(n.max(xidx))    


    I_gnomic=I[yidx,xidx]
    I_gnomic[bidx]=0.0
    if plot:
        plt.imshow(I_gnomic)
        plt.colorbar()
        plt.show()

    iio.imwrite(out_fname,I_gnomic)
    return(I_gnomic)
    


if __name__ == "__main__":


    xn,yn=pixel_units_to_normalized(10,10,w=1920,h=1080)
    xp,yp=normalized_to_pixel(xn,yn,w=1920,h=1080)

    
    xg,yg=stereographic_to_gnomic(n.linspace(0,1,num=10),n.linspace(0,1,num=10))
    xs,ys=gnomic_to_stereographic(xg,yg)

    I=iio.imread("tests/2022_01_09_22_08_02_000_011331.mp4.orig.png")
    img_spherical_to_gnomic(I,out_fname="tests/2022_01_09_22_08_02_000_011331.mp4.gnomic.png",plot=True)

    


