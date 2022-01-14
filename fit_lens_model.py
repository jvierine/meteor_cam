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


def lstsq_it(A,m):
    """
    iterative least-squares fit to remove outliers
    """
    for i in range(3):
        xhat=n.linalg.lstsq(A,m,rcond=None)[0]
        resid=n.abs(n.dot(A,xhat)-m)
        std_est=n.median(resid)
        gidx=n.where(resid < 4.0*std_est)[0]

        A = n.copy(A[gidx,:])
        m = n.copy(m[gidx])

    xhat=n.linalg.lstsq(A,m,rcond=None)[0]
    return(xhat,A,m)

def forward_polymodel(x,y,par):
    """
    Simple second order polynomial 
    """
    model = par[0] + par[1]*x + par[2]*y + par[3]*x**2.0 + par[4]*y**2.0 + par[5]*x*y
    return(model)


def first_guess(x,y,az,el,plot_resid=False):
    """
    Guess optical axis and orientation 
    """
    neu_n=n.cos(n.pi*el/180)*n.cos(n.pi*az/180)
    neu_e=n.cos(n.pi*el/180)*n.sin(n.pi*az/180)
    neu_u=n.sin(n.pi*el/180)
    
    #
    # no = a0 + a1*x + a2*x**2.0 + a3*y**2.0 + a4*x*y
    # ea = a0 + a1*x + a2*x**2.0 + a3*y**2.0 + a4*x*y
    # up = a0 + a1*x + a3*y + a2*x**2.0  + a4*y**2.0 + a5*x*y
    n_m = len(x)
    A = n.zeros([n_m,6])
    A[:,0]=1.0
    A[:,1]=x
    A[:,2]=y    
    A[:,3]=x**2.0
    A[:,4]=y**2.0
    A[:,5]=x*y
    xhat_n,A_n,m_n=lstsq_it(A,neu_n)
    xhat_e,A_e,m_e=lstsq_it(A,neu_e)
    xhat_u,A_u,m_u=lstsq_it(A,neu_u)

    if plot_resid:
        plt.plot(n.dot(A_n,xhat_n)-m_n,".")
        plt.plot(n.dot(A_e,xhat_e)-m_e,".")
        plt.plot(n.dot(A_u,xhat_u)-m_u,".")
        plt.show()
    
    return(xhat_n,xhat_e,xhat_u)

class polycal:
    def __init__(self,x,y,az,el,image_width=1920,image_height=1080):
        """
        given a point cloud of image pixel positions normalized (x,y) and corresponding az,el pointings,
        gind a pointing direction model
        """
        self.image_width=image_width
        self.image_height=image_height

        # fit
        xn,yn=lm.pixel_units_to_normalized(x,y,w=image_width,h=image_height)
        
        neu_n=n.cos(n.pi*el/180)*n.cos(n.pi*az/180)
        neu_e=n.cos(n.pi*el/180)*n.sin(n.pi*az/180)
        neu_u=n.sin(n.pi*el/180)

        # lstsq polynomial values for north, east, and up directions
        npar,epar,upar=first_guess(xn,yn,az,el)

        self.az=az
        self.el=el
        
        self.npar=npar
        self.epar=epar
        self.upar=upar

        # grid search calculations
        pix_x=n.arange(image_width)
        pix_y=n.arange(image_height)
        self.pix_xx,self.pix_yy=n.meshgrid(pix_x,pix_y)

        self.grid_dim=self.pix_xx.shape
        
        self.norm_x,self.norm_y=lm.pixel_units_to_normalized(self.pix_xx,
                                                             self.pix_yy,w=image_width,h=image_height)
#        plt.pcolormesh(self.norm_x)
 #       plt.colorbar()
  #      plt.show()
        self.grid_n,self.grid_e,self.grid_u=self.get_neu(self.norm_x,self.norm_y)

        # on-axis neu
        ax_n, ax_e, ax_u=self.get_neu(0,0)
        self.axis = n.array([ax_n,ax_e,ax_u])
        # normalize
        self.axis=self.axis/n.linalg.norm(self.axis)
        
        # on-axis az,el
        r = n.hypot(self.axis[1], self.axis[0])
        elev = n.arctan2(self.axis[2], r)
        az = n.arctan2(self.axis[1], self.axis[0]) 
       
        self.el = 180.0*elev/n.pi
        self.az = 180.0*az/n.pi

        
    def get_neu(self,x,y):
        n_m=forward_polymodel(x,y,self.npar)
        e_m=forward_polymodel(x,y,self.epar)
        u_m=forward_polymodel(x,y,self.upar)
        return(n_m,e_m,u_m)

    def azel_to_xy(self,az,el):
        """
        tbd: find better than brute-force solution
        """
        point_n=n.cos(n.pi*el/180.0)*n.cos(n.pi*az/180.0)
        point_e=n.cos(n.pi*el/180.0)*n.sin(n.pi*az/180.0)
        point_u=n.sin(n.pi*el/180.0)

        res = (self.grid_n - point_n)**2.0 + (self.grid_e - point_e)**2.0 + (self.grid_u - point_u)**2.0
        print(n.min(res))
        idx_x=self.pix_xx[n.unravel_index(n.argmin(res),self.grid_dim)]
        idx_y=self.pix_yy[n.unravel_index(n.argmin(res),self.grid_dim)]
        return(idx_x,idx_y)

def get_polycal(cam_id="011331"):
    """
    calibrate camera
    """
    print("fitting %s"%(cam_id))
    fl=au.get_solved_videos(cam_id)
    print("found %d solutions"%(len(fl)))

    x=n.array([])
    y=n.array([])
    az=n.array([])
    el=n.array([])
        
    for f in fl:
        h=h5py.File(f,"r")
        gidx=n.where(h["weigth"][()] > 0.99)[0]
        x=n.concatenate((x,h["x_pix"][()][gidx]))
        y=n.concatenate((y,h["y_pix"][()][gidx]))
        az=n.concatenate((az,h["az_deg"][()][gidx]))
        el=n.concatenate((el,h["el_deg"][()][gidx]))
        h.close()

    pc=polycal(x,y,az,el,image_width=au.conf["image_width"],image_height=au.conf["image_height"])
    return(pc)
    

def plot_sky_sphere():
    """
    show the field of views of the cameras as an approxmate grid
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    cam_ids=au.get_cameras()
    
    xg=n.linspace(-1,1,num=20)
    y_max=au.conf["image_height"]/au.conf["image_width"]
    yg=n.linspace(-y_max,y_max,num=20)
    xgg,ygg=n.meshgrid(xg,yg)
    
    for cam in cam_ids:
        pc=get_polycal(cam)
        
        model_n, model_e, model_u = pc.get_neu(xgg,ygg)

        m_norm=n.sqrt(model_n**2.0 + model_e**2.0 + model_u**2.0)
        model_n = model_n/m_norm
        model_e = model_e/m_norm
        model_u = model_u/m_norm        

        ax.scatter3D(model_n.flatten(),model_e.flatten(),model_u.flatten(),label=cam)
    ax.set_xlabel("North")
    ax.set_ylabel("East")
    ax.set_zlabel("Up")        
  #  # show plot
    plt.legend(title="camera id")
    
    plt.show()
        
        

    
if __name__ == "__main__":

    plot_sky_sphere()
    
    cam_ids=au.get_cameras()
    
    for cam in cam_ids:
        
        pc=get_polycal(cam)
        print("on-axis az %1.3f el %1.3f"%(pc.az, pc.el))
        print(pc.azel_to_xy(pc.az,pc.el))

        fl = glob.glob("tests/*%s.mp4.orig.png"%(cam))
        for f in fl:

            t0 = au.file_name_to_datetime(f)
            obs = au.get_obs_loc()

            aa_frame = AltAz(obstime=t0, location=obs)
            
            
            I=imageio.imread(f)
            plt.imshow(I,vmax=128)
            plt.title(t0)

        
            bs=bsc.bright_stars()
            for i in range(500):
                d=bs.get_ra_dec_vmag(i)

                c = SkyCoord(ra=d[0]*u.degree, dec=d[1]*u.degree, frame='icrs')
                altaz=c.transform_to(aa_frame)
                star_az=float(altaz.az/u.deg)
                star_el=float(altaz.alt/u.deg)

                x,y=pc.azel_to_xy(star_az,star_el)
                plt.scatter(x,y,s=100,facecolors='none',edgecolors='white')
            plt.show()

        

