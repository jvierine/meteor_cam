import numpy as n
import matplotlib.pyplot as plt


# use this as a first guess
import polycal as pcal


import star_finder as sfm
import amscam_utils as au
import glob
import imageio
import scipy.optimize as sio

class bjorncal:
    def __init__(self,pc):
        #
        # camera unit vectors:
        #
        # p_x (positive x-pixel direction)
        # p_y (positive y-pixel direction)        
        # p_z (on-axis)
        #
        # arbitrary pointing direction p = [p_x, p_y, p_z], |p|=1
        #
        # alpha = arctan2(p_y, p_x)
        # theta = arccos(p_z)
        #
        # xpix = [f_x sin(alpha_x theta)]*cos(alpha) + x_shift
        # ypix = [f_y sin(alpha_y theta)]*sin(alpha) + y_shift        
        #
        self.pc=pc
        #
        #
#        self.pc.on_axis

        # Model parameters:
        # On_axis unit vector u_z (in local horizon NEU) (3x1)
        # u_y unit vector (3x1)
        # f_x, f_y, alpha_x, alpha_y x_shift, y_shift
        #
        # u_x = u_z cross u_y
        #
        #
        # first guess:
        self.x_shift = 0.0
        self.y_shift = 0.0
        self.alpha_x = 1.0
        self.alpha_y = 1.0
        self.angles=n.zeros(3)

        self.image_width=pc.image_width
        self.image_height=pc.image_height

        max_x=n.max(pc.pix_x)
        max_y=n.max(pc.pix_y)

        # guess directions based on polycal
        pn,pe,pu=pc.get_neu(0.25*max_x,0)
        print(pn,pe,pu)

        self.u_z = pc.axis
        self.u_z = self.u_z/n.linalg.norm(self.u_z)
        print("onax ",self.u_z)
        self.u_y = -n.cross(self.u_z, n.array([pn,pe,pu]))
        self.u_y = self.u_y/n.linalg.norm(self.u_y)
        self.u_x = n.cross(self.u_z, self.u_y)
        print("up ",self.u_y)
        print("right ",self.u_x)

        self.orig_u_x = n.copy(self.u_x)
        self.orig_u_y = n.copy(self.u_y)
        self.orig_u_z = n.copy(self.u_z)

        print("on axis az %1.2f el %1.2f"%(pc.on_axis_az, pc.on_axis_el))
        
        # guess f_x 
        pn,pe,pu=pc.get_neu(0.25*max_x,0)
        xpoint = n.array([pn,pe,pu])
        # theta = xpoint
        theta = n.arccos(n.dot(self.u_z,xpoint)/n.linalg.norm(xpoint))
        self.f_x = 0.25*max_x/n.sin(theta)/1000.0

        # guess f_y
        pn,pe,pu=pc.get_neu(0,0.25*max_y)
        ypoint = n.array([pn,pe,pu])
        theta = n.arccos(n.dot(self.u_z,ypoint)/n.linalg.norm(ypoint))
        self.f_y = 0.25*max_y/n.sin(theta)/1000.0
        print(self.f_y)


        # calibration data in polycal object
        
        # fit to zero-centered
        self.star_x=self.pc.x - self.image_width/2.0 + 0.5
        self.star_y=self.pc.y - self.image_height/2.0 + 0.5
        self.star_az=self.pc.az
        self.star_el=self.pc.el
        self.star_n = n.cos(n.pi*self.star_az/180.0)*n.cos(n.pi*self.star_el/180.0)
        self.star_e = n.sin(n.pi*self.star_az/180.0)*n.cos(n.pi*self.star_el/180.0)        
        self.star_u = n.sin(n.pi*self.star_el/180.0)

        old_pars=self.get_par()
        print("sum of squares %1.2f"%(self.ss(plot=True)))
        self.optimize()
        print("old pars",old_pars)

        self.ss(plot=True)


    def rotate_basis(self,angles):
        Rx = n.zeros([3,3])
        Rx[0,0]=1.0
        Rx[1,1]=n.cos(angles[0])
        Rx[1,2]=-n.sin(angles[0])
        Rx[2,1]=n.sin(angles[0])
        Rx[2,2]=n.cos(angles[0])
        
        Ry = n.zeros([3,3])
        Ry[0,0]=n.cos(angles[1])
        Ry[0,2]=n.sin(angles[1])
        Ry[1,1]=1.0
        Ry[2,0]=-n.sin(angles[1])
        Ry[2,2]=n.cos(angles[1])
        
        Rz = n.zeros([3,3])        

        Rz[0,0]=n.cos(angles[2])
        Rz[0,1]=-n.sin(angles[2])
        Rz[1,0]=n.sin(angles[2])
        Rz[1,1]=n.cos(angles[2])
        Rz[2,2]=1.0

        Rm=n.dot(n.dot(Rx,Ry),Rz)
        self.u_x=n.dot(Rm,self.orig_u_x)
        self.u_y=n.dot(Rm,self.orig_u_y)
        self.u_z=n.dot(Rm,self.orig_u_z)


    def get_par(self):
        self.par=n.array([self.f_x,
                          self.f_y,
                          self.x_shift,
                          self.y_shift,
                          self.alpha_x,
                          self.alpha_y,
                          self.angles[0],
                          self.angles[1],
                          self.angles[2]])
        return(self.par)
    
    def set_par(self, par):
        self.f_x=par[0]
        self.f_y=par[1]
        self.x_shift=par[2]
        self.y_shift=par[3]
        self.alpha_x=par[4]
        self.alpha_y=par[5]
        self.angles[0]=par[6]
        self.angles[1]=par[7]
        self.angles[2]=par[8]
        self.rotate_basis(self.angles)
        self.par=par
        

    def get_fx(self):
        return(1000.0*self.f_x)
    def get_fy(self):
        return(1000.0*self.f_y)
    
    def get_neu(self,x,y):
        """
        x and y in normalized coordinates defined as such
        x = n.arange(image_width) - image_width/2.0 + 0.5
        y = n.arange(image_height) - image_height/2.0 + 0.5
        """
        
        # - self.image_width/2.0 + 0.5
        theta_x = n.arcsin((x-self.x_shift)/self.get_fx())/self.alpha_x
        theta_y = n.arcsin((y-self.y_shift)/self.get_fy())/self.alpha_y

        A_x =  n.tan(theta_x)
        A_y =  n.tan(theta_y)

#        p = A_x*self.u_x + A_y*self.u_y + self.u_z
        p_n = A_x*self.u_x[0] + A_y*self.u_y[0] + self.u_z[0]
        p_e = A_x*self.u_x[1] + A_y*self.u_y[1] + self.u_z[1]
        p_u = A_x*self.u_x[2] + A_y*self.u_y[2] + self.u_z[2]
        p_norm = n.sqrt(p_n**2.0 + p_e**2.0 + p_u**2.0)
        p_n = p_n/p_norm
        p_e = p_e/p_norm
        p_u = p_u/p_norm        
        
        return(p_n,p_e,p_u)
        
        
    def azel_to_xy(self,az,el):
        
        point_z=n.sin(n.pi*el/180.0)
        point_x=n.cos(n.pi*az/180.0)*n.cos(n.pi*el/180.0)
        point_y=n.sin(n.pi*az/180.0)*n.cos(n.pi*el/180.0)

        p_d_x = point_x*self.u_x[0] + point_y*self.u_x[1] + point_z*self.u_x[2]
        p_d_y = point_x*self.u_y[0] + point_y*self.u_y[1] + point_z*self.u_y[2]
        p_d_z = point_x*self.u_z[0] + point_y*self.u_z[1] + point_z*self.u_z[2]

        theta_x = n.arctan2(p_d_x,p_d_z)
        theta_y = n.arctan2(p_d_y,p_d_z)        
        
        
        # - self.image_width/2.0 + 0.5
        xpix = self.get_fx()*n.sin(self.alpha_x * theta_x) + self.x_shift  + self.image_width/2.0 - 0.5
        ypix = self.get_fy()*n.sin(self.alpha_y * theta_y) + self.y_shift  + self.image_height/2.0 - 0.5

        if  theta_x > 90 or theta_y > 90:
            return(None,None)
        return(xpix, ypix)


    def ss(self,plot=False):

        cal_n,cal_e, cal_u = self.get_neu(self.star_x, self.star_y)

#        errsum = n.sum((n.abs(cal_n - self.star_n) + n.abs(cal_e - self.star_e) + n.abs(cal_u - self.star_u)))
        errsum = n.sum(n.abs(cal_n - self.star_n) + n.abs(cal_e - self.star_e) + n.abs(cal_u - self.star_u))
        
        if plot:
            plt.plot(cal_n-self.star_n,".")
            plt.plot(cal_e-self.star_e,".")
            plt.plot(cal_u-self.star_u,".")            
            plt.show()
        return(errsum)


    def optimize(self):
        def ss(trypar):
            old_par = self.get_par()
            self.set_par(trypar)
            errsum=self.ss()
            # go back to old
            self.set_par(old_par)
            print(errsum)
            return(errsum)
        xhat=sio.fmin(ss,self.get_par(),maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)                        
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)                        
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)                        
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)
        xhat=sio.fmin(ss,xhat,maxiter=100000)                        
        print(xhat)        
        self.set_par(xhat)




        


        

    


def get_bjorn_cal(cam_id="011331"):
    """
    calibrate camera
    """
    import amscam_utils as au
    import h5py
    print("fitting %s"%(cam_id))

    print("using gathering azel maps")        
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
    
    if len(x)>10:
        print("found enough points to solve lens")
    else:
        raise Exception("not enough points found")
    pc=pcal.polycal(x=x,y=y,az=az,el=el,image_width=au.conf["image_width"],image_height=au.conf["image_height"],cam_id=cam_id)

    bc=bjorncal(pc)
    
    
    return(bc)



def find_yale_matches(station_id="AMS133"):
    
    cam_ids=au.get_cameras(station_id=station_id)
    
    for cam in cam_ids:
        
        bc=get_bjorn_cal(cam_id=cam)        


        yale_x = []
        yale_y = []
        yale_az = []
        yale_el = []

        x_resid = []
        y_resid = []
            
        sf=sfm.star_finder(pc=bc)            

        fl = glob.glob("tests/*%s.mp4.orig.png"%(cam))
        for f in fl:
            I=imageio.imread(f)
            t0 = au.file_name_to_datetime(f)
            obs = au.get_obs_loc(cam)
                
            x,y,az,el=sf.find_bright_stars_in_image(t0,obs)
  
            plt.imshow(I,vmax=128)
            plt.xlim([0,I.shape[1]])
            plt.ylim([I.shape[0],0])            
            plt.title(t0)
            

            plt.scatter(x,y,s=100,facecolors='none',edgecolors='yellow')
            plt.tight_layout()
            plt.show()



if __name__ == "__main__":
    find_yale_matches()
    bc=get_bjorn_cal(cam_id="011331")
    x,y=bc.azel_to_xy(-142.16,15.48)
    print(x-au.conf["image_width"]/2.0 + 0.5)
    print(y-au.conf["image_height"]/2.0 + 0.5)    
    
    p_n,p_e,p_u=bc.get_neu(x-au.conf["image_width"]/2.0 + 0.5,y-au.conf["image_height"]/2.0 + 0.5)
#    print(p_n)
 #   print(p_e)
  #  print(p_u)    
    #
    #print(bc.azel_to_xy(-152.16,15.48))
