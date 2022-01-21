import numpy as n
import matplotlib.pyplot as plt
import h5py
import os


import amscam_utils as au

import cv2
def lstsq_it(A,m,outlier_rem=True):
    """
    iterative least-squares fit to remove outliers
    """
    if outlier_rem:
        for i in range(3):
            xhat=n.linalg.lstsq(A,m,rcond=None)[0]
            resid=n.dot(A,xhat)-m
            aresid=n.abs(resid)
#            plt.plot(resid,".")
 #           plt.show()
            std_est=n.median(aresid)
            gidx=n.where(aresid < 5.0*std_est)[0]
            
            A = n.copy(A[gidx,:])
            m = n.copy(m[gidx])

    xhat=n.linalg.lstsq(A,m,rcond=None)[0]
    return(xhat,A,m)


def forward_polymodel0(x,y,par):
    """
    Simplest polynomial model
    """
    model = par[0] + par[1]*x + par[2]*y
    return(model)


def forward_polymodel1(x,y,par):
    """
    Simple second order polynomial 
    """
    model = par[0] + par[1]*x + par[2]*y + par[3]*x**2.0 + par[4]*y**2.0 + par[5]*x*y
    return(model)


def forward_polymodel2(x,y,par):
    """
    Simple third order polynomial 
    """
    model = par[0] + par[1]*x + par[2]*y + par[3]*x**2.0 + par[4]*y**2.0 + par[5]*x*y + par[6]*x**3.0 + par[7]*y**3.0 + par[8]*(x**2.0)*y + par[9]*x*(y**2.0)
    return(model)

def forward_polymodel3(x,y,par):
    """
    Simple third order polynomial 

    A[:,10]=x**4.0
    A[:,11]=y**4.0
    A[:,12]=(x**2.0)*(y**2.0)
    A[:,13]=(x**1.0)*(y**3.0)
    A[:,14]=(x**3.0)*(y**1.0)        

    """
    model = par[0] + par[1]*x + par[2]*y + par[3]*x**2.0 + par[4]*y**2.0 + par[5]*x*y + par[6]*x**3.0 + par[7]*y**3.0 + par[8]*(x**2.0)*y + par[9]*x*(y**2.0) + par[10]*x**4.0 + par[11]*y**4.0 + par[12]*(x**2.0)*(y**2.0) + par[13]*(x**1.0)*(y**3.0) + par[14]*(x**3.0)*(y**1.0)
    return(model)


def first_guess0(x,y,az,el,plot_resid=False):
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
    A = n.zeros([n_m,3])
    A[:,0]=1.0
    A[:,1]=x
    A[:,2]=y    

    xhat_n=n.linalg.lstsq(A,neu_n,rcond=None)[0]
    xhat_e=n.linalg.lstsq(A,neu_e,rcond=None)[0]
    xhat_u=n.linalg.lstsq(A,neu_u,rcond=None)[0]    
    
    return(xhat_n,xhat_e,xhat_u)


def first_guess1(x,y,az,el,plot_resid=False):
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

def first_guess2(x,y,az,el,plot_resid=False):
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
    A = n.zeros([n_m,10])
    A[:,0]=1.0
    A[:,1]=x
    A[:,2]=y    
    A[:,3]=x**2.0
    A[:,4]=y**2.0
    A[:,5]=x*y
    A[:,6]=x**3.0
    A[:,7]=y**3.0
    A[:,8]=(x**2.0)*y
    A[:,9]=x*y**2.0    
    
    xhat_n,A_n,m_n=lstsq_it(A,neu_n)
    xhat_e,A_e,m_e=lstsq_it(A,neu_e)
    xhat_u,A_u,m_u=lstsq_it(A,neu_u)

    if plot_resid:
        fig = plt.figure()

        plt.plot(n.dot(A_n,xhat_n)-m_n, ".")
        plt.plot(n.dot(A_e,xhat_e)-m_e, ".")
        plt.plot(n.dot(A_u,xhat_u)-m_u, ".")
        plt.show()
    
    return(xhat_n,xhat_e,xhat_u)


def first_guess3(x,y,az,el,plot_resid=False):
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
    A = n.zeros([n_m,15])
    A[:,0]=1.0
    A[:,1]=x
    A[:,2]=y    
    A[:,3]=x**2.0
    A[:,4]=y**2.0
    A[:,5]=x*y
    A[:,6]=x**3.0
    A[:,7]=y**3.0
    A[:,8]=(x**2.0)*y
    A[:,9]=x*y**2.0
    A[:,10]=x**4.0
    A[:,11]=y**4.0
    A[:,12]=(x**2.0)*(y**2.0)
    A[:,13]=(x**1.0)*(y**3.0)
    A[:,14]=(x**3.0)*(y**1.0)        
    
    xhat_n,A_n,m_n=lstsq_it(A,neu_n)
    xhat_e,A_e,m_e=lstsq_it(A,neu_e)
    xhat_u,A_u,m_u=lstsq_it(A,neu_u)

    if plot_resid:
        fig = plt.figure()

        plt.plot(n.dot(A_n,xhat_n)-m_n, ".")
        plt.plot(n.dot(A_e,xhat_e)-m_e, ".")
        plt.plot(n.dot(A_u,xhat_u)-m_u, ".")
        plt.show()
    
    return(xhat_n,xhat_e,xhat_u)


#fit_fun = first_guess
#forward_model = forward_polymodel


class polycal:
    def __init__(self,fname=None,x=[],y=[],az=[],el=[],model_order=None,cam_id="default",image_width=1920,image_height=1080):
        """
        Given a point cloud of image pixel positions normalized (x,y) and corresponding az,el pointings,
        gind a pointing direction model
        """
        self.image_width=image_width
        self.image_height=image_height
        self.cam_id=cam_id

        print("init")
        print(fname)

        if fname != None:
            h=h5py.File(fname,"r")
            x=n.copy(h["x"][()])
            y=n.copy(h["y"][()])
            az=n.copy(h["az"][()])
            el=n.copy(h["el"][()])
            self.image_width=n.copy(h["image_width"][()])
            self.image_height=n.copy(h["image_height"][()])
            self.cam_id = n.copy(h["cam_id"][()])
            h.close()
            print("%d points in file"%(len(x)))

            self.fname=fname
            self.model_order = 0
            if len(x) < 3:
                raise Exception("not enough points")
            elif len(x) < 10:
                self.model_order = 1
            elif len(x) < 50:
                self.model_order = 2
            else:
                self.model_order = 3
                
            if model_order != None:
                self.model_order=model_order
            
            
            self.fit_pointcloud(x,y,az,el)
        elif len(x) == len(y) == len(az) == len(el) and len(x) > 0:
            self.model_order = 0
            if len(x) < 3:
                raise Exception("not enough points")
            elif len(x) < 10:
                self.model_order = 1
            elif len(x) < 50:
                self.model_order = 2
            else:
                self.model_order = 3
            if model_order != None:
                self.model_order=model_order
                
            
            self.fit_pointcloud(n.array(x),n.array(y),n.array(az),n.array(el))
        else:
            raise Exception("pass file name with point cloud or pass (x,y,az,el) that are same length should be same length")
            
    def save(self,fname="default"):
        self.fname=fname
        ho=h5py.File(fname,"w")
        ho["x"]=self.x
        ho["y"]=self.y
        ho["az"]=self.az
        ho["el"]=self.el
        ho["cam_id"]=self.cam_id
        ho["image_width"]=self.image_width
        ho["image_height"]=self.image_height        
        ho.close()

    def fit_pointcloud(self, x, y, az, el):
        self.x=x
        self.y=y
        self.az=az
        self.el=el
        # fit to zero-centered
        xn=x - self.image_width/2.0 + 0.5
        yn=y - self.image_height/2.0 + 0.5
#        xn,yn=lm.pixel_units_to_normalized(x,y,w=self.image_width,h=self.image_height)
        
        neu_n=n.cos(n.pi*el/180)*n.cos(n.pi*az/180)
        neu_e=n.cos(n.pi*el/180)*n.sin(n.pi*az/180)
        neu_u=n.sin(n.pi*el/180)

        # lstsq polynomial values for north, east, and up directions
        if self.model_order == 0:
            npar,epar,upar=first_guess0(xn,yn,az,el)
        elif self.model_order == 1:
            npar,epar,upar=first_guess1(xn,yn,az,el)            
        elif self.model_order == 2:
            npar,epar,upar=first_guess2(xn,yn,az,el)            
        elif self.model_order == 3:
            npar,epar,upar=first_guess3(xn,yn,az,el)            

#        print(npar)
 #       print(epar)
  #      print(upar)

        self.az=az
        self.el=el
        
        self.npar=npar
        self.epar=epar
        self.upar=upar

        # grid search calculations
        self.pix_x=n.arange(self.image_width)
        self.pix_y=n.arange(self.image_height)
        self.pix_xx,self.pix_yy=n.meshgrid(self.pix_x,self.pix_y)

        self.grid_dim=self.pix_xx.shape

        self.norm_x = self.pix_xx - self.image_width/2.0 + 0.5
        self.norm_y = self.pix_yy - self.image_height/2.0 + 0.5   
#        self.norm_x,self.norm_y=lm.pixel_units_to_normalized(self.pix_xx,
#                                                             self.pix_yy,w=self.image_width,h=self.image_height)

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
       
        self.on_axis_el = 180.0*elev/n.pi
        self.on_axis_az = 180.0*az/n.pi
        
        
    def get_neu(self,x,y):

        if self.model_order == 0:
            n_m=forward_polymodel0(x,y,self.npar)
            e_m=forward_polymodel0(x,y,self.epar)
            u_m=forward_polymodel0(x,y,self.upar)
            return(n_m,e_m,u_m)            
        elif self.model_order == 1:
            n_m=forward_polymodel1(x,y,self.npar)
            e_m=forward_polymodel1(x,y,self.epar)
            u_m=forward_polymodel1(x,y,self.upar)
            return(n_m,e_m,u_m)                        
        elif self.model_order == 2:
            n_m=forward_polymodel2(x,y,self.npar)
            e_m=forward_polymodel2(x,y,self.epar)
            u_m=forward_polymodel2(x,y,self.upar)
            return(n_m,e_m,u_m)                        
        elif self.model_order == 3:
            n_m=forward_polymodel3(x,y,self.npar)
            e_m=forward_polymodel3(x,y,self.epar)
            u_m=forward_polymodel3(x,y,self.upar)
            return(n_m,e_m,u_m)                        
        
    def to_gnomic(self,I,scale=0.6,plot=False):

        max_x=n.max(self.image_width/2.0)
        max_y=n.max(self.image_height/2.0)

        # guess directions based on polycal
        pn,pe,pu=self.get_neu(0.25*max_x,0)
        print(pn,pe,pu)

        self.u_z = self.axis
        self.u_z = self.u_z/n.linalg.norm(self.u_z)
        print("onax ",self.u_z)
        self.u_y = -n.cross(self.u_z, n.array([pn,pe,pu]))
        self.u_y = self.u_y/n.linalg.norm(self.u_y)
        self.u_x = n.cross(self.u_z, self.u_y)
        print("up ",self.u_y)
        print("right ",self.u_x)

        I_gnomic=n.copy(I)

        new_dim=(int(I.shape[1]*scale),int(I.shape[0]*scale))
        print(new_dim)
        I_gnomic = cv2.resize(I_gnomic, new_dim)        

        I_gnomic[:,:]=0.0
        grid_norm = n.sqrt(self.grid_n**2.0+self.grid_e**2.0+self.grid_u**2.0)

        p_d_x = self.grid_n*self.u_x[0] + self.grid_e*self.u_x[1] + self.grid_u*self.u_x[2]
        p_d_y = self.grid_n*self.u_y[0] + self.grid_e*self.u_y[1] + self.grid_u*self.u_y[2]
        alpha=n.arctan2(p_d_y,p_d_x)
        
#        p_d_y = point_x*self.u_y[0] + point_y*self.u_y[1] + point_z*self.u_y[2]
 #       p_d_z = point_x*self.u_z[0] + point_y*self.u_z[1] + point_z*self.u_z[2]

#        theta_x = n.arctan2(p_d_x,p_d_z)
 #       theta_y = n.arctan2(p_d_y,p_d_z)        
        
#        alpha = n.arctan2(self.norm_y,self.norm_x)

        theta=n.arccos( (self.axis[0]*self.grid_n + self.axis[1]*self.grid_e + self.axis[2]*self.grid_u)/grid_norm )
        print(theta.shape)
        theta_max = theta[0,1920-1]
        #R_g = f_g*n.tan(theta)
        self.f_g = (self.image_width/2.0)/n.tan(theta_max)
        print(self.f_g)
        #1920.0/f_g = f_g*n.tan(theta)
        #f_g =

        R_g=self.f_g*n.tan(theta)
        #
        # 
        # 
        xg = n.array(n.round( scale*(R_g*n.cos(alpha) + self.image_width/2.0 - 0.5)),dtype=n.int)
        yg = n.array(n.round( scale*(R_g*n.sin(alpha) + self.image_height/2.0 - 0.5)),dtype=n.int)
        xg[xg < 0]=0
        yg[yg < 0]=0
        xg[xg > (scale*self.image_width-1)]=(scale*self.image_width-1)
        yg[yg > (scale*self.image_height-1)]=(scale*self.image_height-1)
        print(n.max(xg))
        print(n.max(yg))        
#        plt.pcolormesh(xg)
 #       plt.colorbar()
  #      plt.show()
   #     plt.pcolormesh(yg)
    #    plt.colorbar()
     #   plt.show()

        I_gnomic[yg,xg]=I

        I_gnomic=cv2.resize(I_gnomic,(I.shape[1],I.shape[0]))
        if plot:
            plt.imshow(I_gnomic)
            plt.show()
        return(I_gnomic)


    def azel_to_xy(self,az,el):
        """
        tbd: find better than brute-force solution
        """
        point_n=n.cos(n.pi*el/180.0)*n.cos(n.pi*az/180.0)
        point_e=n.cos(n.pi*el/180.0)*n.sin(n.pi*az/180.0)
        point_u=n.sin(n.pi*el/180.0)

        res = (self.grid_n - point_n)**2.0 + (self.grid_e - point_e)**2.0 + (self.grid_u - point_u)**2.0

        ami=n.argmin(res)
        bidx=n.unravel_index(ami,self.grid_dim)

        if res[bidx] < 1e-3:
            idx_x=self.pix_xx[bidx]
            idx_y=self.pix_yy[bidx]
            return(idx_x,idx_y)
        else:
            return(None,None)

def get_polycal(cam_id="011331",yale=True, astrometry=True, model_order=1):
    """
    calibrate camera
    """
    print("fitting %s"%(cam_id))

    yale_fname = "calibrations/yale_%s.h5"%(cam_id)
    astrometry_fname = "calibrations/%s.h5"%(cam_id)

    c_x = n.array([])
    c_y = n.array([])    
    c_az = n.array([])
    c_el = n.array([])    
    
    if os.path.exists(yale_fname) and yale:
        print("using yale save file")
        h=h5py.File(yale_fname,"r")
        c_x = n.concatenate( (c_x, h["x"][()]) )
        c_y = n.concatenate( (c_y, h["y"][()]) )
        c_az = n.concatenate( (c_az, h["az"][()]) )
        c_el = n.concatenate( (c_el, h["el"][()]) )        
        h.close()
#        pc=polycal(fname=yale_fname)#x=x,y=y,az=az,el=el,image_width=au.conf["image_width"],image_height=au.conf["image_height"],cam_id=cam_id)

    if os.path.exists(astrometry_fname) and astrometry:
        print("using astrometry save file")
        h=h5py.File(astrometry_fname,"r")        
        c_x = n.concatenate( (c_x, h["x"][()]) )
        c_y = n.concatenate( (c_y, h["y"][()]) )
        c_az = n.concatenate( (c_az, h["az"][()]) )
        c_el = n.concatenate( (c_el, h["el"][()]) )
        h.close()
        
        pc=polycal(fname=astrometry_fname)#x=x,y=y,az=az,el=el,image_width=au.conf["image_width"],image_height=au.conf["image_height"],cam_id=cam_id)
    elif astrometry:
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

        c_x = n.concatenate( (c_x, x) )
        c_y = n.concatenate( (c_y, y) )
        c_az = n.concatenate( (c_az, az) )
        c_el = n.concatenate( (c_el, el) )

    if len(c_x)>3:
        print("found enough points (%d) to solve lens"%(len(c_x)))
    else:
        raise Exception("not enough points found (%d)"%(len(c_x)))
    pc=polycal(x=c_x,y=c_y,az=c_az,el=c_el,image_width=au.conf["image_width"],image_height=au.conf["image_height"],cam_id=cam_id, model_order=model_order)
#    pc.save(fname="calibrations/%s.h5"%(cam_id))
    
    return(pc)
    



if __name__ == "__main__":
    import imageio
    pc=get_polycal(cam_id="011337")
    import glob
    import re
    fl=glob.glob("tests/*.orig.png")
    for f in fl:
        prefix=re.search("(.*).(...)",f).group(1)
        print(prefix)
        I=imageio.imread(f)
        I_gnomic=pc.to_gnomic(I,plot=False)
        imageio.imwrite("%s.gnom.png"%(prefix),I_gnomic)
