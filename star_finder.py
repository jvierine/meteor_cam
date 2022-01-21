import bright_stars as bsc


from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.time import TimeDelta
from astropy.coordinates import SkyCoord
import numpy as n

from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder


class star_finder:
    def __init__(self,pc):
        self.pc=pc
        self.bs=bsc.bright_stars()        

    def find_bright_stars_in_image(self,t0,obs,N_stars=1000):
        
        aa_frame = AltAz(obstime=t0, location=obs)
        xs=[]
        ys=[]
        azs=[]
        els=[]
        for i in range(N_stars):
            d=self.bs.get_ra_dec_vmag(i)
            c = SkyCoord(ra=d[0]*u.degree, dec=d[1]*u.degree, frame='icrs')
            
            altaz=c.transform_to(aa_frame)
            star_az=float(altaz.az/u.deg)
            star_el=float(altaz.alt/u.deg)
        
            x,y=self.pc.azel_to_xy(star_az,star_el)
            if x != None:
                xs.append(x)
                ys.append(y)
                azs.append(star_az)
                els.append(star_el)                
        return(n.array(xs),n.array(ys),n.array(azs),n.array(els))



def detect_stars(data,sigma=4):
    """
    detect stars within image
    """

#    from astropy.stats import SigmaClip
 #   from photutils.background import Background2D, MedianBackground
 #   sigma_clip = SigmaClip(sigma=3.)
 #   bkg_estimator = MedianBackground()
 #   bkg = Background2D(data, (50, 50), filter_size=(3, 3),
 #                      sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

 #   data=data-bkg.background

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    
#    plt.imshow(data,vmin=0)#median,vmax=median+3*std)
 #   plt.colorbar()


    daofind = DAOStarFinder(fwhm=3.0, threshold=sigma*std)    
    sources = daofind(data - median)
    return(sources)

#    if sources != None:
 #       plt.scatter(sources['xcentroid'], sources['ycentroid'],s=100,facecolors='none',edgecolors='white')
  #  plt.show()


def test_star_det():
    fl = glob.glob("tests/*orig.png")
    fl.sort()
    for f in fl:
        data=n.array(imageio.imread(f),dtype=n.float32)
        detect_stars(data[0:400,0:400])

