#!/usr/bin/bash

# install astrometry.net and other packages needed for star cal

sudo apt-get install python3-photutils python3-opencv
sudo apt-get install astrometry.net libastrometry-dev libastrometry0

cd /usr/share/astrometry
sudo wget http://broiler.astrometry.net/\~dstn/4100/index-4119.fits
sudo wget http://broiler.astrometry.net/\~dstn/4100/index-4118.fits
sudo wget http://broiler.astrometry.net/\~dstn/4100/index-4117.fits
sudo wget http://broiler.astrometry.net/\~dstn/4100/index-4116.fits 
