#Reduce spectral science frames from the KAST spectrograph.
#The reduction should include bias subtraction and flat field correction.

import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls
import pyfits

image = pyfits.open('data-2013-10-26-shane-public/b153.fits')
imagehead = image[0].header
print imagehead['EXPTIME']

imagedata = image[0].data
print(type(imagedata))