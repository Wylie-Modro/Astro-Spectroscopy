#Reduce spectral science frames from the KAST spectrograph.
#The reduction should include bias subtraction and flat field correction.

import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls
import pyfits

image = pyfits.open('data-2013-10-26-shane-public/biasData/b101.fits')
#print('image: '+str(type(image)))
#print('image: '+str(image))
#imagehead = image[0].header
#print imagehead['EXPTIME']

imagedata = image[0].data
for i in imagedata:
    print(i)
print(len(imagedata))

LoadData = SpectraCls.LoadingData()
DataTools = SpectraCls.DataTools()
'''
allBiasData = LoadData.LoadTextFromDirectoryIntoArray('b', 'data-2013-10-26-shane-public/biasData/')


        listOfData = []

        print("Loading in images ...")

        for filename in os.listdir(directoryPath):
    
            if commonStartOfFileName in str(filename):
                data = np.loadtxt(directoryPath+filename, skiprows=17, comments = '>>', usecols=(0,1))
                listOfData.append(data)
            else:
                print('Found: ' + str(filename))
    
        return listOfData

averagedBiasData = DataTools.GetAveragedImage(allBiasData, 2048)

fig = plt.figure()
plt.show()
'''
