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
    xlength=len(i)
ylength=len(imagedata)

LoadData = SpectraCls.LoadingData()
DataTools = SpectraCls.DataTools()

[allBiasData,xlength,ylength] = LoadData.LoadDataFromDirectoryIntoArray('b1', 'data-2013-10-26-shane-public/biasData/')

averagedResult = DataTools.NewGetAveragedData(allBiasData, xlength, ylength)#print('averagedResult: ' + str(averagedResult))
'''
averagedBiasData = DataTools.GetAveragedData(allBiasData, 2048)
fig = plt.figure()     
ax1 = fig.add_subplot(111)
DataTools.SpectraPlot(averagedBiasData, ax1, 'Averaged Bias Image', 'xPixel', 'yPixel')
plt.show()
'''
#print(allBiasData)
print(xlength)
print(ylength)
print('-------------------------------------------------------')

#for dataSet in allBiasData:
    #print('dataSet: ' + str(dataSet))
    


