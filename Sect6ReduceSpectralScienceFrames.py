#Reduce spectral science frames from the KAST spectrograph.
#The reduction should include bias subtraction and flat field correction.

import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls
import pyfits

'''
image = pyfits.open('data-2013-10-26-shane-public/biasData/b101.fits')
#print('image: '+str(type(image)))
#print('image: '+str(image))
#imagehead = image[0].header
#print imagehead['EXPTIME']

imagedata = image[0].data
for i in imagedata:
    xlength=len(i)
ylength=len(imagedata)
'''

LoadData = SpectraCls.LoadingData()
DataTools = SpectraCls.DataTools()

[allBiasData,xlength,ylength] = LoadData.LoadDataFromDirectoryIntoArray('b1', 'data-2013-10-26-shane-public/biasData/')
averagedBiasData = DataTools.NewGetAveragedData(allBiasData, xlength, ylength)

[b151ScienceData,xlength,ylength] = LoadData.LoadDataFromDirectoryIntoArray('b151', 'data-2013-10-26-shane-public/scienceData/')
[b157ScienceData,xlength,ylength] = LoadData.LoadDataFromDirectoryIntoArray('b157', 'data-2013-10-26-shane-public/scienceData/')

s151MinusBias = np.array(b151ScienceData)-averagedBiasData
s157MinusBias = np.array(b157ScienceData)-averagedBiasData
#print(s151MinusBias)
#print(s157MinusBias)

[allDomeData,xlength,ylength] = LoadData.LoadDataFromDirectoryIntoArray('b1', 'data-2013-10-26-shane-public/domeData/')
averagedDomeData = DataTools.NewGetAveragedData(allDomeData, xlength, ylength)
domeMinusBias = averagedDomeData-averagedBiasData
#print(domeMinusBias)

flatNormalized = (domeMinusBias+1)/(np.median(domeMinusBias)+1)

#print(flatNormalized)

#print(s151MinusBias/(flatNormalized+1))
oneDb151Normalized=(s151MinusBias/(flatNormalized+1)).flatten()
oneDb157Normalized=(s157MinusBias/(flatNormalized+1)).flatten()

plt.figure(1)
plt.subplot(211)
plt.plot(oneDb151Normalized)

plt.subplot(212)
plt.plot(oneDb157Normalized)
plt.show()
'''
fig1 = plt.figure(1)

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
'''

