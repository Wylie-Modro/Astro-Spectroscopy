#Reduce spectral science frames from the KAST spectrograph.
#The reduction should include bias subtraction and flat field correction.

import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls
import pyfits


LoadData = SpectraCls.LoadingData()
DataTools = SpectraCls.DataTools()
CentroidAl = SpectraCls.CentroidAlgorithm()

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


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.title.set_text('b151 Normalized')
ax2.title.set_text('b157 Normalized')
ax1.set_xlabel('Pixel Number')
ax1.set_ylabel('Intensity (W/m^2)')
ax2.set_xlabel('Pixel Number')
ax2.set_ylabel('Intensity (W/m^2)')   
ax1.plot(oneDb151Normalized)
ax2.plot(oneDb157Normalized)
#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.title.set_text('b151 Normalized')
ax2.title.set_text('b157 Normalized')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Intensity (W/m^2)')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Intensity (W/m^2)')   
ax1.plot(CentroidAl.SimplePixlToWLMapping(np.arange(len(oneDb151Normalized))), oneDb151Normalized)
ax2.plot(CentroidAl.SimplePixlToWLMapping(np.arange(len(oneDb157Normalized))), oneDb157Normalized)
plt.show()

