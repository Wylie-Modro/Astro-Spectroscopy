import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls

#Part A: Determine the wavelength calibration of the spectrometer, i.e., the mapping between pixel number and wavelength
#-----------------------------------------------------------------------------------------------------------------------
#Initialize classes
LoadData = SpectraCls.LoadingData()
DataTools = SpectraCls.DataTools()
CentroidAl = SpectraCls.CentroidAlgorithm()
    
#Load Data and get averaged-spectra
allHeGDL = LoadData.LoadTextFromDirectoryIntoArray('HeGDL', '2017Lab2GroupB/HeGDL/')
averagedHeGDL = DataTools.GetAveragedImage(allHeGDL , 2048)

#Create dict and convert keys to wavelength outputed as np.array
HeGDLDictPixel = DataTools.CreateDictOfSpectra(averagedHeGDL)

HeGDLWLList = []
for key, val in HeGDLDictPixel.items():
    HeGDLWLList.append([CentroidAl.SimplePixlToWLMapping(key), val])
HeGDLWLnpArr = np.array(HeGDLWLList)  


#Setup plots comparing pixel number and wavelength representations
fig = plt.figure()     
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
DataTools.SpectraPlot(averagedHeGDL, ax1, 'Averaged HeGDL (Pixel)', 'Pixel (Number)', 'Intensity (Counts)')
DataTools.SpectraPlot(HeGDLWLnpArr, ax2, 'Averaged HeGDL (with simple WL Conversion)', 'Wavelength (nm)', 'Intensity (Counts)')

plt.show()
#-----------------------------------------------------------------------------------------------------------------------

#Part B: Use the method of linear least squares to determine a polynomial fit to these centroid data to derive the wavelength solution.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++








