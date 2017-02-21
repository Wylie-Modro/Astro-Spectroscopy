import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls

#Initialize classes
LoadData = SpectraCls.LoadingData()
DataTools = SpectraCls.DataTools()
CentroidAl = SpectraCls.CentroidAlgorithm()

#Load Data and get averaged-spectra
allHeGDL = LoadData.LoadTextFromDirectoryIntoArray('HeGDL', '2017Lab2GroupB/HeGDL/')
averagedHeGDL = DataTools.GetAveragedImage(allHeGDL , 2048)

#Setup plot of averaged spectra
fig = plt.figure()     
ax1 = fig.add_subplot(111)
DataTools.SpectraPlot(averagedHeGDL, ax1, 'Averaged HeGDL', 'Pixel (Number)', 'Intensity (Counts)')

#Use the automatic centriod algorithm to get all emission line pixel positions
HeGDLDict = DataTools.CreateDictOfSpectra(averagedHeGDL)
allEmissionLineRangeExtremas = CentroidAl.GetEmissionLineExtremas(averagedHeGDL, 10., 800.)

allEmissionLinePositions = []
for pairOfExtremas in allEmissionLineRangeExtremas: 
    allEmissionLinePositions.append(CentroidAl.GetMeanOfIntensities(HeGDLDict, pairOfExtremas[0], pairOfExtremas[1]))
print('allEmissionLinePositions: ' + str(allEmissionLinePositions))  


#plt.show()


    

    