import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls


LoadData = SpectraCls.LoadingData()
DataTools = SpectraCls.DataTools()

allIncandescentBulb = LoadData.LoadTextFromDirectoryIntoArray('incandescentBulb', '2017Lab2GroupB/incandescentBulb/')
allFluoroscentBulb = LoadData.LoadTextFromDirectoryIntoArray('fluoroscentBulb', '2017Lab2GroupB/fluoroscentBulb/')
allH2GDL = LoadData.LoadTextFromDirectoryIntoArray('H2GDL', '2017Lab2GroupB/H2GDL/')

AveragedIncandescentBulb = DataTools.GetAveragedImage(allSpectrasIncandescentBulb, 2048)


DataTools.SpectraPlot(AveragedIncandescentBulb, plt, 'AveragedIncandescentBulb', 'Pixel (Number)', 'Intensity (Counts)')

'''
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

DataTools.SpectraPlot(singleSpectra, ax1, 'incandescentBulb1', 'Pixel (Number)', 'Intensity (Counts)')

DataTools.SpectraPlot(singleSpectra2, ax2, 'incandescentBulb2', 'Pixel (Number)', 'Intensity (Counts)')
'''

fig = plt.figure()

plt.show()



