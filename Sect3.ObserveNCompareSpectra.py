import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls


LoadData = SpectraCls.LoadingData()
DataTools = SpectraCls.DataTools()

allIncandescentBulb = LoadData.LoadTextFromDirectoryIntoArray('incandescentBulb', '2017Lab2GroupB/incandescentBulb/')
allFluoroscentBulb = LoadData.LoadTextFromDirectoryIntoArray('fluoroscentBulb', '2017Lab2GroupB/fluoroscentBulb/')
allH2GDL = LoadData.LoadTextFromDirectoryIntoArray('H2GDL', '2017Lab2GroupB/H2GDL/')

averagedIncandescentBulb = DataTools.GetAveragedImage(allIncandescentBulb, 2048)
averagedFluoroscentBulb = DataTools.GetAveragedImage(allFluoroscentBulb , 2048)




fig = plt.figure()

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

DataTools.SpectraPlot(averagedIncandescentBulb, ax1, 'AveragedIncandescentBulb', 'Pixel (Number)', 'Intensity (Counts)')
DataTools.SpectraPlot(averagedIncandescentBulb, ax2, 'AveragedFluoroscentBulb', 'Pixel (Number)', 'Intensity (Counts)')



plt.show()



