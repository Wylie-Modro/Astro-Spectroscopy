import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls

LoadData = SpectraCls.LoadingData()
DataTools = SpectraCls.DataTools()

allIncandescentBulb = LoadData.LoadTextFromDirectoryIntoArray('incandescentBulb', '2017Lab2GroupB/incandescentBulb/')
allFluoroscentBulb = LoadData.LoadTextFromDirectoryIntoArray('fluoroscentBulb', '2017Lab2GroupB/fluoroscentBulb/')
allH2GDL = LoadData.LoadTextFromDirectoryIntoArray('H2GDL', '2017Lab2GroupB/H2GDL/')
allHeGDL = LoadData.LoadTextFromDirectoryIntoArray('HeGDL', '2017Lab2GroupB/HeGDL/')
allHgGDL = LoadData.LoadTextFromDirectoryIntoArray('HgGDL', '2017Lab2GroupB/HgGDL/')
allNeGDL = LoadData.LoadTextFromDirectoryIntoArray('NeGDL', '2017Lab2GroupB/NeGDL/')

averagedIncandescentBulb = DataTools.GetAveragedImage(allIncandescentBulb, 2048)
averagedFluoroscentBulb = DataTools.GetAveragedImage(allFluoroscentBulb, 2048)
averagedH2GDL = DataTools.GetAveragedImage(allH2GDL, 2048)
averagedHeGDL = DataTools.GetAveragedImage(allHeGDL, 2048)
averagedHgGDL = DataTools.GetAveragedImage(allHgGDL, 2048)
averagedNeGDL = DataTools.GetAveragedImage(allNeGDL, 2048)

fig = plt.figure()

ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

DataTools.SpectraPlot(averagedIncandescentBulb, ax1, 'Averaged Incandescent Bulb', 'Pixel (Number)', 'Intensity (Counts)')
DataTools.SpectraPlot(averagedFluoroscentBulb, ax2, 'Averaged Fluorescent Bulb', 'Pixel (Number)', 'Intensity (Counts)')
DataTools.SpectraPlot(averagedH2GDL, ax3, 'Averaged H2GDL', 'Pixel (Number)', 'Intensity (Counts)')
DataTools.SpectraPlot(averagedHeGDL, ax4, 'Averaged HeGDL', 'Pixel (Number)', 'Intensity (Counts)')
DataTools.SpectraPlot(averagedHgGDL, ax5, 'Averaged HgGDL', 'Pixel (Number)', 'Intensity (Counts)')
DataTools.SpectraPlot(averagedNeGDL, ax6, 'Averaged NeGDL', 'Pixel (Number)', 'Intensity (Counts)')

plt.show()



