import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls


LoadData = SpectraCls.LoadingData()
DataTools = SpectraCls.DataTools()
CentroidAl = SpectraCls.CentroidAlgorithm()

allHeGDL = LoadData.LoadTextFromDirectoryIntoArray('HeGDL', '2017Lab2GroupB/HeGDL/')
averagedHeGDL = DataTools.GetAveragedImage(allHeGDL , 2048)


    

DataTools.CompareAveragedToIndividualPlots('HeGDL', '2017Lab2GroupB/HeGDL/', 2048)
print(CentroidAl.GetEmissionLineExtremas(averagedHeGDL, 10., 800.) )


plt.show()

def GetCentroidPoisson(intensities, xmin, xmax):    
    variance, ISum = 0
    xMean = GetMeanOfIntensities(intensities, xmin, xmax)
    for x in range(xmin,xmax):
        ISum += intensities[x]
        variance += intensities[x] * ((float(x) - xMean)**2)
    errorInCentroid = variance/(ISum**2)
    return errorInCentroid
    

    