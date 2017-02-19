import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls


LoadData = SpectraCls.LoadingData()
DataTools = SpectraCls.DataTools()

allHeGDL = LoadData.LoadTextFromDirectoryIntoArray('HeGDL', '2017Lab2GroupB/HeGDL/')
averagedHeGDL = DataTools.GetAveragedImage(allHeGDL , 2048)

def CompareAveragedToIndividualPlots(commonStartOfFileName, directoryPath, resolution):
    
    allFiles = LoadData.LoadTextFromDirectoryIntoArray(commonStartOfFileName, directoryPath)
    averagedSpectra = DataTools.GetAveragedImage(allFiles , resolution)
   
    fig = plt.figure()
    
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    
    for file in allFiles:
        i=1
        #DataTools.SpectraPlot(file, 'ax'+str(i), str(commonStartOfFileName + i), 'Pixel (Number)', 'Intensity (Counts)')
        i+=1

    #DataTools.SpectraPlot(averagedH2GDL, ax1, 'Averaged H2GDL', 'Pixel (Number)', 'Intensity (Counts)')
    DataTools.SpectraPlot(averagedSpectra, ax6, 'Averaged HeGDL', 'Pixel (Number)', 'Intensity (Counts)')
    


def PredictEmissionLines(averagedSpectra, gap, threashhold):
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    peaks=[]
    intensitiesDict = {}
    for entry in averagedSpectra:
        intensitiesDict[entry.T[0]] = entry.T[1]
        
    for key in intensitiesDict.keys():
        if key+gap >= 2047.0:
            gap = float(int(gap/2))
        startValue = intensitiesDict[key]
        midValue = intensitiesDict[key+int(gap/2)]
        endValue = intensitiesDict[key+gap]
        if midValue - startValue > threashhold and midValue - endValue > threashhold :
            print('Peak at:' + str(key + gap/2) +'!!!')
            peaks.append(key + gap/2)
        else:
            #print('No peak at:' + str(key + gap/2)+' intensitiesDict[key+gap]: ' + str(intensitiesDict[key+gap]))
            pass
    #++++++++++++++++++++++++++++++++++++++++++++++++++
    #--------------------------------------------------
    peakCollection = set()
    masterList = []
    tempList= []
    for peak1 in peaks:
        if np.abs(peak1 - peaks[peaks.index(peak1)+1]) < 10:
              peakCollection.add(peak1)
              peakCollection.add(peaks[peaks.index(peak1)+1])
        else: 
            print("peakCollection in else: " + str(peakCollection))
            if len(peakCollection) > 0:
                for peak in peakCollection:
                    tempList.append(peak)
                copyOftempList = list(tempList)
                print('copyOftempList: ' + str(copyOftempList))
                masterList.append(copyOftempList)
                tempList.clear()
                peakCollection.clear()
        
        if peaks[peaks.index(peak1)+1] == peaks[-1]:
            for peak in peakCollection:
                tempList.append(peak)
            masterList.append(tempList)
            print(masterList)
            break
            #return masterList
    #--------------------------------------------------
    listOftrailingPoints = []
    for peakColl in masterList:
        listOftrailingPoints.append([max(peakColl),min(peakColl)])
    return listOftrailingPoints



CompareAveragedToIndividualPlots('HeGDL', '2017Lab2GroupB/HeGDL/', 2048)
print(PredictEmissionLines(averagedHeGDL, 10., 1000.) )


plt.show()

def GetCentroidPoisson(intensities, xmin, xmax):    
    variance, ISum = 0
    xMean = GetMeanOfIntensities(intensities, xmin, xmax)
    for x in range(xmin,xmax):
        ISum += intensities[x]
        variance += intensities[x] * ((float(x) - xMean)**2)
    errorInCentroid = variance/(ISum**2)
    return errorInCentroid
    

    