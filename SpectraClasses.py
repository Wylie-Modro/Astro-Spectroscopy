import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math as maths
import os
import pyfits

class LoadingData: 
    
    def LoadTextFromDirectoryIntoArray(self, commonStartOfFileName, directoryPath):
        listOfData = []

        print("Loading in images ...")

        for filename in os.listdir(directoryPath):
    
            if commonStartOfFileName in str(filename):
                data = np.loadtxt(directoryPath+filename, skiprows=17, comments = '>>', usecols=(0,1))
                listOfData.append(data)
            else:
                print('Found: ' + str(filename))
    
        return listOfData
            
    def LoadImageIntoVariable(self, nameOfFileWithDir):
        img = mpimg.imread(nameOfFileWithDir)
        img = 1.0*img #to make float
        return img
    
    def LoadDataFromDirectoryIntoArray(self, commonStartOfFileName, directoryPath):
        listOfData = []
        
        print("Loading in images ...")

        for filename in os.listdir(directoryPath):
    
            if commonStartOfFileName in str(filename):
                image = pyfits.open(directoryPath+filename)
                imagedata = image[0].data
                for i in imagedata:
                    xlength=len(i)
                    ylength=len(imagedata)
                listOfData.append(imagedata)
            else:
                print('Found: ' + str(filename))
                      
        return [listOfData, xlength, ylength]
    
class DataTools:
    
    @staticmethod
    def CreateDictOfSpectra(averagedData):
        dictOfSpectra= {}
        for entry in averagedData:
            dictOfSpectra[entry[0]] = entry[1]
        return dictOfSpectra
    
    
    @staticmethod
    def SpectraPlot(singleSpectra, Ax, title, xLabel, yLabel):
        Ax.plot(singleSpectra.T[0], singleSpectra.T[1]) 
        Ax.title.set_text(title)
        Ax.set_xlabel(xLabel)
        Ax.set_ylabel(yLabel)
    
    @staticmethod
    def CompareAveragedToIndividualPlots(commonStartOfFileName, directoryPath, resolution):
        LoadData = LoadingData()
        allFiles = LoadData.LoadTextFromDirectoryIntoArray(commonStartOfFileName, directoryPath)
        averagedSpectra = DataTools.GetAveragedImage(allFiles , resolution)
       
        fig = plt.figure()
        
        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(325)
        ax6 = fig.add_subplot(326)
        
        DataTools.SpectraPlot(allFiles[0], ax1, commonStartOfFileName + str(1), 'Pixel (Number)', 'Intensity (Counts)')
        DataTools.SpectraPlot(allFiles[1], ax2, commonStartOfFileName + str(2), 'Pixel (Number)', 'Intensity (Counts)')
        DataTools.SpectraPlot(allFiles[2], ax3, commonStartOfFileName + str(3), 'Pixel (Number)', 'Intensity (Counts)')
        DataTools.SpectraPlot(allFiles[3], ax4, commonStartOfFileName + str(4), 'Pixel (Number)', 'Intensity (Counts)')
        DataTools.SpectraPlot(allFiles[4], ax5, commonStartOfFileName + str(5), 'Pixel (Number)', 'Intensity (Counts)')
        
        DataTools.SpectraPlot(averagedSpectra, ax6, 'Averaged ' + str(commonStartOfFileName), 'Pixel (Number)', 'Intensity (Counts)')
    
    @staticmethod
    def GetAveragedImage(allSpectras, numOfPixels):
        total = [] #Create empty list of desired size, numOfPixels
        for i in range(numOfPixels):
            total.append(0.) #initializes array with numOfPixels entries of 0
            
        averagedSpectra = [] 
        for eachSpectra in allSpectras:
            for i in range(numOfPixels):
                total[i] += eachSpectra[i].T[1] #Summing values of allSpectras for each pixel
        for i in range(numOfPixels):
            #Dividing that total by number of spectras to get average
            total[i] = total[i]/len(allSpectras)
        #Sequence the pixel number with new average value and put into list
        for i in range(numOfPixels):
            averagedSpectra.append([float(i),  total[i]]) 
        return np.array(averagedSpectra)
    
    @staticmethod
    def NewGetAveragedData(allSpectras, xlength, ylength):
        summedList = []
        for y in range(ylength):
            summedList.append([])
            for x in range(xlength):
                summedList[y].append(0)
        summedMatrix = np.matrix(summedList)
        
        for eachSpectra in allSpectras:
            summedMatrix += np.matrix(eachSpectra)
        
        averageMatrix =  summedMatrix/len(allSpectras)
        return np.array(averageMatrix)
        
    @staticmethod
    def GetAveragedData(allSpectras, xlength, ylength):
        print('xlength:' + str(xlength))
        print('ylength:' + str(ylength))
        totalx = [] 
        totaly =[]
        
        for i in range(xlength):
            totalx.append(0.)
        print('len(totalx): ' + str(len(totalx)))
        for i in range(ylength):
            totaly.append(0.)
            
        averagedSpectra = [] 
        matrix = []
        for eachSpectra in allSpectras:
            print('len(eachSpectra): ' + str(len(eachSpectra)))
            #for j in range(ylength):
                #totalx[j] += eachSpectra[i]
            for entry in eachSpectra:
                row = []
                print('len(entry): ' + str(len(entry)))
                for i in range(xlength):
                    #matrix[entry[row[i]]] += entry[i]  
                    row[i] += entry[i]
                rowCopy = row.copy()
                matrix.append(rowCopy)
                del row
        print('matrix: ' + str(matrix))       
                
        for i in range(xlength):
            totalx[i] = totalx[i]/len(allSpectras)
        for j in range(ylength):
            totalx[j] = totaly[j]/len(allSpectras)
            
        for i in range(xlength):
            averagedSpectra.append([totalx[i]]) 
        for j in range(ylength):
            averagedSpectra.append([totaly[j]])
            
        return np.array(averagedSpectra)
    '''
        iterDimOfMainMatrix = range(orderOfApprox + 2)
        mainMatrix = []
    
        for col in iterDimOfMainMatrix:
            rowMatrix = []
            for row in iterDimOfMainMatrix:
                rowMatrix.append(np.sum(i**(row+col-2)))
            rowMatrixCopy = rowMatrix.copy()
            mainMatrix.append(rowMatrixCopy)
            #print('theMatrix: ' + str(theMatrix))
            del rowMatrix
        return mainMatrix
    '''
    def GetMean(self, img):
        x = img.flatten()
        mean1 = np.sum(x)/(np.size(x))
        return mean1
    
    def GetArrayOfMeans(self, arrayOfImages):
        arrayOfMeans = []
        for img in arrayOfImages:
            x = img.flatten()
            mean1 = np.sum(x)/(np.size(x))
            arrayOfMeans.append(mean1)
        return arrayOfMeans
    
    
    def GetMOM(self, arrayOfImages):
        meansSumed = 0
        arrayOfMeans = []
        for img in arrayOfImages:
            x = img.flatten()
            mean1 = np.sum(x)/(np.size(x)) 
            meansSumed += mean1
            arrayOfMeans.append(mean1)
        meanOfMeans = meansSumed / len(arrayOfMeans)
        return meanOfMeans 
    
    def GetStandardDeviation(self, img):
        x = img.flatten()
        mean1 = np.sum(x)/(np.size(x))
        m2 = np.sum(x*x)/(np.size(x))
        standardDeviation = np.sqrt(m2 - mean1*mean1)
        return standardDeviation
    
    def GetArrayOfStandardDeviation(self, arrayOfImages):
        arrayOfStd = []
        for img in arrayOfImages:
            x = img.flatten()
            mean1 = np.sum(x)/(np.size(x))
            m2 = np.sum(x*x)/(np.size(x))
            standardDeviation = np.sqrt(m2 - mean1*mean1)
            arrayOfStd.append(standardDeviation)
        return arrayOfStd
    
    def GetSDOM(self, arrayOfImages):
        stdSumed = 0
        arrayOfStd = []
        for img in arrayOfImages:
            x = img.flatten()
            mean1 = np.sum(x)/(np.size(x))
            m2 = np.sum(x*x)/(np.size(x))
            standardDeviation = np.sqrt(m2 - mean1*mean1)
            stdSumed += standardDeviation
            arrayOfStd.append(standardDeviation)
        stdOfMeans = stdSumed / maths.sqrt(len(arrayOfStd))
        return stdOfMeans 
    
    def ShowHistogram(self, arrayForYAxis, arrayforXAxis):
        hist = np.array([np.where(arrayForYAxis == i)[0].size for i in arrayforXAxis])
        plt.plot(arrayforXAxis, hist, drawstyle='steps-mid')
        plt.title('Pixel number vs. No. of potons')
        plt.xlabel('Pixel number', fontsize=15)
        plt.ylabel('No. of photons', fontsize=15)
        plt.show()
        
    def SetHistogram(self, arrayForYAxis, arrayforXAxis):
        hist = np.array([np.where(arrayForYAxis == i)[0].size for i in arrayforXAxis])
        plt.plot(arrayforXAxis, hist, drawstyle='steps-mid')
        plt.title('Pixel number vs. No. of photons')
        plt.xlabel('Pixel number', fontsize=15)
        plt.ylabel('No. of photons', fontsize=15)
        
    def AverageImages(self, arrayOfImages):
        print("Start Sublisting")
        #x = len([item for sublist in arrayOfImages for item in sublist])
        #x = len(arrayOfImages.flatten())
        #print(x)
        averagedImageArray = [0]*480
        print("averaging images")
        for img in arrayOfImages:
            i=0
            #print(len(img))
            for p in img:
                averagedImageArray[i]+=p
                i+=1
        return averagedImageArray



class CentroidAlgorithm(): 
      
    @staticmethod
    def GetMeanOfIntensities(DictOfSpectra, xmin, xmax):
        XISum = 0.
        ISum = 0.
        for x in np.arange(xmin,xmax):
            XISum += float(x)*DictOfSpectra[x]
            ISum += DictOfSpectra[x]
        xMean = XISum/ISum
        return np.round(xMean, 0) 
    

    def GetCentroidPoisson(self, intensities, xmin, xmax):    
        variance, ISum = 0
        xMean = self.GetMeanOfIntensities(intensities, xmin, xmax)
        for x in range(xmin,xmax):
            ISum += intensities[x]
            variance += intensities[x] * ((float(x) - xMean)**2)
        errorInCentroid = variance/(ISum**2)
        return errorInCentroid
    
    
    @staticmethod  
    def LocatePeakRanges(averagedSpectra, gap, threashhold):
        peakRanges=[]
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
                #print('Peak at:' + str(key + gap/2) +'!!!')
                peakRanges.extend([key, key + gap/2, key + gap])
            else:
                #print('No peak at:' + str(key + gap/2)+' intensitiesDict[key+gap]: ' + str(intensitiesDict[key+gap]))
                pass
        return peakRanges

    @staticmethod 
    def IsolatePeaks(peakRanges):
        peakSet = set()
        masterList = []
        tempList = []
        for peak1 in peakRanges:
            if np.abs(peak1 - peakRanges[peakRanges.index(peak1)+1]) < 10:
                peakSet.add(peak1)
                peakSet.add(peakRanges[peakRanges.index(peak1)+1])
            else: 
                if len(peakSet) > 0:
                    for peak in peakSet:
                        tempList.append(peak)
                    copyOftempList = list(tempList)
                    masterList.append(copyOftempList)
                    del tempList[:]
                    peakSet.clear()
            
            if peakRanges[peakRanges.index(peak1)+1] == peakRanges[-1]:
                for peak in peakSet:
                    tempList.append(peak)
                masterList.append(tempList)
                return masterList
    
    
    def GetEmissionLineExtremas(self, averagedSpectra, gap, threashhold):
        peakRanges = self.LocatePeakRanges(averagedSpectra, gap, threashhold)
        masterList = self.IsolatePeaks(peakRanges)
        listOfTrailingPoints = []
        for peakColl in masterList:
            listOfTrailingPoints.append([min(peakColl),max(peakColl)])
        print('Number of Emission lines: ' + str(len(listOfTrailingPoints)))
        #print('listOfTrailingPoints: ' + str(listOfTrailingPoints))
        return listOfTrailingPoints
    
    @staticmethod
    def SimplePixlToWLMapping(pixelValue):
        a0 = 339.672754
        a1 = 0.38194
        a2 = -0.0000185296
        a3 = -0.00000000197619
        WL = a0+a1*(pixelValue)+a2*(pixelValue**2)+a3*(pixelValue**3)
        return WL
            
            
class Distributions:
    
    def PoissonDistribution(self,n,theMean):
        n = np.float64(n)
        logStirlingApprox = n*(maths.log(n)) - n + (1/2)*(maths.log(2*(maths.pi)*n))
        logPoissonDistribution = n*(maths.log(theMean))-(theMean)-logStirlingApprox
        return maths.exp(logPoissonDistribution)

    def GaussianDistribution(self, n, theMean, standardDev):
        n = np.float64(n)
        firstTerm = -(maths.log(standardDev*(maths.sqrt(2.0*(maths.pi)))))
        logGaussianDistribution = firstTerm - (1/2)*(((n - theMean)/standardDev)**2.0)
        return maths.exp(logGaussianDistribution)
    