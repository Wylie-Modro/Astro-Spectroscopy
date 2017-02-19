import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math as maths
import os


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
    

class DataTools:
    
    @staticmethod
    def SpectraPlot(singleSpectra, pltOrAx, title, xLabel, yLabel):
        pltOrAx.plot(singleSpectra.T[0], singleSpectra.T[1]) 
        pltOrAx.title.set_text(title)
        pltOrAx.set_xlabel(xLabel)
        pltOrAx.set_ylabel(yLabel)
        
    @staticmethod
    def GetAveragedImage(allSpectras, numOfPixels):
        total = [] #Create empty list of desired size,numOfPixels
        for i in range(numOfPixels):
            total.append(0.)
            
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
    def GetMeanOfIntensities(intensities, xmin, xmax):
        XIsum, ISum = 0
        for x in range(xmin,xmax):
            XIsum += float(x)*intensities[x]
            ISum += intensities[x]
        xMean = XIsum/ISum
        return xMean   

        
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
    