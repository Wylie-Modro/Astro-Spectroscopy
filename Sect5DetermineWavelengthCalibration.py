import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls

'''
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
'''
#Part B: Use the method of linear least squares to determine a polynomial fit to these centroid data to derive the wavelength solution.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#___Least Squares Fitting___

N = 4 #Number of data points
a0 = 1
a1 = 1
a2 = 1
a3 = 1

i = np.arange(N, dtype= float) #i=[1,2,3,4]
WL = []

#Create the Matrices
ma = np.array([[N, np.sum(i), np.sum(i**2), np.sum(i**3)], [np.sum(i), np.sum(i**2), np.sum(i**3), np.sum(i**4)], 
               [np.sum(i**2), np.sum(i**3), np.sum(i**4), np.sum(i**5)], [np.sum(i**3), np.sum(i**4), np.sum(i**5), np.sum(i**6)]])
mc = np.array([[WL], [WL*np.sum(i)], [WL*np.sum(i**2)], [WL*np.sum(i**3)]])
print('mc: ' + str(mc))
#Compute the gradient and intercept
invMa = np.linalg.inv(ma)
print('invMa: ' + str(invMa))
print('---------------------------------------------------------------')
print('Test matrix inversion gives identity: '+str(np.dot(invMa,ma)))
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('mc: ' + str(mc))
constantsResult = np.dot(invMa, mc)


#Overplot the best fit
a0Fit = constantsResult[0,0]
a1Fit = constantsResult[1,0]
a2Fit = constantsResult[2,0]
a3Fit = constantsResult[3,0]
print('a0Fit: ' + str(a0Fit))
print('a1Fit: ' + str(a1Fit))
print('a2Fit: ' + str(a2Fit))
print('a3Fit: ' + str(a3Fit))

'''
plt.plot(x, mFit*x + cFit)
plt.axis('scaled')
plt.text(5, 15, 'm = {:.3f}\nc = {:.3f}' .format(mFit, cFit))


#Deriving Standard Deviation 
stDev = (1/(N-2))*(np.sum((y - (mFit*x + cFit))**2))
print('stDev: ' + str(stDev))

uncertaintyM = (N*(stDev**2))/((N*np.sum(x**2))-((np.sum(np.sum(x)))**2))
print('uncertaintyM: ' + str(uncertaintyM))

uncertaintyC = ((np.sum(x**2))*(stDev**2))/((N*np.sum(x**2))-((np.sum(np.sum(x)))**2))
print('uncertaintyC: ' + str(uncertaintyC))
'''









'''
#___Least Squares Fitting___

N = 20 #Number of data points
m = 1.0
c = 0.0

x = np.arange(N, dtype= float)
print('xType: ' + str(type(x)))
y = m * x + c 

#Generate Gaussian errors 
sigma = 1.0 #Measurement error
np.random.seed(1)
errors = sigma*np.random.rand(N)
ye = y + errors

plt.plot(x, ye, 'o', label='data')
plt.xlabel('x')
plt.ylabel('y')

#Create the Matrices
ma = np.array([[np.sum(x**2), np.sum(x)], [np.sum(x), N]])
mc = np.array([[np.sum(x*ye)],[np.sum(ye)]])

#Compute the gradient and intercept
invMa = np.linalg.inv(ma)
print('Test matrix inversion gives identity: '+str(np.dot(invMa,ma)))
mcResult = np.dot(invMa, mc)


#Overplot the best fit
mFit = mcResult[0,0]
cFit = mcResult[1,0]
print('mFit: ' + str(mFit))
print('cFit: ' + str(cFit))

plt.plot(x, mFit*x + cFit)
plt.axis('scaled')
plt.text(5, 15, 'm = {:.3f}\nc = {:.3f}' .format(mFit, cFit))


#Deriving Standard Deviation 
stDev = (1/(N-2))*(np.sum((y - (mFit*x + cFit))**2))
print('stDev: ' + str(stDev))

uncertaintyM = (N*(stDev**2))/((N*np.sum(x**2))-((np.sum(np.sum(x)))**2))
print('uncertaintyM: ' + str(uncertaintyM))

uncertaintyC = ((np.sum(x**2))*(stDev**2))/((N*np.sum(x**2))-((np.sum(np.sum(x)))**2))
print('uncertaintyC: ' + str(uncertaintyC))

plt.show()
'''
