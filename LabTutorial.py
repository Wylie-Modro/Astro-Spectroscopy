import numpy as np
import matplotlib.pyplot as plt
import SpectraClasses as SpectraCls


LoadData = SpectraCls.LoadingData()
arrayOfData = LoadData.LoadTextFromDirectoryIntoArray(' light', 'lights/')
print(arrayOfData)


data = np.loadtxt('lights\light00000.txt', skiprows=17, comments = '>>', usecols=(0,1))
print (data)

plt.plot(data.T[0], data.T[1])
plt.show()


np.zeros((2048, 100))



#names = np.loadtxt('filenames.txt', dtype=str)
data_all = np.zeros((2048,100))

counter = 0
for line in data:
    temp = np.loadtxt(line[0,14], skiprows=17, comments='>>', usecols=(0,1)).T[1]
    data_all[:counter] =  temp
    counter+=1
    
spec_1d = np.mean(data_all, axis=1)
emission_line = spec_1d[150:180]

    
    
plt.imshow(data_all, origin='lower', interpolation ='nearest')
plt.show()


