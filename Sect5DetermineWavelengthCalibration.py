def SimplePixlToWLMapping(pixelValue):
    a0 = 339.672754
    a1 = 0.38194
    a2 = -0.0000185296
    a3 = -0.00000000197619
    WL = a0+a1*(pixelValue)+a2*(pixelValue**2)+a3*(pixelValue**3)
    return WL
    
    
print (SimplePixlToWLMapping(2048.0))
    
