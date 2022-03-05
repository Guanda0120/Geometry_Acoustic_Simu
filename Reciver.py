import numpy as np

class Reciver:
    
    def __init__ (self, location, radius=0.1):
        self.location = location
        self.radius = radius
    
    def checkIntersection(self, rayDirection, rayLocation):
        numerator = np.sum(np.multiply(rayDirection,(self.location-rayLocation)),axis=1)
        denominator = np.sum(np.multiply(rayDirection,rayDirection),axis=1)
        scaler_1 = np.divide(numerator, denominator)
        scaler_1 = scaler_1[:,np.newaxis]
        scaler = np.concatenate((scaler_1,scaler_1,scaler_1),axis=1)
        distance = np.linalg.norm((self.location-(rayLocation+np.multiply(rayDirection,scaler))),axis=1)
        slope = np.maximum(scaler_1,0)
        time = distance/343
        return distance, time
