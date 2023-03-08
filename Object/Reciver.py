import numpy as np

class Reciver:
    
    def __init__ (self, location, radius):
        self.location = location
        self.radius = radius
        
    def reshapeTensor(self,rayDirection, rayLocation, rayStartT, rayPreasure):
        
        # trans into ndarray
        
        rayDirection = np.array(rayDirection)
        rayLocation = np.array(rayLocation)
        rayStartT = np.array(rayStartT)
        rayPreasure = np.array(rayPreasure)
        
        # reshape the array
        rayDirection=rayDirection.reshape(-1,rayDirection.shape[2])
        rayLocation=rayLocation.reshape(-1,rayLocation.shape[2])
        rayStartT=rayStartT.reshape(-1,rayStartT.shape[2])
        rayPreasure=rayPreasure.reshape(-1,rayPreasure.shape[2])
        
        return rayDirection,rayLocation,rayStartT,rayPreasure
    
    def checkIntersection(self, rayDirection, rayLocation, rayStartT, rayPreasure):
        
        numerator = np.sum(np.multiply(rayDirection,(self.location-rayLocation)),axis=1)
        denominator = np.sum(np.multiply(rayDirection,rayDirection),axis=1)
        scaler_1 = np.divide(numerator, denominator)
        scaler_1 = scaler_1[:,np.newaxis]
        scaler = np.concatenate((scaler_1,scaler_1,scaler_1),axis=1)
        distance = np.linalg.norm((self.location-(rayLocation+np.multiply(rayDirection,scaler))),axis=1)
        
        boolean_cond = np.maximum(scaler_1,0)
        boolean_cond = np.where(boolean_cond<=0,boolean_cond,1)
        boolean_cond = boolean_cond[:,0]
        
        distance = np.multiply(boolean_cond,distance)
        
        dis_boolean = np.where(distance<self.radius,distance,0)
        dis_boolean = np.where(dis_boolean==0,dis_boolean,1)
        
        int_rayD=[]
        int_rayS=[]
        int_startT=[]
        start_preasure=[]
        int_scaler=[]
        
        for i in range(dis_boolean.shape[0]):
            if dis_boolean[i]==1:
                int_rayD.append(rayDirection[i])
                int_rayS.append(rayLocation[i])
                int_startT.append(rayStartT[i])
                start_preasure.append(rayPreasure[i])
                int_scaler.append(scaler_1[i])
        

        int_rayD = np.array(int_rayD)
        int_rayS = np.array(int_rayS)
        int_startT = np.array(int_startT)
        start_preasure = np.array(start_preasure)
        int_scaler = np.array(int_scaler)
        
        return int_rayD, int_rayS, int_startT, start_preasure, int_scaler
    
    def propagateReduce(self,rayD,rayS,startT,startPreasure,intScaler):
        
        distance = np.linalg.norm((rayS+np.multiply(rayD,intScaler)),axis=1)
        distance = np.tile(distance, (6,1))
        distance = np.transpose(distance)
        duration = distance/343
        end_time = startT+duration
        
        num_ray = end_time.shape[0]
        p_0 = 20e-6
        octave_band = np.array([125,250,500,1000,2000,4000])
        octave_band = np.tile(octave_band,(num_ray,1))
        attenuation = np.array([1.5625e-06,6.2500e-06,2.5000e-05,1.0000e-04,4.0000e-04,1.6000e-03])
        attenuation = np.tile(attenuation,(num_ray,1))
    
        # var[end_preasure] is the sound preasure without attenuation
        upper = (-1j)*(np.pi*2*octave_band*distance/343)
        end_preasure = np.abs(np.multiply(np.abs(startPreasure),np.exp(upper)))
        
        # var[end_spl_a] is the sound preasure level of the end preasure with attenuation
        # var[end_preasure_a] is the end preasure with attenuation
        end_spl_a = 20*np.log10(end_preasure/p_0)-np.multiply(attenuation,distance)
   
        return end_time,end_spl_a
