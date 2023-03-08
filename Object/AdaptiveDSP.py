import numpy as np

class AdaptiveDSP():
    
    def __init__(self):
       
        return 
    
    def linearRegresion(self,response, time):
        
        # Least Square Method
        
        assert response.shape[0]==time.shape[0]
        
        bias = np.ones((response.shape[0],1))
        time = time[:,np.newaxis]
        time = np.hstack([time,bias])
        
        slope_1 = np.linalg.inv((np.dot(np.transpose(time),time)))
        slope_2 = np.dot(np.transpose(time),response)
        slope = np.dot(slope_1,slope_2)
        slope = slope[:,np.newaxis]
        
        fillter_response = np.dot(time,slope)
        
        rt = -60/slope[0,0]
        
        return fillter_response,slope,rt
    
    def backIntegral(self,response,time,init_Spl):
        
        # This algorthim refer 'M. R. Schroeder' publish in 1965 Bell Lab
        # New method of measuring reverberation time. J Acoust Soc Am 1965; 37: 409.
        
        energy_res = 10**(response/20)*(20e-6)
        energy_res = energy_res**2
        
        init_energy = (10**(init_Spl/20)*(20e-6))**2
        
        seq_len = response.shape[0]
        energy_res = energy_res[:,np.newaxis]

        conv_m = np.tri(seq_len,seq_len,0)*init_energy
        filltered_res = np.dot(energy_res.T,conv_m)
        filltered_res = np.sqrt(filltered_res)
        filltered_res = np.log10(filltered_res/20e-6)*20
        filltered_res = filltered_res.reshape(-1)
    
        start_spl = filltered_res[0]
        t_5 = filltered_res-start_spl
        t_25 = filltered_res-25
        r_t = (time[(t_5<=-25).argmax(axis=0)]-time[(t_5<=-5).argmax(axis=0)])*3

        
        return filltered_res,r_t
    
