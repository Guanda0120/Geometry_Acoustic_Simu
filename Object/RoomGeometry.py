import numpy as np

class RoomGeometry:
    
    def __init__(self, planeMatrix, absorption_Dictionary, scater_Dictionary, material_ID):
        self.planeMatrix = planeMatrix
        self.absorption_Dictionary = absorption_Dictionary
        self.scater_Dictionary = scater_Dictionary
        self.material_ID = material_ID
    
    '''
    planeAlpha refers to absorption coefficient from 125Hz to 4kHz Octave Band
        shape n by 6 dimensional 
    '''
    
    '''check the dimension of two tensor'''
    
    def check4PointInPlane(self):
        det_test_vec = np.ones((self.planeMatrix.shape[0],self.planeMatrix.shape[1],1))
        det_plane = np.linalg.det(np.concatenate((self.planeMatrix,det_test_vec),axis = 2))
        if np.sum(det_plane) == 0:
            return True
        else:
            return False
    
    def planeNormalDirec(self):
        centre_point = np.sum(self.planeMatrix, axis = 1)/self.planeMatrix.shape[1]
        normal_direc = np.array([])
        for i in range(self.planeMatrix.shape[0]):
            temp_1 = self.planeMatrix[i,1,...]
            temp_2 = self.planeMatrix[i,2,...]
            temp_3 = self.planeMatrix[i,3,...]
            temp_p = np.cross((temp_3-temp_2),(temp_1-temp_2))
            temp_p = temp_p/np.linalg.norm(temp_p)
            normal_direc = np.concatenate((normal_direc,temp_p),axis = 0)
        normal_direc = np.reshape(normal_direc,(self.planeMatrix.shape[0],3))
        panel_d_temp = self.planeMatrix[:,0,:]
        panel_d = -np.dot(normal_direc,panel_d_temp.T)[:,0]
        return centre_point,normal_direc,panel_d
    
    '''
        centre_point is the center point
        normal_direc is the normal direction of the panel
        panel_d = is the analitical [Ax+By+Cz+D=0]~D
    '''

    def absorption_List(self):
        absorption_List = []
        
        for i in range(self.material_ID.shape[0]):
            index = material_ID[i]-1
            absorption_List.append(self.absorption_Dictionary[index])
        
        absorption_List = np.array(absorption_List)
        absorption_List.reshape((-1,6))
        return absorption_List
    
    def scater_List(self):
        scatter_List = []
        
        for i in range(self.material_ID.shape[0]):
            index = material_ID[i]-1
            scatter_List.append(self.scater_Dictionary[index])
        
        scatter_List = np.array(scatter_List)
        scatter_List.reshape((-1,6))
        return scatter_List
