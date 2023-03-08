import numpy as np

class Simulation():
    
    def __init__(self, normal_Direction, plane_D, acoustic_Ray, acoustic_Loction, acoustic_Alpha, acoustic_Scatter):
        self.normal_Direction = normal_Direction
        self.plane_D = plane_D
        self.acoustic_Ray = acoustic_Ray
        self.acoustic_Loction = acoustic_Loction
        self.acoustic_Alpha = acoustic_Alpha
        self.acoustic_Scatter = acoustic_Scatter
        
    def spl2Preasure(self):
        return 10**(self.SPL/20)/(20*(10**(-6)))
    
    def preasure2SPL(self, presure_Array):
        spl = np.log10(presure_Array/20e-6)*20
        return spl
    
    def randomGenerateOrthognal(self, normal_Vector):
        
        orthognal_vector = np.random.random([3])
        if normal_Vector[2]!=0:
            orthognal_vector[2] = (0-(orthognal_vector[0]*normal_Vector[0]+orthognal_vector[1]*normal_Vector[1]))/normal_Vector[2]
        elif normal_Vector[1]!=0:
            orthognal_vector[1] = (0-(orthognal_vector[0]*normal_Vector[0]+orthognal_vector[2]*normal_Vector[2]))/normal_Vector[1]
        else:
            orthognal_vector[0] = (0-(orthognal_vector[1]*normal_Vector[1]+orthognal_vector[2]*normal_Vector[2]))/normal_Vector[0]
        
        return orthognal_vector
    
    
    def pointInFinitePlane(self,orthognal_Vector,speci_Point,temp_Plane):
        
        '''follow algorithm from this web'''
        '''https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
        https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect?newreg=6fd1d1e602a847c2b9959dca99f92963'''
        
        assert orthognal_Vector.shape[0]==3
        assert speci_Point.shape[0]==3
        
        temp_Plane_1 = np.vstack([temp_Plane[-1,:],temp_Plane[0:-1,:]])
        side_Vector = temp_Plane-temp_Plane_1
        cross_time = 0
        
        for i in range(side_Vector.shape[0]):
            temp_Side_Vector = side_Vector[i]
            edge_point = temp_Plane_1[i]
            std_vector = np.array([1,1,1])
            
            cross_res_1 = np.linalg.det(np.vstack([std_vector,orthognal_Vector,temp_Side_Vector]))
            cross_res_2 = np.linalg.det(np.vstack([std_vector,temp_Side_Vector,orthognal_Vector]))
            cross_res_1a = np.linalg.det(np.vstack([std_vector,(edge_point-speci_Point),temp_Side_Vector]))
            cross_res_2a = np.linalg.det(np.vstack([std_vector,(speci_Point-edge_point),orthognal_Vector]))

            if cross_res_1 == 0:
                cross_state = 0
            else:
                coeff_1 = cross_res_1a/cross_res_1
                coeff_2 = cross_res_2a/cross_res_2
            
                if (coeff_2>=0)&(coeff_2<=1)&(coeff_1>=0):
                    cross_state = 1
                else:
                    cross_state = 0
                cross_time+=cross_state
        
        if (cross_time%2==1):
            in_state = 1
        else:
            in_state = 0
        
        return in_state
    
    "there should be some problem"
    
    def vectorInfinitePlaneIntersection(self,plane_Dirc,acoustic_Dirc,acoustic_Sour):
        
        geometry = self.plane_D
        planeIntersectionP = []
        coef = []
        
        for i in range(self.plane_D.shape[0]):
            temp_source_point = np.tile(acoustic_Sour,(4,1)) # 4 is for formal 4 sides plane, it should be mod
            temp_normalD = plane_Dirc[i,:]
            prod1 = np.dot((temp_source_point-geometry[i,:,:]),temp_normalD)
            prod2 = np.dot(acoustic_Dirc,temp_normalD.T)
            prod3 = 0-np.divide(prod1,prod2)
            speci_point = temp_source_point+np.multiply(np.tile(prod3,(3,1)).T,np.tile(acoustic_Dirc,(4,1)))
            speci_point = speci_point[0,:]
        
            planeIntersectionP.append(speci_point)
            coef.append(prod3[0])
                
        planeIntersectionP = np.array(planeIntersectionP)
        coef = np.array(coef)

        return planeIntersectionP,coef
            
    
    def checkIntersection3D(self,plane_Dirc,acoustic_Dirc,acoustic_Sour):
        
        intersection_point = []
        plane_index = []
        ray_index = []
        
        for i in range(acoustic_Dirc.shape[0]):
            
            infinite_intersection,coef = self.vectorInfinitePlaneIntersection(plane_Dirc,acoustic_Dirc[i],acoustic_Sour[i])
            plane = self.plane_D
            normal_Direc = self.normal_Direction
            
            for k in range(plane.shape[0]):
                if coef[k]>0:
                    temp_plane = plane[k]
                    temp_intersection_point = infinite_intersection[k]
                    temp_normal_direc = normal_Direc[k]
                
                    orthognal_Vector = self.randomGenerateOrthognal(temp_normal_direc)
                    cross_state = self.pointInFinitePlane(orthognal_Vector,temp_intersection_point,temp_plane)
                
                    lenght_vec = np.linalg.norm(temp_intersection_point-acoustic_Sour[i])
                    if cross_state == 1 and lenght_vec>1e-14:
                        intersection_point.append(temp_intersection_point)
                        plane_index.append(k)
                        ray_index.append(i)
        intersection_point = np.array(intersection_point)
        # assert intersection_point.shape==acoustic_Sour.shape
        
        return plane_index,ray_index,intersection_point

    def propagateReduce(self,start_Point,end_Point,start_Preasure,start_Time):
        '''
           start_Point n*3
           end_Point   n*3
           start_Point.shape == end_Point.shape
           start_Preasure  n*6
           start_Time      n*1
        '''
        assert start_Point.shape[0] == end_Point.shape[0]
        assert start_Point.shape[0] == start_Preasure.shape[0]
    
        num_ray = start_Point.shape[0]
        p_0 = 20e-6
        octave_band = np.array([125,250,500,1000,2000,4000])
        octave_band = np.tile(octave_band,(num_ray,1))
        attenuation = np.array([1.5625e-06,6.2500e-06,2.5000e-05,1.0000e-04,4.0000e-04,1.6000e-03])
        attenuation = np.tile(attenuation,(num_ray,1))
        
        distance = np.reshape(np.linalg.norm((start_Point-end_Point),axis=1),(num_ray,1))
        # var[duration] is the propagate time
        duration = distance/343
        end_time = start_Time+duration
        distance = np.tile(distance,(1,6))
    
        # var[end_preasure] is the sound preasure without attenuation
        upper = (-1j)*(np.pi*2*octave_band*distance/343)
        end_preasure = np.abs(np.multiply(np.abs(start_Preasure),np.exp(upper)))
        
        # var[end_spl_a] is the sound preasure level of the end preasure with attenuation
        # var[end_preasure_a] is the end preasure with attenuation
        end_spl_a = 20*np.log10(end_preasure/p_0)-np.multiply(attenuation,distance)
        end_preasure_a = (10**(end_spl_a/20))*p_0
    
        return end_preasure_a,end_time
    
    def reflection(self,alpha_List,scatter_List,plane_Index,acoustic_Dirc,normal_Dirc,insert_Preasure):
        
        '''
            alpha_List is the list match the material absorption coefficient of the plane
            scatter_List is the list match the material scattering coefficient of the plane
            plane_Index is the list match the
        '''
        
        alpha_list = []
        scatter_list = []
        normal_dirc_list = []
        
        for i in range(acoustic_Dirc.shape[0]):
            alpha_list.append(alpha_List[plane_Index[i]])
            scatter_list.append(scatter_List[plane_Index[i]])
            normal_dirc_list.append(normal_Dirc[plane_Index[i]])
    
        alpha_list = np.array(alpha_list)
        scatter_list = np.array(scatter_list)
        normal_dirc_list = np.array(normal_dirc_list)
    
        assert scatter_list.shape == alpha_list.shape
    
        reflect_coef = np.sqrt(np.multiply((1-alpha_list),(1-scatter_list)))
        out_preasure = np.multiply(reflect_coef,insert_Preasure)
    
        coef = np.sum(np.multiply(normal_dirc_list,acoustic_Dirc), axis=1)
        coef = np.tile(coef,(3,1)).T
        reflection_vec = acoustic_Dirc-np.multiply(normal_dirc_list,coef)*2
        ref_vec_norm = np.linalg.norm(reflection_vec,axis=1)
        ref_vec_norm = np.tile(ref_vec_norm,(3,1)).T
        reflection_vec = np.divide(reflection_vec,ref_vec_norm)
    
        return reflection_vec,out_preasure
        
