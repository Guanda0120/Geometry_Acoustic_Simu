import numpy as np
import matplotlib.pyplot as plt



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
        det_test_vec = np.ones((self.planeMatrix.shape[0], self.planeMatrix.shape[1], 1))
        det_plane = np.linalg.det(np.concatenate((self.planeMatrix, det_test_vec), axis=2))
        if np.sum(det_plane) == 0:
            return True
        else:
            return False

    def planeNormalDirec(self):
        centre_point = np.sum(self.planeMatrix, axis=1) / self.planeMatrix.shape[1]
        normal_direc = np.array([])
        for i in range(self.planeMatrix.shape[0]):
            temp_1 = self.planeMatrix[i, 1, ...]
            temp_2 = self.planeMatrix[i, 2, ...]
            temp_3 = self.planeMatrix[i, 3, ...]
            temp_p = np.cross((temp_3 - temp_2), (temp_1 - temp_2))
            temp_p = temp_p / np.linalg.norm(temp_p)
            normal_direc = np.concatenate((normal_direc, temp_p), axis=0)
        normal_direc = np.reshape(normal_direc, (self.planeMatrix.shape[0], 3))
        panel_d_temp = self.planeMatrix[:, 0, :]
        panel_d = -np.dot(normal_direc, panel_d_temp.T)[:, 0]
        return centre_point, normal_direc, panel_d

    '''
        centre_point is the center point
        normal_direc is the normal direction of the panel
        panel_d = is the analitical [Ax+By+Cz+D=0]~D
    '''

    def absorption_List(self):
        absorption_List = []

        for i in range(self.material_ID.shape[0]):
            index = self.material_ID[i] - 1
            absorption_List.append(self.absorption_Dictionary[index])

        absorption_List = np.array(absorption_List)
        absorption_List.reshape((-1, 6))
        return absorption_List

    def scater_List(self):
        scatter_List = []

        for i in range(self.material_ID.shape[0]):
            index = self.material_ID[i] - 1
            scatter_List.append(self.scater_Dictionary[index])

        scatter_List = np.array(scatter_List)
        scatter_List.reshape((-1, 6))
        return scatter_List


class AcousticSource:

    def __init__(self, volume, location, sound_Power_Level):
        self.volume = volume
        self.location = location
        self.sound_Power_Level = sound_Power_Level

    '''
    numRay refers to number of acoustic ray to generate from 125Hz to 8000Hz Octave Band
        shape n by 6 dimensional 
    '''

    def computeNumRay(self):
        return 3746 * self.volume

    def swl2SoundPower(self):
        w_0 = 1e-12
        sound_presure = 10 ** (self.sound_Power_Level / 10) * w_0
        return sound_presure * np.ones((self.computeNumRay(), 6))

    def preasure2SPL(self, presure_Array):
        spl = np.log10(presure_Array / 20e-6) * 20
        return spl

    def initeTime(self):
        return np.zeros((self.computeNumRay(), 1))

    def generateAcousticRay(self):
        ray_direction = np.random.random([self.computeNumRay(), 3])
        ray_direction = ray_direction * 2 - 1
        source_location = np.tile(self.location, (self.computeNumRay(), 1))
        return ray_direction, source_location


class Simulation():

    def __init__(self, normal_Direction, plane_D, acoustic_Ray, acoustic_Loction, acoustic_Alpha, acoustic_Scatter):
        self.normal_Direction = normal_Direction
        self.plane_D = plane_D
        self.acoustic_Ray = acoustic_Ray
        self.acoustic_Loction = acoustic_Loction
        self.acoustic_Alpha = acoustic_Alpha
        self.acoustic_Scatter = acoustic_Scatter

    def preasure2SPL(self, presure_Array):
        spl = np.log10(presure_Array / 20e-6) * 20
        return spl

    def randomGenerateOrthognal(self, normal_Vector):

        orthognal_vector = np.random.random([3])
        if normal_Vector[2] != 0:
            orthognal_vector[2] = (0 - (
                        orthognal_vector[0] * normal_Vector[0] + orthognal_vector[1] * normal_Vector[1])) / \
                                  normal_Vector[2]
        elif normal_Vector[1] != 0:
            orthognal_vector[1] = (0 - (
                        orthognal_vector[0] * normal_Vector[0] + orthognal_vector[2] * normal_Vector[2])) / \
                                  normal_Vector[1]
        else:
            orthognal_vector[0] = (0 - (
                        orthognal_vector[1] * normal_Vector[1] + orthognal_vector[2] * normal_Vector[2])) / \
                                  normal_Vector[0]

        return orthognal_vector

    def pointInFinitePlane(self, orthognal_Vector, speci_Point, temp_Plane):

        '''
        follow algorithm from this web
        '''
        '''
        https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
        https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect?newreg=6fd1d1e602a847c2b9959dca99f92963
        '''

        assert orthognal_Vector.shape[0] == 3
        assert speci_Point.shape[0] == 3

        temp_Plane_1 = np.vstack([temp_Plane[-1, :], temp_Plane[0:-1, :]])
        side_Vector = temp_Plane - temp_Plane_1
        cross_time = 0

        for i in range(side_Vector.shape[0]):
            temp_Side_Vector = side_Vector[i]
            edge_point = temp_Plane_1[i]
            std_vector = np.array([1, 1, 1])

            cross_res_1 = np.linalg.det(np.vstack([std_vector, orthognal_Vector, temp_Side_Vector]))
            cross_res_2 = np.linalg.det(np.vstack([std_vector, temp_Side_Vector, orthognal_Vector]))
            cross_res_1a = np.linalg.det(np.vstack([std_vector, (edge_point - speci_Point), temp_Side_Vector]))
            cross_res_2a = np.linalg.det(np.vstack([std_vector, (speci_Point - edge_point), orthognal_Vector]))

            if cross_res_1 == 0:
                cross_state = 0
            else:
                coeff_1 = cross_res_1a / cross_res_1
                coeff_2 = cross_res_2a / cross_res_2

                if (coeff_2 >= 0) & (coeff_2 <= 1) & (coeff_1 >= 0):
                    cross_state = 1
                else:
                    cross_state = 0
                cross_time += cross_state

        if (cross_time % 2 == 1):
            in_state = 1
        else:
            in_state = 0

        return in_state

    "there should be some problem"

    def vectorInfinitePlaneIntersection(self, plane_Dirc, acoustic_Dirc, acoustic_Sour):

        geometry = self.plane_D
        planeIntersectionP = []
        coef = []

        for i in range(self.plane_D.shape[0]):
            temp_source_point = np.tile(acoustic_Sour, (4, 1))  # 4 is for formal 4 sides plane, it should be mod
            temp_normalD = plane_Dirc[i, :]
            prod1 = np.dot((temp_source_point - geometry[i, :, :]), temp_normalD)
            prod2 = np.dot(acoustic_Dirc, temp_normalD.T)
            prod3 = 0 - np.divide(prod1, prod2)
            speci_point = temp_source_point + np.multiply(np.tile(prod3, (3, 1)).T, np.tile(acoustic_Dirc, (4, 1)))
            speci_point = speci_point[0, :]

            planeIntersectionP.append(speci_point)
            coef.append(prod3[0])

        planeIntersectionP = np.array(planeIntersectionP)
        coef = np.array(coef)

        return planeIntersectionP, coef

    def checkIntersection3D(self, acoustic_Dirc, acoustic_Sour):

        intersection_point = []
        plane_index = []
        ray_index = []

        for i in range(acoustic_Dirc.shape[0]):

            infinite_intersection, coef = self.vectorInfinitePlaneIntersection(self.normal_Direction, acoustic_Dirc[i],
                                                                               acoustic_Sour[i])
            plane = self.plane_D
            normal_Direc = self.normal_Direction

            for k in range(plane.shape[0]):
                if coef[k] > 0:
                    temp_plane = plane[k]
                    temp_intersection_point = infinite_intersection[k]
                    temp_normal_direc = normal_Direc[k]

                    orthognal_Vector = self.randomGenerateOrthognal(temp_normal_direc)
                    cross_state = self.pointInFinitePlane(orthognal_Vector, temp_intersection_point, temp_plane)

                    lenght_vec = np.linalg.norm(temp_intersection_point - acoustic_Sour[i])
                    if cross_state == 1 and lenght_vec > 1e-14:
                        intersection_point.append(temp_intersection_point)
                        plane_index.append(k)
                        ray_index.append(i)
        intersection_point = np.array(intersection_point)
        # assert intersection_point.shape==acoustic_Sour.shape

        return plane_index, ray_index, intersection_point

    def attenuation(self, total_distance):
        '''
           total_distance is m*1 matrix, m is m acoustic rays
        '''

        num_ray = total_distance.shape[0]
        p_0 = 20e-6
        attenuation_coef = np.array([1.5625e-06, 6.2500e-06, 2.5000e-05, 1.0000e-04, 4.0000e-04, 1.6000e-03])
        attenuation_coef = np.tile(attenuation_coef, (num_ray, 1))

        # distance = np.tile(total_distance,(6,1)).T

        assert total_distance.shape == attenuation_coef.shape

        attenuation = np.multiply(attenuation_coef, total_distance)

        return attenuation

    def reflection(self, energy_Total, plane_Index, acoustic_Dirc):
        '''
        var interpretation
            【energy_Total】m*6 matrix, m is for m acoustic ray, 6 is for 6 octave band, The total sound
                            absorption during k times reflection
            【plane_Index】is m*1 matrix, m is for m acoustic ray. the m_th ray intersect with n_th plane
            【acoustic_Dirc】is m*3 matrix, m is for m acoustic ray, 3 is for x-y-z direction
        '''

        alpha_List = self.acoustic_Alpha
        scatter_List = self.acoustic_Scatter
        normal_Dirc = self.normal_Direction

        alpha_list = []
        scatter_list = []
        normal_dirc_list = []

        for i in range(acoustic_Dirc.shape[0]):
            alpha_list.append(alpha_List[plane_Index[i]])
            scatter_list.append(scatter_List[plane_Index[i]])
            normal_dirc_list.append(normal_Dirc[plane_Index[i]])

        # temp alpha list & scatter list
        alpha_temp = np.array(alpha_list)
        scatter_temp = np.array(scatter_list)
        normal_dirc_list = np.array(normal_dirc_list)

        # reflection vector
        coef = np.sum(np.multiply(normal_dirc_list, acoustic_Dirc), axis=1)
        coef = np.tile(coef, (3, 1)).T
        reflection_vec = acoustic_Dirc - np.multiply(normal_dirc_list, coef) * 2
        ref_vec_norm = np.linalg.norm(reflection_vec, axis=1)
        ref_vec_norm = np.tile(ref_vec_norm, (3, 1)).T
        reflection_vec = np.divide(reflection_vec, ref_vec_norm)

        # alpha and scatter
        energy_temp = np.multiply((1 - alpha_temp), (1 - scatter_temp))
        energy_total = np.multiply(energy_Total, energy_temp)

        return reflection_vec, energy_total

    def propagate(self, startPoint, rayDirec, totalDistance):

        # first check itersection point
        plane_index, _, intersection_point = self.checkIntersection3D(rayDirec, startPoint)
        temp_distance = np.linalg.norm((startPoint - intersection_point), axis=1, keepdims=True)

        # distance add
        temp_distance = np.tile(temp_distance, (1, 6))
        totalDistance = totalDistance + temp_distance

        # convert to ndarray
        plane_index = np.array(plane_index)
        intersection_point = np.array(intersection_point)
        totalDistance = np.array(totalDistance)

        return plane_index, intersection_point, totalDistance

    '''
    error here
    '''

    def defusion(self, plane_Index, acoustic_Dirc, insert_Point, source_Power, reciver_Location, prev_Energy,
                 total_Dis):

        '''
        【plane_Index】is m*1 vector, m is for m acoustic rays
        【acoustic_Dirc】is m*3 matrix, m is for m acoustic rays, 3 is for x-y-z
        【insert_Point】is m*3 matrix, m is for m acoustic rays, 3 is for x-y-z
        【source_Power】is m*6 matrix, m is for m acoustic rays, 6 is for 6 Octave Bands
        【reciver_Location】is 1*3 vector
        【total_Energy】is m*6 matrix, m is for m acoustic rays, 6 is for 6 Octave Bands
        【total_Dis】is m*1 vector, m is for m acoustic rays
        '''

        # defusion rain algorthim

        alpha_List = self.acoustic_Alpha
        scatter_List = self.acoustic_Scatter
        normal_Dirc = self.normal_Direction

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

        '''
        compute cosine of normal direction and ray  direction
        every vec is normalized
        '''
        scatter_vec = reciver_Location - insert_Point
        scatter_vec_norm = np.linalg.norm(scatter_vec, axis=1)
        scatter_vec_norm = np.tile(scatter_vec_norm, (3, 1)).T
        scatter_vec = np.divide(scatter_vec, scatter_vec_norm)
        cos_scatter = np.sum(np.multiply(scatter_vec, normal_dirc_list), axis=1)

        boolean_cond = np.maximum(cos_scatter, 0)
        boolean_cond = np.where(boolean_cond <= 0, boolean_cond, 1)

        cos_scatter = np.tile(cos_scatter, (6, 1)).T

        assert cos_scatter.shape == prev_Energy.shape

        distance = np.linalg.norm((insert_Point - reciver_Location), axis=1, keepdims=True)
        distance = np.tile(distance, (1, 6))
        total_distance = distance + total_Dis

        '''
        energy compute
        '''
        energy_coef = np.multiply(scatter_list, (1 - alpha_list))
        total_coef = np.multiply(np.multiply(energy_coef, prev_Energy), cos_scatter)
        # z_0 is air impedance
        rho_0 = 1.224
        c_0 = 343
        z_0 = c_0 * rho_0

        # error here [total_distance]
        sound_preasure = np.sqrt(
            np.divide(np.multiply(source_Power, total_coef), total_distance ** 2) * (z_0 / (4 * np.pi)))

        '''
        attenuation and time 
        '''
        attenuate_spl = self.attenuation(total_distance)
        end_spl = self.preasure2SPL(sound_preasure)
        end_spl_a = end_spl - attenuate_spl

        total_time = total_distance / c_0
        '''
        kick off cos<0 items
        '''
        valid_spl = []
        valid_time = []
        for i in range(boolean_cond.shape[0]):
            if boolean_cond[i] == 1:
                valid_spl.append(end_spl_a[i])
                valid_time.append(total_time[i])

        valid_spl = np.array(valid_spl)
        valid_time = np.array(valid_time)

        return valid_spl, valid_time

    def currentSpl(self, totalDis, energyCoef, sourcePower):

        '''
        totalDis: dim:[m*6] distance of m rays travel to this reflection plane
        energyCoef: dim:[m*6] energy portion of original power
        sourcePl: dim[m*6]
        '''
        # Prepare for compute
        rho_0 = 1.224
        c_0 = 343
        z_0 = c_0 * rho_0

        redu_power = np.multiply(energyCoef, sourcePower)
        sound_preasure = np.sqrt(np.divide(((z_0 / (4 * np.pi)) * redu_power), (totalDis ** 2)))
        spl = self.preasure2SPL(sound_preasure)
        time = totalDis / c_0

        return spl, time


class Reciver:

    def __init__(self, location, radius):
        self.location = location
        self.radius = radius

    def swl2Power(self,swl):
        w_0 = 1e-12
        power = (10**(swl/10))*w_0
        return power

    def preasure2SPL(self, presure_Array):
        p_0 = 2.0e-5
        spl = np.log10(presure_Array / p_0) * 20
        return spl

    def reshapeTensor(self, rayDirection, rayStart, rayDistance, rayEnergy):

        # reshape the array
        rayDirection = rayDirection.reshape(-1, rayDirection.shape[2])
        rayStart = rayStart.reshape(-1, rayStart.shape[2])
        rayDistance = rayDistance.reshape(-1, rayDistance.shape[2])
        rayEnergy = rayEnergy.reshape(-1, rayEnergy.shape[2])

        return rayDirection, rayStart, rayDistance, rayEnergy

    def checkIntersection(self, rayDirection, rayStart, rayDistance, rayEnergy):

        numerator = np.sum(np.multiply(rayDirection, (self.location - rayStart)), axis=1)
        denominator = np.sum(np.multiply(rayDirection, rayDirection), axis=1)
        scaler_1 = np.divide(numerator, denominator)
        scaler_1 = scaler_1[:, np.newaxis]
        scaler = np.concatenate((scaler_1, scaler_1, scaler_1), axis=1)
        r = np.linalg.norm((self.location - (rayStart + np.multiply(rayDirection, scaler))), axis=1)

        boolean_cond = np.maximum(scaler_1, 0)
        boolean_cond = np.where(boolean_cond <= 0, boolean_cond, 1)
        boolean_cond = boolean_cond[:, 0]

        r = np.multiply(boolean_cond, r)

        dis_boolean = np.where(r < self.radius, r, 0)
        dis_boolean = np.where(dis_boolean == 0, dis_boolean, 1)

        distance = np.linalg.norm(np.multiply(rayDirection,scaler),axis=1)
        distance = np.tile(distance, (6, 1)).T
        rayDistance = distance+rayDistance

        int_rayDirec = []
        int_raySour = []
        int_rayDis = []
        int_rayEnergy = []

        for i in range(dis_boolean.shape[0]):
            if dis_boolean[i] == 1:
                int_rayDirec.append(rayDirection[i])
                int_raySour.append(rayStart[i])
                int_rayDis.append(rayDistance[i])
                int_rayEnergy.append(rayEnergy[i])

        int_rayDirec = np.array(int_rayDirec)
        int_raySour = np.array(int_raySour)
        int_rayDis = np.array(int_rayDis)
        int_rayEnergy = np.array(int_rayEnergy)

        return int_rayDis, int_rayEnergy

    def propagateReduce(self, distance, startPower, reyEnergy):

        '''
        :param distance: ndarray, dim: i*6. where i is times that cross the reciver
        :param startPower: initial power level of sound source[dB], dim=/single value/
        :return:
        end_spl_a: cross Sound Preasure Level[dB]，dim: i*6.
        end_time: travel time[s], dim: i*6
        '''

        num_ray = distance.shape[0]
        startPower = self.swl2Power(startPower)

        c_0 = 343
        rho_0 = 1.224
        z_0 = c_0*rho_0
        reduPower = startPower*reyEnergy
        # end Spl without air attenuation
        coef = (z_0/(np.pi*4))
        end_preasure = np.sqrt(np.divide(reduPower,(distance**2))*coef)
        end_spl = self.preasure2SPL(end_preasure)
        #
        attenuation = np.array([1.5625e-06, 6.2500e-06, 2.5000e-05, 1.0000e-04, 4.0000e-04, 1.6000e-03])
        attenuation = np.tile(attenuation, (num_ray, 1))

        attenuated_spl = np.multiply(attenuation,distance)

        end_time = distance/c_0
        end_spl_a = end_spl-attenuated_spl

        return end_spl_a,end_time

    def combine(self, refSpl, refTime, scaSpl, scaTime):
        '''

        :param refSpl:
        :param refTime:
        :param scaSpl:
        :param scaTime:
        :return:
        '''
        # reshape the tensor

        scaTime = scaTime.reshape(-1, scaTime.shape[2])
        scaSpl = scaSpl.reshape(-1, scaSpl.shape[2])

        assert scaTime.shape[1] == refTime.shape[1]
        assert scaSpl.shape[1] == scaSpl.shape[1]

        # append
        spl_all = np.vstack([refSpl, scaSpl])
        time_all = np.vstack([refTime, scaTime])

        return spl_all, time_all

    '''
    Error Here
    '''
    def sortAsTime(self, spl, time):

        '''

        :param spl:
        :param time:
        :return:
        '''

        sorted_spl = np.zeros(spl.shape)
        temp_time = time[:, 0]
        time_sort = np.sort(temp_time)
        index_time = np.argsort(temp_time)

        for i in range(time_sort.shape[0]):
            sorted_spl[index_time[i], :] = spl[i, :]

        return sorted_spl, time_sort

class AdaptiveDSP():

    def __init__(self):
        return

    def linearRegresion(self, response, time):
        # Least Square Method

        assert response.shape[0] == time.shape[0]

        bias = np.ones((response.shape[0], 1))
        time = time[:, np.newaxis]
        time = np.hstack([time, bias])

        slope_1 = np.linalg.inv((np.dot(np.transpose(time), time)))
        slope_2 = np.dot(np.transpose(time), response)
        slope = np.dot(slope_1, slope_2)
        slope = slope[:, np.newaxis]

        fillter_response = np.dot(time, slope)

        rt = -60 / slope[0, 0]

        return fillter_response, slope, rt

    def backIntegral(self, response, time, init_Spl):
        # This algorthim refer 'M. R. Schroeder' publish in 1965 Bell Lab
        # New method of measuring reverberation time. J Acoust Soc Am 1965; 37: 409.

        energy_res = 10 ** (response / 20) * (20e-6)
        energy_res = energy_res ** 2

        init_energy = (10 ** (init_Spl / 20) * (20e-6)) ** 2

        seq_len = response.shape[0]
        energy_res = energy_res[:, np.newaxis]

        conv_m = np.tri(seq_len, seq_len, 0) * init_energy
        filltered_res = np.dot(energy_res.T, conv_m)
        filltered_res = np.sqrt(filltered_res)
        filltered_res = np.log10(filltered_res / 20e-6) * 20
        filltered_res = filltered_res.reshape(-1)

        start_spl = filltered_res[0]
        t_5 = filltered_res - start_spl
        t_25 = filltered_res - 25
        r_t = (time[(t_5 <= -25).argmax(axis=0)] - time[(t_5 <= -5).argmax(axis=0)]) * 3

        return filltered_res, r_t


# Parameter input
'''
    Basic Room Geometry Parameter Settings
    Right Hand Law
'''
room_1 = [[[ 4.0, 4.0, 0.0],
           [-4.0, 4.0, 0.0],
           [-4.0,-4.0, 0.0],
           [ 4.0,-4.0, 0.0]],
          [[ 4.0, 4.0, 0.0],
           [ 4.0,-4.0, 0.0],
           [ 4.0,-4.0, 3.0],
           [ 4.0, 4.0, 3.0]],
          [[ 4.0,-4.0, 0.0],
           [-4.0,-4.0, 0.0],
           [-4.0,-4.0, 3.0],
           [ 4.0,-4.0, 3.0]],
          [[-4.0,-4.0, 0.0],
           [-4.0, 4.0, 0.0],
           [-4.0, 4.0, 3.0],
           [-4.0,-4.0, 3.0]],
          [[-4.0, 4.0, 0.0],
           [ 4.0, 4.0, 0.0],
           [ 4.0, 4.0, 3.0],
           [-4.0, 4.0, 3.0]],
          [[ 4.0, 4.0, 3.0],
           [ 4.0,-4.0, 3.0],
           [-4.0,-4.0, 3.0],
           [-4.0, 4.0, 3.0]]]

room_1=np.array(room_1)

# Material Absorption Coeffiecient
material_abs_list = np.array([[0.1,0.1,0.1,0.1,0.1,0.1],
                              [0.3,0.3,0.3,0.3,0.3,0.3],
                              [0.5,0.5,0.5,0.5,0.5,0.5]])

# Material Absorption Coeffiecient
material_sca_list = np.array([[0.1,0.1,0.1,0.1,0.1,0.1],
                              [0.15,0.15,0.15,0.15,0.15,0.15],
                              [0.2,0.2,0.2,0.2,0.2,0.2]])

# Material Dictionary
# '1' is the first element
material_ID = np.array([2,3,1,2,3,1])


roomTest = RoomGeometry(room_1, material_abs_list, material_sca_list, material_ID)
absMatrix = roomTest.absorption_List()
scaMatrix = roomTest.scater_List()
cenPoint, norDric, planeD = roomTest.planeNormalDirec()

# Source Prepare
'''
    var in source object
    volume is the volume of the sound field
    location is the ndarray,[x,y,z]position of the source
    spl is the inite sound presure level of the 
'''
volume = 1
sourceLocation = np.array([0, 0, 1.2])
startSwl = 94
endSpl = startSwl - 65

singleSource = AcousticSource(volume, sourceLocation, startSwl)
startEnergy = singleSource.swl2SoundPower()
startTime = singleSource.initeTime()
rayDirec, raySource = singleSource.generateAcousticRay()
numRay = singleSource.computeNumRay()

# reciver condition
reciverLocation = np.array([2, 2, 1.5])

# Simulation Prepare
# trigger var
maxSpl = startSwl
iterNum = 0

# reflection list
direcList_R = []
sourceList_R = []
direcList_R.append(rayDirec)
sourceList_R.append(raySource)

# every reflect ray travel distance
init_dis = np.zeros((numRay, 6))
disList_R = []
disList_R.append(init_dis)

# scatter list
spl_S = []
time_S = []

# initial distacne, time, energy coef
energyList_R = []
totalDistance = np.zeros((numRay, 6))
energyCoef = np.ones((numRay, 6))
energyList_R.append(energyCoef)

while maxSpl >= endSpl:
    simuProgress = Simulation(norDric, room_1, rayDirec, raySource, absMatrix, scaMatrix)
    planeIndex, intersectPoint, totalDistance = simuProgress.propagate(raySource, rayDirec, totalDistance)
    reflectionVec, energyCoef_temp = simuProgress.reflection(energyCoef, planeIndex, rayDirec)
    # var energyCoef_temp is the energy coeff after the current reflection
    # var energyCoef is the energy coeff before the current reflection
    validSpl, validTime = simuProgress.defusion(planeIndex, rayDirec, intersectPoint, startEnergy, reciverLocation,
                                                energyCoef, totalDistance)
    # the validSpl is the SPL after reflection
    refSpl, refTime = simuProgress.currentSpl(totalDistance, energyCoef_temp, startSwl)
    maxSpl = np.max(refSpl)
    minSpl = np.min(refSpl)
    raySource = intersectPoint
    rayDirec = reflectionVec

    spl_S.append(validSpl)
    time_S.append(validTime)

    direcList_R.append(reflectionVec)
    sourceList_R.append(intersectPoint)
    disList_R.append(totalDistance)
    energyList_R.append(energyCoef_temp)

    energyCoef = energyCoef_temp

    iterNum += 1
    print('In iter', iterNum, ':')
    print('The max SPL is:', round(maxSpl, 2), 'dB')
    print('The min SPL is:', round(minSpl, 2), 'dB')
    print('===========================================================')

# transfer into ndarray
spl_S = np.array(spl_S)
time_S = np.array(time_S)
direcList_R = np.array(direcList_R)
sourceList_R = np.array(sourceList_R)
disList_R = np.array(disList_R)
energyList_R =np.array(energyList_R)

assert spl_S.shape[0] == time_S.shape[0] == (direcList_R.shape[0] - 1) == (sourceList_R.shape[0] - 1) == (
            disList_R.shape[0] - 1)


r_1 = Reciver(reciverLocation,0.1)
direcList_R1, sourceList_R1, disList_R1, energyList_R1 = r_1.reshapeTensor(direcList_R,sourceList_R,disList_R,energyList_R)
disList_R1,energyList_R1 = r_1.checkIntersection(direcList_R1, sourceList_R1, disList_R1, energyList_R1)
spl_R, time_R = r_1.propagateReduce(disList_R1,startSwl,energyList_R1)
spl_All,time_All = r_1.combine(spl_R,time_R,spl_S,time_S)
spl_Sort, time_Sort = r_1.sortAsTime(spl_All,time_All)



# print 1kHz

spl_1000 = spl_Sort[:,3]

fig, ax = plt.subplots()
fig.suptitle('1k Hz Decay Curve')
ax.plot(time_Sort, spl_1000, linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Sound Preasure Level (dB)')
plt.show()
