import numpy as np


class AcousticSource:

    def __init__(self, volume, location, sound_Presure_Level):
        self.volume = volume
        self.location = location
        self.sound_Presure_Level = sound_Presure_Level

    '''
    numRay refers to number of acoustic ray to generate from 125Hz to 8000Hz Octave Band
        shape n by 6 dimensional 
    '''

    def computeNumRay(self):
        return 3746 * self.volume

    def spl2Preasure(self):
        sound_presure = 10 ** (self.sound_Presure_Level / 20) * (20 * 10 ** (-6))
        return sound_presure * np.ones((self.computeNumRay(), 6))

    def initeTime(self):
        return np.zeros((self.computeNumRay(), 1))

    def generateAcousticRay(self):
        ray_direction = np.random.random([self.computeNumRay(), 3])
        ray_direction = ray_direction * 2 - 1
        source_location = np.tile(self.location, (self.computeNumRay(), 1))
        return ray_direction, source_location
