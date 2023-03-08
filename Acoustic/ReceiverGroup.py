import numpy as np
from typing import List
from Acoustic.Receiver import Receiver
from Acoustic.Source import Source
from RoomGeometry.FieldGeometry import FieldGeometry
from Acoustic.RayGroup import RayGroup


class ReceiverGroup:

    # This a Group Contain Receiver

    def __init__(self, location: np.ndarray, src: Source, sound_field: FieldGeometry, rec_radius: float):
        """

        :param location:
        :param src:
        :param sound_field:
        :param rec_radius:
        """
        # Check legal input
        assert len(location.shape) == 2
        # Create a list as receiver container
        self.receiver_container: List[Receiver] = []
        # Init every receiver
        for i in range(location.shape[0]):
            self.receiver_container.append(Receiver(location[i], src, sound_field, rec_radius))

    def predict_direc(self):
        """
        Renew the spl  and time of direc sound get receiver
        :return:
        """
        for receiver in self.receiver_container:
            receiver.direc_insert()

    def image_source_all(self, max_order: int):
        """
        Image source all the receiver
        :param max_order: Image source order, the max reflex time
        :return: Renew the receiver
        """
        for receiver in self.receiver_container:
            receiver.image_source(max_order)

    def image_diffuse_all(self):
        """
        Use this method after image source method to predict image source path diffusion
        :return: Renew the diffusion list
        """
        for receiver in self.receiver_container:
            receiver.image_diffuse()

    def predict_stochastic(self, ray_group: RayGroup):
        """
        Use stochastic_insert to predict every direct ray insert receiver in ray group
        :param ray_group: Ray group object, not change the obj
        :return: Renew the receiver
        """
        for receiver in self.receiver_container:
            receiver.stochastic_insert(ray_group)

    def predict_diffuse(self, ray_group: RayGroup):
        """
        Use predict_diffuse to predict every diffuse ray insert receiver in ray group
        :param ray_group: Ray group object, not change the obj
        :return: Renew the receiver, and diffuse num return
        """
        diffuse_num_list = []
        for receiver in self.receiver_container:
            diffuse_num = receiver.lambert_diffuse(ray_group)
            diffuse_num_list.append(diffuse_num)

        return diffuse_num_list

    def sto2img_all(self, max_order: int):
        """
        Get stochastic to image_source, A Huge Innovation
        :param max_order: Max image source order
        :return: Renew the image_source info
        """
        for receiver in self.receiver_container:
            receiver.stochastic_2_image(max_order)

    def merge_sort(self):
        """
        Merge and sort all the receiver
        :return: Renew all the receiver
        """
        for receiver in self.receiver_container:
            receiver.merge_together()

    def fsmatch_all(self, fs: int):
        for receiver in self.receiver_container:
            receiver.match_fs(fs)

    def rt_estimate(self, back_integral_method: str):
        """
        Use LR and BI method to estimate rt
        :param back_integral_method: T20, T30, T60 could be choice
        :return: Renew each receiver
        """
        for receiver in self.receiver_container:
            receiver.lr_process()
            receiver.back_integral_process(back_integral_method)


if __name__ == '__main__':
    locat = np.array([[1, 1, 1], [2, 2, 2]])
    rec_group = ReceiverGroup(locat, 0.1)
    for rec in rec_group.receiver_container:
        print(rec.rec_location)
