import numpy as np
from Acoustic.BaseAcoustic import BaseAcoustic


class RT_Process:

    @staticmethod
    def linear_regression(response: np.ndarray, time: np.ndarray):
        """
        Use linear regression to estimate the frequency response
        :param response: receiver record sound pressure pa, from 125Hz to 4kHz
        :param time: record the ray insert receiver time
        :return: List of rt time, list of slope coe, list of smooth curve
        """
        assert response.shape[0] == time.shape[0]
        # Transform the sound pressure to sound pressure level
        response = BaseAcoustic.pressure2spl(response)
        # Create Bias term
        bias = np.ones((response.shape[0], 1))
        # Add dim to the time
        time = time[:, np.newaxis]
        # Construct the X
        time_bias = np.hstack([time, bias])
        # Compute the coe matrix (X^{T}X)^{-1}X^{T}
        coe_matrix = np.dot(np.linalg.inv(np.dot(time_bias.T, time_bias)), time_bias.T)

        # Data container
        smooth_list = []
        slope_list = []
        rt_list = []
        time_list = []

        # Loop over all the 6 band frequency
        for i in range(response.shape[1]):
            # Get the specific frequency response
            tmp_response = response[:, i]
            # omega first term is the slope, second term is
            omega = np.dot(coe_matrix, tmp_response)
            # The smooth decay curve
            smooth_response = np.dot(time_bias, omega)
            # RT time
            rt = -60 / omega[0]

            # Record data
            smooth_list.append(smooth_response)
            slope_list.append(omega)
            rt_list.append(rt)
            time_list.append(time.reshape(-1))

        rt_list = np.array(rt_list)
        slope_list = np.array(slope_list)
        smooth_list = np.array(smooth_list).T
        time_list = np.array(time_list).T

        return rt_list, smooth_list, time_list

    @staticmethod
    def back_integral(response: np.ndarray, time: np.ndarray, method: str):
        """
        Use back integral method to smooth
        :param response: receiver record sound pressure pa, from 125Hz to 4kHz
        :param time: record the ray insert receiver time
        :param method: T20, T30, T60 method only
        :return: List of rt time, list of smooth curve
        """
        method_list = ["T20", "T30", "T60"]
        assert method in method_list
        assert response.shape[0] == time.shape[0]

        """
        # Get the sequence length
        seq_len = response.shape[0]
        # Construct convolution matrix
        # TODO BUG IS HERE, Use np.cumsum()
        print("Before Conv")
        conv_m = np.tri(seq_len, seq_len, 0)
        print("After Construct Conv Matrix")
        """
        # Data container
        smooth_list = []
        smooth_time = []
        rt_list = []

        # Set Buffer
        buffer_spl = -20
        buffer_pressure = BaseAcoustic.spl2pressure(buffer_spl)

        # TODO SOME BUG IS HERE
        # Loop over all the frequency band
        for i in range(response.shape[1]):
            tmp_response = response[:, i]
            '''
            # Inside Value, Remove the value less than buffer
            needed_idx = np.where(tmp_response > buffer_pressure)[0]
            tmp_response = np.array(tmp_response[needed_idx])
            tmp_time = np.array(time[needed_idx])
            '''
            tmp_power_res = np.abs(tmp_response) ** 2
            # Get the init pressure
            p_init = max(tmp_power_res)
            # p_init = tmp_power_res[0]
            # Need to reverse the response
            tmp_power_res = tmp_power_res[::-1]
            # Get the smoothed sound pressure
            smooth_response = np.cumsum(tmp_power_res)
            # print(f"Back Integral: After Cumsum: {smooth_response}")
            smooth_response = smooth_response[::-1]
            # smooth_response = np.dot(tmp_response[:, np.newaxis].T, (p_init * conv_m))
            # smooth_response = smooth_response.reshape(-1)
            # Turn the pressure to the sound pressure level
            smooth_spl = BaseAcoustic.pressure2spl(np.sqrt(smooth_response))

            if method == "T20":
                # T20 method compute
                t_5 = smooth_spl - smooth_spl[0]
                r_t = (time[(t_5 <= -25).argmax(axis=0)] - time[(t_5 <= -5).argmax(axis=0)]) * 3

            elif method == "T30":
                # T30 method compute
                t_5 = smooth_spl - smooth_spl[0]
                r_t = (time[(t_5 <= -35).argmax(axis=0)] - time[(t_5 <= -5).argmax(axis=0)]) * 2

            else:
                # T60 method compute
                t_5 = smooth_spl - smooth_spl[0]
                r_t = time[(t_5 <= -65).argmax(axis=0)] - time[(t_5 <= -5).argmax(axis=0)]

            smooth_list.append(smooth_spl)
            smooth_time.append(time)

            rt_list.append(r_t)
        # TODO Bug is here
        ret_smooth_list = np.array(smooth_list)
        rt_list = np.array(rt_list)

        return rt_list, ret_smooth_list, smooth_time

    @staticmethod
    def bi_method(response: np.ndarray, time: np.ndarray, method: str):
        method_list = ["T20", "T30", "T60"]
        assert method in method_list
        assert response.shape[0] == time.shape[0]
        # Data container
        smooth_list_bi = []
        smooth_time = []
        rt_list =[]

        for i in range(response.shape[1]):
            tmp_response = np.abs(response[:, i])
            tmp_power_res = tmp_response ** 2
            seq_len = len(tmp_response)
            conv_m = np.tri(seq_len, seq_len, 0)
            # conv_m = np.tri(seq_len, seq_len, 0) * np.max(tmp_response)
            tmp_power_res = tmp_power_res[:, np.newaxis]

            filltered_res = np.dot(tmp_power_res.T, conv_m)
            # filltered_res = np.sqrt(filltered_res)
            filltered_res = BaseAcoustic.pressure2spl(np.sqrt(filltered_res))
            filltered_res = filltered_res.reshape(-1)

            if method == "T20":
                # T20 method compute
                t_5 = filltered_res - filltered_res[0]
                r_t = (time[(t_5 <= -25).argmax(axis=0)] - time[(t_5 <= -5).argmax(axis=0)]) * 3

            elif method == "T30":
                # T30 method compute
                t_5 = filltered_res - filltered_res[0]
                r_t = (time[(t_5 <= -35).argmax(axis=0)] - time[(t_5 <= -5).argmax(axis=0)]) * 2

            else:
                # T60 method compute
                t_5 = filltered_res - filltered_res[0]
                r_t = time[(t_5 <= -65).argmax(axis=0)] - time[(t_5 <= -5).argmax(axis=0)]

            smooth_list_bi.append(filltered_res)
            smooth_time.append(time)
            rt_list.append(r_t)

        smooth_list_bi = np.asarray(smooth_list_bi)
        smooth_time = np.asarray(smooth_time)
        rt_list = np.asarray(rt_list)

        return rt_list, smooth_list_bi, smooth_time


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Acoustic.BaseAcoustic import BaseAcoustic

    test_x = np.linspace(0, 10, 1000)
    test_y = -10 * test_x + 60
    noise = np.random.random(test_x.shape) * 10
    test_y += noise
    print(f"Response Max: {max(test_y)}")
    # BI response
    bi_res = BaseAcoustic.spl2pressure(np.array([test_y]).T)
    print(f"bi_res: {bi_res.shape}")

    smooth_list = RT_Process.back_integral(bi_res, test_x, "T60")

    smooth_list_1 = RT_Process.bi_method(bi_res, test_x, "T60")
    print(smooth_list_1)

    print(test_x)
    fig, ax = plt.subplots()
    ax.plot(test_x, test_y, color='green', linewidth=1.0)
    ax.plot(test_x, smooth_list[1], color='blue', linewidth=2.0)
    ax.plot(test_x, smooth_list, color='red', linewidth=2.0)
    plt.show()
