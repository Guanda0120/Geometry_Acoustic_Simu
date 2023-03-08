import numpy as np
import scipy
import lmfit
from Acoustic.CONST import air_rho, acoustic_speed


class Material:
    """
    Create a Material Object
    """

    def __init__(self, name: str, diffuse_abs: np.ndarray, scatter: np.ndarray, **kwargs):
        """

        :param name:
        :param diffuse_abs:
        """
        self.name = name
        self.diffuse_abs = diffuse_abs
        self.scatter = scatter
        self.model_name = kwargs['plane_wave_model']
        self.opt_max_res = kwargs['opt_max_res']
        assert self.model_name in ['JCA', 'DB_Miki']
        # Get the normal incidence absorption coefficient
        self.normal_abs, self.normal_res = self.fsolve_opt()

        # Record normal incidence plane model parameter
        self.param_dict = {}
        self.ref_0 = np.sqrt(1-self.normal_abs)

        # Choose the model to use
        if self.model_name == 'JCA':
            # Optimize use JCA model
            tort, flow_res, porous, viscous, thermal, thick, output = self.jca_opt(self.normal_abs)
            self.param_dict['tort'] = tort
            self.param_dict['flow_res'] = flow_res
            self.param_dict['porous'] = porous
            self.param_dict['viscous'] = viscous
            self.param_dict['thermal'] = thermal
            self.param_dict['thick'] = thick
            self.opt_info = output

            # Check the residual
            opt_abs, opt_ref = self.jcaModel(tort, flow_res, porous, viscous, thermal, thick)
            # Res less than max_res or not
            if np.max(np.abs(opt_abs - self.normal_abs)) <= self.opt_max_res:
                # Valid optimization result, use complex number reflection coe
                self.ref_0 = opt_ref

        else:
            # Optimize use DB_Miki model
            flow_res, thick, output = self.dbmiki_opt(self.normal_abs)
            self.param_dict['flow_res'] = flow_res
            self.param_dict['thick'] = thick
            self.opt_info = output

            # Check the residual
            opt_abs, opt_ref = self.dbMiki(flow_res, thick)
            # Res less than max_res or not
            if np.max(np.abs(opt_abs - self.normal_abs)) <= self.opt_max_res:
                # Valid optimization result, use complex number reflection coe
                self.ref_0 = opt_ref

    def __str__(self):
        return f"{self.name}; reflection: {self.ref_0}; scatter: {self.scatter}"

    @staticmethod
    def normal2diffuse_res(normal_alpha: float, *args):
        """
        The object function use to solve the nonlinear equation.
        :param normal_alpha: The normal incidence absorption, the param to optimize.
        :param args: args[0] refers to diffuse alpha
        :return: The object function to be optimized
        """
        reflection = np.sqrt(1 - normal_alpha)
        diffuse_alpha = args[0]
        return (2 / (1 - reflection) - (1 - reflection) / 2 + 2 * np.log((1 - reflection) / 2)) * 8 * (
                (1 - reflection) / (1 + reflection)) ** 2 - diffuse_alpha

    def fsolve_opt(self):
        """
        Use scipy fsolve method to solve the normal2diffuse function, No Boundary
        :param diffuse_alpha: The absorption coefficient measure in diffuse field
        :return: The optima normal incidence absorption coefficient, and the residual of the object function
        """

        normal_abs = []
        res = []

        # Loop over all the octave band coe
        for tmp_abs in self.diffuse_abs:
            # Optimize the function
            normal_alpha = scipy.optimize.fsolve(self.normal2diffuse_res, tmp_abs, args=tmp_abs)
            # Compute the residual of the function
            tmp_res = self.normal2diffuse_res(normal_alpha[0], tmp_abs)
            # append the result
            normal_abs.append(normal_alpha[0])
            res.append(tmp_res)

        # Transform into array
        normal_abs = np.asarray(normal_abs)
        res = np.asarray(res)

        return normal_abs, res

    @staticmethod
    def jcaModel(tort: float, flow_res: float, porous: float, viscous: float, thermal: float, thick: float):
        """
        Use JCA Model to compute the normal incidence absorption
        :param tort: Tortuosity [-] range[1-3]
        :param flow_res: Airflow resistivity [Ns/m4]
        :param porous: Porosity [-] range [0-1]
        :param viscous: Viscous characteristic dimension [m] [range 10-1000 10^-6m]
        :param thermal: Thermal characteristic dimension [m] [range 10-1000 10^-6m]
        :param thick: thickness [m]
        :return: normal incidence alpha, reflection coefficient
        """
        # Some Const
        freq = np.array([125, 250, 500, 1e3, 2e3, 4e3])
        z_0 = air_rho * acoustic_speed
        eta = 1.84e-5  # Visocity of air [Poiseuille] (1.84*10^-5)
        omega = 2 * np.pi * freq  # Angular frequency
        gamma = 1.4  # Ratio of the specific heat capacity [-]
        npr = 0.71  # Prandtl number (0.77 at 20*C)
        p_0 = 101320  # Atmospheric pressure [N/m2]

        rho_eq = (tort * air_rho / porous) * (1 + ((flow_res * porous) / (1j * omega * air_rho * tort)) * np.sqrt(
            1 + (4 * 1j * (tort ** 2) * eta * air_rho * omega) / ((flow_res * viscous * porous) ** 2)))
        k_eq = (gamma * p_0 / porous) * ((gamma - (gamma - 1) / (
                1 + ((8 * eta) / (1j * thermal ** 2 * npr * omega * air_rho)) * np.sqrt(
            1 + (1j * air_rho * omega * npr * thermal ** 2) / (16 * eta)))) ** (-1))

        # Characristic impedance
        z_c = np.sqrt(rho_eq * k_eq)
        # Complex wave number
        k_c = np.sqrt(rho_eq / k_eq) * omega
        # Surface impedance of sample
        z_s = -1j * z_c * (1 / np.tan(k_c * thick))
        # Normalized specific acoustic impedance
        z_n = z_s / z_0
        # Reflection coefficient
        r = (z_s - z_0) / (z_s + z_0)
        # normal absorption coefficient
        normal_abs = 1 - np.abs(r) ** 2

        return normal_abs, r

    @staticmethod
    def jcaModel_Residual(para: lmfit.Parameters, freq: np.ndarray, obj_data: np.ndarray = None):
        """
        DB Miki model, predict the normal incidence absorption coefficient.
        :param para: A lmfit.Parameters object, with the parameter to be optimized
        :param freq: The frequency of the absorption coefficient
        :param obj_data: The objective absorption coefficient, ** Normal incidence
        :return: The residual of the predict value and truth value
        """
        # Wrap the Objective Value
        val_dict = para.valuesdict()
        tort = val_dict["tort"]
        flow_res = val_dict["flow_res"]
        porous = val_dict["porous"]
        viscous = val_dict["viscous"]
        thermal = val_dict["thermal"]
        thick = val_dict["thick"]

        # Some Const
        z_0 = air_rho * acoustic_speed
        eta = 1.84e-5  # Visocity of air [Poiseuille] (1.84*10^-5)
        omega = 2 * np.pi * freq  # Angular frequency
        gamma = 1.4  # Ratio of the specific heat capacity [-]
        npr = 0.71  # Prandtl number (0.77 at 20*C)
        p_0 = 101320  # Atmospheric pressure [N/m2]

        rho_eq = (tort * air_rho / porous) * (1 + ((flow_res * porous) / (1j * omega * air_rho * tort)) * np.sqrt(
            1 + (4 * 1j * (tort ** 2) * eta * air_rho * omega) / ((flow_res * viscous * porous) ** 2)))
        k_eq = (gamma * p_0 / porous) * ((gamma - (gamma - 1) / (
                1 + ((8 * eta) / (1j * thermal ** 2 * npr * omega * air_rho)) * np.sqrt(
            1 + (1j * air_rho * omega * npr * thermal ** 2) / (16 * eta)))) ** (-1))

        # Characristic impedance
        z_c = np.sqrt(rho_eq * k_eq)
        # Complex wave number
        k_c = np.sqrt(rho_eq / k_eq) * omega
        # Surface impedance of sample
        z_s = -1j * z_c * (1 / np.tan(k_c * thick))
        # Normalized specific acoustic impedance
        z_n = z_s / z_0
        # Reflection coefficient
        r = (z_s - z_0) / (z_s + z_0)
        # normal absorption coefficient
        normal_abs = 1 - np.abs(r) ** 2

        return normal_abs - obj_data

    def jca_opt(self, normal_alpha: np.ndarray):
        """
        Use JCA model to compute minimize the L2 norm of predict subtract real, get 5 params
        :param normal_alpha: The real data of absorption coe
        :return: A tuple of 6 result
        tort: Tortuosity [-] range[1-3]
        flow_res: Airflow resistivity [Ns/m4]
        porous: Porosity [-] range [0-1]
        viscous: Viscous characteristic dimension [m] [range 10-1000 10^-6m]
        thermal: Thermal characteristic dimension [m] [range 10-1000 10^-6m]
        thick: thickness [m]
        output: lmfit output object
        """
        assert normal_alpha.shape == (6,)
        # Construct Parameter
        para = lmfit.Parameters()
        para.add('tort', value=2, min=1, max=3)
        para.add('flow_res', value=2e5, min=2e3, max=8e5)
        para.add('porous', value=0.3, min=0, max=1)
        para.add('viscous', value=1e-4, min=1e-6, max=1e-3)
        para.add('thermal', value=1e-4, min=1e-6, max=1e-3)
        para.add('thick', value=0.1, min=0.01, max=1)

        # Frequency to optimize
        freq = np.array([125, 250, 500, 1e3, 2e3, 4e3])

        # Construct optimizer
        output = lmfit.minimize(self.jcaModel_Residual, para, args=(freq,), kws={'obj_data': normal_alpha},
                                method='nelder', nan_policy='omit')

        # Wrap the output
        tort = output.params.valuesdict()['tort']
        flow_res = output.params.valuesdict()['flow_res']
        porous = output.params.valuesdict()['porous']
        viscous = output.params.valuesdict()['viscous']
        thermal = output.params.valuesdict()['thermal']
        thick = output.params.valuesdict()['thick']

        return tort, flow_res, porous, viscous, thermal, thick, output

    @staticmethod
    def dbMiki(flow_res: float, thick: float):
        """
        Use dbMiki model to predict the normal incidence absorption
        :param flow_res: Airflow resistivity [Ns/m4]
        :param thick: thickness [m]
        :return: normal incidence alpha, reflection coefficient
        """
        # DB_Miki Model Core
        freq = np.array([125, 250, 500, 1e3, 2e3, 4e3])
        # air impedance
        z_0 = acoustic_speed * air_rho
        # freq to flow_res ratio
        f_r = 1e3 * (freq / flow_res)
        # angular freq
        omega = 2 * np.pi * freq
        # Characristic impedance
        z_c = z_0 * (1 + 5.5 * f_r ** (-0.632) - 1j * 8.43 * f_r * (-0.623))
        # Complex wave number
        k_c = (omega / acoustic_speed) * (1 + 7.81 * f_r ** (-0.618) - 1j * 11.41 * f_r ** (-0.618))
        # Surface impedance
        z_s = -1j * z_c / np.tan(k_c * thick)
        # Reflection coe
        r = (z_s - z_0) / (z_s + z_0)
        # Normal incidence absorption coefficient
        normal_abs = 1 - (np.abs(r)) ** 2

        return normal_abs, r

    @staticmethod
    def dbMiki_Residual(para: lmfit.Parameters, freq: np.ndarray, obj_data: np.ndarray = None):
        """
        DB Miki model, predict the normal incidence absorption coefficient.
        :param para: A lmfit.Parameters object, with the parameter to be optimized
        :param freq: The frequency of the absorption coefficient
        :param obj_data: The objective absorption coefficient, ** Normal incidence
        :return: The residual of the predict value and truth value
        """
        # Wrap the Objective Value
        val_dict = para.valuesdict()
        flow_res = val_dict["flow_res"]
        thick = val_dict["thick"]

        # DB_Miki Model Core
        # air impedance
        z_0 = acoustic_speed * air_rho
        # freq to flow_res ratio
        f_r = 1e3 * (freq / flow_res)
        # angular freq
        omega = 2 * np.pi * freq
        # Characristic impedance
        z_c = z_0 * (1 + 5.5 * f_r ** (-0.632) - 1j * 8.43 * f_r * (-0.623))
        # Complex wave number
        k_c = (omega / acoustic_speed) * (1 + 7.81 * f_r ** (-0.618) - 1j * 11.41 * f_r ** (-0.618))
        # Surface impedance
        z_s = -1j * z_c / np.tan(k_c * thick)
        # Normal incidence absorption coefficient
        alpha_0 = 1 - (np.abs((z_s - z_0) / (z_s + z_0))) ** 2

        return obj_data - alpha_0

    def dbmiki_opt(self, normal_alpha: np.ndarray):
        """
        Use lmfit optimizer to fit the curve
        :param normal_alpha: the
        :return:
        """
        assert normal_alpha.shape == (6,)
        # Construct Parameter
        para = lmfit.Parameters()
        para.add('flow_res', value=2e5, min=2e3, max=8e5)
        para.add('thick', value=0.1, min=0.01, max=1)

        # Frequency to optimize
        freq = np.array([125, 250, 500, 1e3, 2e3, 4e3])

        # Construct optimizer
        output = lmfit.minimize(self.dbMiki_Residual, para, args=(freq,), kws={'obj_data': normal_alpha},
                                method='nelder', nan_policy='omit')

        # Wrap the output
        flow_res = output.params.valuesdict()['flow_res']
        thick = output.params.valuesdict()['thick']

        return flow_res, thick, output


if __name__ == '__main__':
    mat_1 = Material("wood", np.array([0.26, 0.13, 0.08, 0.06, 0.06, 0.06]), plane_wave_model='JCA')
    print(mat_1.ref_0)
