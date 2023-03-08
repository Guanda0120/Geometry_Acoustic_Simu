from Acoustic.Material import Material


class MaterialDict:
    """

    """

    def __init__(self, abs_dict: dict, sca_dict: dict, **kwargs):
        """

        :param abs_dict:
        :param sca_dict:
        """
        self.material_dict = {}
        self.model_name = kwargs['model_name']
        self.opt_max_res = kwargs['opt_max_res']

        for key in abs_dict.keys():
            self.material_dict[key] = Material(key, abs_dict[key], sca_dict[key], plane_wave_model=self.model_name,
                                               opt_max_res=self.opt_max_res)


if __name__ == '__main__':
    from CONFIG import absorptionDict, scatterDict, opt_max_res, model_name
    mat_dict = MaterialDict(absorptionDict, scatterDict, model_name=model_name, opt_max_res=opt_max_res)
    print(mat_dict.material_dict)
