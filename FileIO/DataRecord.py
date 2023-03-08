import copy
import os
import json
import pickle
import numpy as np
from Acoustic.Receiver import Receiver



def rec_to_json(rec: Receiver, output_folder: str, rec_id: int):
    """

    :param rec:
    :param output_folder:
    :param rec_id:
    :return:
    """
    # Turn the class into a python dict
    rec_dict = rec.to_json()
    exist_state = os.path.exists(output_folder)
    if exist_state:
        rec_json = json.dumps(rec_dict, indent=4, ensure_ascii=False)
        out_file_path = os.path.join(output_folder, f"\\rec_{str(rec_id)}.json")
        with open(out_file_path, 'w', encoding="utf-8") as json_file:
            json.dump(rec_json, json_file)


def rec2pkl(rec: Receiver, output_folder: str, rec_id: int):
    """
    Turn the rec data to .pkl format file
    :param rec: A Receiver Object
    :param output_folder: The output folder direction
    :param rec_id: the id
    :return: A rec_n.pkl file in output folder
    """
    exist_state = os.path.exists(output_folder)
    rec.sound_field = None
    for i in rec.im_ray:
        i.sound_field = None

    if exist_state:
        file_name = f"\\rec_{str(rec_id)}.pkl"
        file_dir = output_folder+file_name
        data_output = open(file_dir, 'wb')
        pickle.dump(rec, data_output)
        data_output.close()
        print(f"File {file_dir} exports successfully!")
