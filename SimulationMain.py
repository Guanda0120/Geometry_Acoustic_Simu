import time
from FileIO.ReadRhino import Read3dm
from RoomGeometry.FieldGeometry import FieldGeometry
from Acoustic.RayGroup import RayGroup
from Acoustic.ReceiverGroup import ReceiverGroup
from Acoustic.Source import Source
from Acoustic.MaterialDict import MaterialDict
from CONFIG import absorptionDict, scatterDict, fileDir, init_swl, num_ray, rt_method, receiver_radius, end_rt_spl, \
    im_order, output_folder, sample_freq, opt_max_res, model_name
from FileIO.ConvAudio import conv_audio, save_ir
from FileIO.ExportData import ExportData

print("===================== Pre Process the Material =====================")
start_time = time.time()
material_dict = MaterialDict(absorptionDict, scatterDict, opt_max_res=opt_max_res, model_name=model_name)
end_time = time.time()
print(f"===================== Pre Process Use {round((end_time - start_time), 2)}s =====================")

start_time = time.time()
# Read the rhino file, and convert it
rh_model = Read3dm(fileDir)
# Get the rh_plane, receiver and source
rh_plane, source_location, receiver_location, correct_line = rh_model.convert_file()

# Init a sound field
sound_field = FieldGeometry(rh_plane, material_dict, correct_line)
ray_group = RayGroup(num_ray, init_swl, source_location, sound_field)

# Init a Source and Receiver
source = Source(source_location, init_swl)
receiver_group = ReceiverGroup(receiver_location, source, sound_field, receiver_radius)
end_time = time.time()
print("===================== Initialize The Simulation =====================")
print(f"===================== Time Duration {round((end_time - start_time), 2)}s =====================")

# Direc Sound
start_time = time.time()
receiver_group.predict_direc()
end_time = time.time()
print(f"===================== Direct Sound Predict Finish =====================")
print(f"===================== Time Duration {round((end_time - start_time), 2)}s =====================")

# Image Source Method
start_time = time.time()
receiver_group.image_source_all(im_order)
receiver_group.image_diffuse_all()

end_time = time.time()
print(f"===================== Image Source Method Finish =====================")
print(f"===================== Time Duration {round((end_time - start_time), 2)}s =====================")

# Stochastic Process
start_time = time.time()
# Get the first iter
reflect_order = 0
receiver_group.predict_stochastic(ray_group)
# print(f"Stochastic Process Pressure is {receiver_group.receiver_container[0].stochastic_pressure}")
max_spl = ray_group.reflect_all()
diffuse_num_list = receiver_group.predict_diffuse(ray_group)
print(f"=======In the {reflect_order}th reflection, The Max SPL is {round(max_spl, 2)} dB=======")
reflect_order += 1

# T60
start_pressure = max_spl
while max_spl > (start_pressure - end_rt_spl):
    receiver_group.predict_stochastic(ray_group)
    max_spl = ray_group.reflect_all()
    diffuse_num_list = receiver_group.predict_diffuse(ray_group)
    print(f"=======In the {reflect_order}th reflection, The Max SPL is {round(max_spl, 2)} dB=======")
    reflect_order += 1

# From stochastic ray to back image source ray
# receiver_group.sto2img_all(im_order)

end_time = time.time()
print(f"===================== Stochastic and Diffuse Finish =====================")
print(f"===================== Time Duration {round((end_time - start_time), 2)}s =====================")

receiver_group.merge_sort()
print(f"===================== Merge Sort Finish =========================")
receiver_group.fsmatch_all(sample_freq)
inputfile = r'C:\\Users\\12748\\Desktop\\1.wav'
outputfile = r'C:\\Users\\12748\\Desktop\\Conv.wav'
save_ir(receiver_group.receiver_container[0], r'C:\\Users\\12748\\Desktop\\Impulse_Res.wav')
conv_audio(receiver_group.receiver_container[0], inputfile, outputfile)
print(f"===================== Fs Match Finish =========================")
receiver_group.rt_estimate(rt_method)

for tmp_rec in receiver_group.receiver_container:
    print(f"RT Time by LR Method is {tmp_rec.rt_time_lr}")
    print(f"RT Time by BI Method is {tmp_rec.rt_time_bi}")

export_data = ExportData(receiver_group, output_folder)
export_data.save_plot()
export_data.save_pkl()
print(f"The Simulation Process Finished")
