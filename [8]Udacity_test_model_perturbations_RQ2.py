from perturbationdrive import PerturbationDrive,CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator
from examples.models.dave2_agent import Dave2Agent
import traceback
from datetime import datetime
import random

road_angles_list=[]
road_segments_list=[]
road_types_list=[]

def generate_random_roads(angles_number=20):
    road_angles_list = []
    road_segments_list = []
    road_types_list=[]
    # for i in range(10, 1, -1):
    i=7
    road_angles = [random.randint(-2 * i, 2 * i) for _ in range(angles_number)]
    road_segments = [random.randint(20, 30) for _ in range(angles_number)]
    road_angles_list.append(road_angles)
    road_segments_list.append(road_segments)
    road_types_list.append("random")
    if i != 0:
        road_angles = [random.randint(-4 * i, 4 * i) for _ in range(angles_number)]
        road_segments = [random.randint(20, 30) for _ in range(angles_number)]
        road_angles_list.append(road_angles)
        road_segments_list.append(road_segments)
        road_types_list.append("random")
    
        road_angles = [random.randint(-6 * i, 6 * i) for _ in range(angles_number)]
        road_segments = [random.randint(20, 30) for _ in range(angles_number)]
        road_angles_list.append(road_angles)
        road_segments_list.append(road_segments)
        road_types_list.append("random")
    return road_angles_list, road_segments_list, road_types_list

road_angles_list,road_segments_list,road_types_list=generate_random_roads()
print(len(road_angles_list))



try:
    i=1
    simulator = UdacitySimulator(
        simulator_exe_path="./examples/udacity/sim/udacity_sim_weather_sky_ready_angles.app",
        host="127.0.0.1",
        port=9091
    )    
    road_generator = CustomRoadGenerator(num_control_nodes=len(road_angles_list[i]))
    ads = Dave2Agent(model_path=" /PerturbationDrive/checkpoints/original_extended3.h5")
    model = ads.model
    attention_map = {
        "map": "grad_cam",
        "model": model,
        "threshold": 0.1,
        "layer": "conv2d_5",
    }
    functions=['canny_edges_mapping', 'contrast', 'cutout_filter', 'defocus_blur', 'dotted_lines_mapping', 'dynamic_lightning_filter', 'dynamic_object_overlay', 'dynamic_rain_filter', 'dynamic_raindrop_filter', 'dynamic_smoke_filter', 'dynamic_snow_filter', 'dynamic_sun_filter', 'effects_attention_regions', 'elastic', 'fog_filter', 'frost_filter', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'grayscale_filter', 'histogram_equalisation', 'impulse_noise', 'increase_brightness', 'jpeg_filter', 'low_pass_filter', 'motion_blur', 'object_overlay',  'phase_scrambling', 'pixelate', 'poisson_noise', 'posterize_filter', 'reflection_filter', 'rotate_image', 'sample_pairing_filter', 'saturation_decrease_filter', 'saturation_filter', 'scale_image', 'sharpen_filter', 'shear_image' '']
    print("All: ", len(functions))
    functions_done=[]
    functions_todo = [element for element in functions if element not in functions_done]
    print("Remaining: ",len(functions_todo)," names: ", functions_todo)
    



    benchmarking_obj = PerturbationDrive(simulator, ads)
    print(f"{5 * '#'} Testing road {i} {5 * '#'}")
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    benchmarking_obj.grid_seach(
        perturbation_functions=functions,
        attention_map=attention_map,
        road_generator=road_generator,
        road_angles=road_angles_list[i],
        road_segments=road_segments_list[i],
        log_dir=f"./output_logs/test_perturbation_extended3/udacity_road_{road_types_list[i]}_{i}_logs_{time}.json",
        overwrite_logs=True,
        image_size=(240, 320),
        test_model=True,
        collect_train_data=True,
        perturb=True,
    )
    print(f"{5 * '#'} Finished Testing road {i} {5 * '#'}")
except Exception as e:
    print(
        f"{5 * '#'} Udacity Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
    )