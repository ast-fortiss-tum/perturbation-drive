from perturbationdrive import PerturbationDrive,CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator
from examples.models.dave2_agent import Dave2Agent
import traceback
from datetime import datetime
import random

road_angles_list=[]
road_segments_list=[]
road_types_list=[]


road_angles_list=[[-37, 44, 53, -60, 69, 17, -1, -32, -43, 57], [-52, 41, -20, 21, 63, 50, -22, -29, 8, -6], [8, 27, -35, -2, -28, -2, -13, 49, 2, 11], [35, -20, -23, 9, -24, -14, 43, 26, -23, -3], [9, -9, -5, -26, -28, -20, 9, -27, 5, -34], [36, 32, 28, 24, 20, 16, 12, 8, 4, 0], [-36, -32, -28, -24, -20, -16, -12, -8, -4, 0], [28, 24, 20, 16, 12, 8, 4, 0, -4, -8], [-28, -24, -20, -16, -12, -8, -4, 0, 4, 8], [35, 35, 35, 35, 35, 35, 35, 35], [-35, -35, -35, -35, -35, -35, -35, -35], [25, 25, 25, 25, 25, 25, 25, 25], [-25, -25, -25, -25, -25, -25, -25, -25], [45, -45, 45, -45, 45, -45, 45, -45, 45, -45], [35, -35, 35, -35, 35, -35, 35, -35, 35, -35]]
road_segments_list=[[20, 22, 29, 24, 23, 21, 26, 27, 21, 20], [20, 26, 21, 21, 23, 20, 23, 29, 24, 27], [27, 28, 22, 20, 30, 25, 23, 23, 29, 26], [22, 29, 26, 27, 30, 27, 24, 25, 20, 27], [25, 21, 28, 20, 30, 21, 26, 29, 25, 30], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [10, 10, 10, 10, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10, 10, 10], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]]
road_types_list=['random', 'random', 'random', 'random', 'random', 'curvy', 'curvy', 'curvy', 'curvy', 'simple', 'simple', 'simple', 'simple', 'swiggly', 'swiggly']
model_paths=[" s/PerturbationDrive/checkpoints/original_extended3_extended5.h5"]

try:
    for model_path in model_paths:
        for i in range(len(road_angles_list)):
                    simulator = UdacitySimulator(
                        simulator_exe_path="./examples/udacity/sim/udacity_sim_weather_sky_ready_angles.app",
                        host="127.0.0.1",
                        port=9091
                    )    
                    road_generator = CustomRoadGenerator(num_control_nodes=len(road_angles_list[i]))
                    
                    ads = Dave2Agent(model_path=model_path)
                    model = ads.model
                    attention_map = {
                        "map": "grad_cam",
                        "model": model,
                        "threshold": 0.1,
                        "layer": "conv2d_5",
                    }
                    model_name=model_path.split("/")[-1].split(".h5")[0]
                    benchmarking_obj = PerturbationDrive(simulator, ads)
                    print(f"{5 * '#'} Testing road {i} {5 * '#'}")
                    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    benchmarking_obj.grid_seach(
                        perturbation_functions=[],
                        attention_map=attention_map,
                        road_generator=road_generator,
                        road_angles=road_angles_list[i],
                        road_segments=road_segments_list[i],
                        log_dir=f"./output_logs/test_nominal_{model_name}/udacity_road_{road_types_list[i]}_{i}_logs_{time}.json",
                        overwrite_logs=True,
                        image_size=(240, 320),
                        test_model=True,
                        perturb=False
                    )
                    print(f"{5 * '#'} Finished Testing road {i} {5 * '#'}")
except Exception as e:
    print(
        f"{5 * '#'} Udacity Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
    )