from perturbationdrive import PerturbationDrive,CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator
from examples.models.dave2_agent import Dave2Agent
import traceback
from datetime import datetime

try:
    simulator = UdacitySimulator(
        simulator_exe_path="./examples/udacity/sim/udacity_sim_weather_ready.app",
        host="127.0.0.1",
        port=9091
    )    
    ads = Dave2Agent(model_path="/PerturbationDrive/checkpoints/trained_low.h5")
    
    road_angles=[10,20,30,40,0,-10,-20,-30,0,10,40,10,0,-10,-40,-10]
    road_segments=[10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    road_generator = CustomRoadGenerator(num_control_nodes=len(road_angles))

    # "Sun", "Rain", "Snow", "Fog"

    benchmarking_obj = PerturbationDrive(simulator, ads)
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = ads.model
    attention_map = {
                "map": "grad_cam",
                "model": model,
                "threshold": 0.1,
                "layer": "conv2d_5",
            }
    
    weather = "Sun"
    intens=20
    # start the benchmarking
    benchmarking_obj.grid_seach(
        perturbation_functions=[],
        attention_map={},
        road_generator=road_generator,
        road_angles=road_angles,
        road_segments=road_segments,
        log_dir=f"./logs/udacity_logs_{time}.json",
        overwrite_logs=True,
        image_size=(240, 320),
        test_model=True,
        perturb=False,
        weather=weather,
        weather_intensity=intens
    )
    print(f"{5 * '#'} Finished Running Udacity Sim {5 * '#'}")
except Exception as e:
    print(
        f"{5 * '#'} Udacity Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
    )