from perturbationdrive import (
    PerturbationDrive,
    CustomRoadGenerator,
    GridSearchConfig,
    RoadGenerationFrequency,
)
from examples.udacity.udacity_simulator import UdacitySimulator
from examples.models.dave2_agent import Dave2Agent
import traceback
from datetime import datetime

try:
    simulator = UdacitySimulator(
        simulator_exe_path="./examples/udacity/sim/udacity_sim.app",
        host="127.0.0.1",
        port=9091,
    )
    ads = Dave2Agent(model_path="./examples/models/checkpoints/dave_90k_v1.h5")

    road_angles = [10, 10, 10, 10, 0, -10, -10, -10]
    road_segments = [10, 10, 10, 10, 10, 10, 10, 10]
    road_generator = CustomRoadGenerator(num_control_nodes=len(road_angles))

    benchmarking_obj = PerturbationDrive(simulator, ads)
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = ads.model
    attention_map = {
        "map": "grad_cam",
        "model": model,
        "threshold": 0.1,
        "layer": "conv2d_5",
    }
    config = GridSearchConfig(
        perturbation_functions=["gaussian_noise"],
        attention_map={},
        road_generator=road_generator,
        road_angles=road_angles,
        road_segments=road_segments,
        road_generation_frequency=RoadGenerationFrequency.ONCE,
        log_dir=f"./logs/udacity_logs_{time}.json",
        overwrite_logs=True,
        image_size=(240, 320),
    )
    # start the benchmarking
    benchmarking_obj.grid_seach(config=config)
    print(f"{5 * '#'} Finished Running Udacity Sim {5 * '#'}")
except Exception as e:
    print(
        f"{5 * '#'} Udacity Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
    )
