from perturbationdrive import PerturbationDrive,RandomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator
from examples.models.dave2_agent import Dave2Agent
import traceback

try:
    simulator = UdacitySimulator(
        simulator_exe_path="./examples/udacity/sim/udacity_sim.app",
        host="127.0.0.1",
        port=9091,
    )    
    ads = Dave2Agent(model_path="./examples/models/checkpoints/dave_90k_v1.h5")
    road_generator = RandomRoadGenerator(num_control_nodes=8)
    benchmarking_obj = PerturbationDrive(simulator, ads)
    # start the benchmarking
    benchmarking_obj.grid_seach(
        perturbation_functions=["gaussian_noise"],
        attention_map={},
        road_generator=road_generator,
        log_dir="./examples/udacity/logs.json",
        overwrite_logs=True,
        image_size=(240, 320),  # images are resized to these values
    )
    print(f"{5 * '#'} Finished Running Udacity Sim {5 * '#'}")
except Exception as e:
    print(
        f"{5 * '#'} Udacity Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
    )