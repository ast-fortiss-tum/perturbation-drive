from perturbationdrive import PerturbationDrive,CustomRoadGenerator
from examples.self_driving_sandbox_donkey.sdsandbox_simulator import SDSandboxSimulator
from examples.models.dave2_agent import Dave2Agent
import traceback
from datetime import datetime




try:
    simulator = SDSandboxSimulator(
        simulator_exe_path="./examples/self_driving_sandbox_donkey/sim/donkey-sim.app",
        host="127.0.0.1", 
        port=9091,
        show_image_cb=True
    )
    ads = Dave2Agent(model_path="./examples/models/checkpoints/dave_90k_v1.h5")
    road_angles = [0, -35, 0, -17, -35, 35, 6, -22]
    road_segments = [25, 25, 25, 25, 25, 25, 25, 25]
    road_generator = CustomRoadGenerator(num_control_nodes=len(road_angles))

    benchmarking_obj = PerturbationDrive(simulator, ads)
    # start the benchmarking
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = ads.model
    attention_map = {
                "map": "grad_cam",
                "model": model,
                "threshold": 0.1,
                "layer": "conv2d_5",
            }
    


    # "dynamic_raindrop_filter","effects_attention_rain",
    benchmarking_obj.grid_seach(
        perturbation_functions=[],
        attention_map = {},
        road_generator=road_generator,
        road_angles=road_angles,
        road_segments=road_segments,
        log_dir=f"./logs/donkey_logs_{time}.json",
        overwrite_logs=True,
        image_size=(240, 320),  # images are resized to these values
    )
    print(f"{5 * '#'} Finished Running SDSandBox Sim {5 * '#'}")
except Exception as e:
    print(
        f"{5 * '#'} SDSandBox Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
    )