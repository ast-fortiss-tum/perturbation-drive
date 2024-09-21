from perturbationdrive import (
    PerturbationDrive,
    CustomRoadGenerator,
    GridSearchConfig,
    RoadGenerationFrequency,
)
from examples.self_driving_sandbox_donkey.sdsandbox_simulator import SDSandboxSimulator
from examples.models.dave2_agent import Dave2Agent
import traceback
from datetime import datetime


try:
    simulator = SDSandboxSimulator(
        simulator_exe_path="./examples/self_driving_sandbox_donkey/sim/donkey-sim.app",
        host="127.0.0.1",
        port=9091,
        show_image_cb=True,
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

    perturbations = [
        "gaussian_noise",
        "poisson_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "increase_brightness",
        "contrast",
        "elastic",
        "pixelate",
        "jpeg_filter",
        "translate_image",
        "scale_image",
        "splatter_mapping",
        "dotted_lines_mapping",
        "zigzag_mapping",
        "canny_edges_mapping",
        "speckle_noise_filter",
        "false_color_filter",
        "high_pass_filter",
        "low_pass_filter",
        "phase_scrambling",
        "histogram_equalisation",
        "reflection_filter",
        "white_balance_filter",
        "sharpen_filter",
        "grayscale_filter",
        "fog_filter",
        "frost_filter",
        "snow_filter",
        "posterize_filter",
        "cutout_filter",
        "sample_pairing_filter",
        "gaussian_blur",
        "saturation_filter",
        "saturation_decrease_filter",
        "candy",
        "la_muse",
        "mosaic",
        "feathers",
        "the_scream",
        "udnie",
        "effects_attention_rain",
        "effects_attention_rain_dynamic",
        "effects_rain_dynamic",
        # "dynamic_snow_filter",
        # "dynamic_rain_filter",
        # "dynamic_object_overlay",
        # "dynamic_sun_filter",
        # "dynamic_lightning_filter",
        # "dynamic_smoke_filter",
        # "static_snow_filter",
        # "static_rain_filter",
        # "static_object_overlay",
        # "static_sun_filter",
        # "static_lightning_filter",
        # "static_smoke_filter",
    ]
    config = GridSearchConfig(
        perturbation_functions=perturbations,
        attention_map=attention_map,
        road_generator=road_generator,
        road_angles=road_angles,
        road_segments=road_segments,
        road_generation_frequency=RoadGenerationFrequency.ONCE,
        log_dir=f"./logs/donkey_logs_{time}.json",
        overwrite_logs=True,
        image_size=(240, 320),  # images are resized to these values
        drop_perturbation=(lambda outcome: (not outcome.isSuccess) or outcome.timeout),
        increment_perturbation_scale=(lambda outcomes: True),
    )
    # "dynamic_raindrop_filter","effects_attention_rain",
    benchmarking_obj.grid_seach(config=config)
    print(f"{5 * '#'} Finished Running SDSandBox Sim {5 * '#'}")
except Exception as e:
    print(
        f"{5 * '#'} SDSandBox Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
    )
