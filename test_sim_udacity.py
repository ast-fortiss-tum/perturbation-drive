from perturbationdrive import PerturbationDrive,CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator
from examples.models.dave2_agent import Dave2Agent
import traceback
from datetime import datetime

try:
    simulator = UdacitySimulator(
        # TODO change path depending on OS
        simulator_exe_path="./examples/udacity/sim/udacity_linux/udacity_binary.x86_64",
        host="127.0.0.1",
        port=9091
    )    
    ads = Dave2Agent(model_path="./examples/models/checkpoints/dave_90k_v1.h5")
    
    road_angles=[10,10,10,10,0,-10,-10,-10]
    road_segments=[10,10,10,10,10,10,10,10]
    road_generator = CustomRoadGenerator(num_control_nodes=len(road_angles))



    benchmarking_obj = PerturbationDrive(simulator, ads)
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = ads.model
    
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
        # "candy",
        # "la_muse",
        # "mosaic",
        # "feathers",
        # "the_scream",
        # "udnie",
        # "effects_attention_rain",
        # "effects_attention_rain_dynamic",
        # "effects_rain_dynamic",
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
        # "dynamic_raindrop_filter",
        # "effects_attention_rain",
    ]

    # start the benchmarking
    print("WILL RUN GAUSSIAN NOISE")
    benchmarking_obj.grid_seach(
        perturbation_functions=perturbations,
        attention_map={},
        road_generator=road_generator,
        road_angles=road_angles,
        road_segments=road_segments,
        log_dir=f"./udacity_logs_{time}.json",
        overwrite_logs=True,
        image_size=(240, 320),  # images are resized to these values
        test_model=False,  #choose to run NPC or model
        perturb=True,  
    )
    print(f"{5 * '#'} Finished Running Udacity Sim {5 * '#'}")
except Exception as e:
    print(
        f"{5 * '#'} Udacity Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
    )