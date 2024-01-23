import pyperf
import numpy as np
from perturbationdrive import ImagePerturbation


def benchmark_normal():
    perturbation_funcs = [
        "gaussian_noise",
        "poisson_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "increase_brightness",
        "contrast",
        "elastic",
        "pixelate",
        "jpeg_filter",
        "shear_image",
        "translate_image",
        "scale_image",
        "rotate_image",
        "fog_mapping",
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
    ]

    # create objects to benchmark
    img = np.random.rand(320, 240, 3) * 255
    img = img.astype(np.uint8)

    # Create a Runner object to manage benchmarks
    perturbation_obj = ImagePerturbation(funcs=perturbation_funcs)

    runner = pyperf.Runner(values=10, loops=5)

    for i in range(5):
        for func in perturbation_funcs:
            runner.bench_func(
                f"{func} scale {i}", perturbation_obj.perturbation, img, func, i
            )


if __name__ == "__main__":
    """
    Creates a benchmark for all image perturbations in all severities
    """
    benchmark_normal()
