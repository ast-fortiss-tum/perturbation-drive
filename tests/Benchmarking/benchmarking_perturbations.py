import pyperf
import numpy as np
from perturbationdrive.imageperturbations import *


def benchmark_demo():
    # create objects to benchmark
    img = img = np.random.rand(256, 256, 3) * 255
    perturbation = ImagePerturbation(scale=0)

    # Create a Runner object to manage benchmarks
    runner = pyperf.Runner(values=10)

    for i in range(5):
        runner.bench_func(f"gaussian_noise scale {i}", gaussian_noise, 1, img)
        perturbation.increment_scale()


def benchmark():
    # create objects to benchmark
    img = img = np.random.rand(256, 256, 3) * 255
    # Typically, images should be numpy arrays with a dtype of uint8 for most common operations.
    img = img.astype(np.uint8)

    # Create a Runner object to manage benchmarks
    runner = pyperf.Runner(values=30, loops=10)

    for i in range(5):
        runner.bench_func(f"gaussian_noise scale {i}", gaussian_noise, i, img)
        runner.bench_func(f"poisson_noise scale {i}", poisson_noise, i, img)
        runner.bench_func(f"impulse_noise scale {i}", impulse_noise, i, img)
        runner.bench_func(f"defocus_blur scale {i}", defocus_blur, i, img)
        runner.bench_func(f"glass_blur scale {i}", glass_blur, i, img)
        runner.bench_func(f"motion_blur scale {i}", motion_blur, i, img)
        runner.bench_func(f"zoom_blur scale {i}", zoom_blur, i, img)
        runner.bench_func(f"increase_brightness scale {i}", increase_brightness, i, img)
        runner.bench_func(f"contrast scale {i}", contrast, i, img)
        runner.bench_func(f"elastic scale {i}", elastic, i, img)
        runner.bench_func(f"pixelate scale {i}", pixelate, i, img)
        runner.bench_func(f"jpeg_filter scale {i}", jpeg_filter, i, img)
        runner.bench_func(f"shear_image scale {i}", shear_image, i, img)
        runner.bench_func(f"translate_image scale {i}", translate_image, i, img)
        runner.bench_func(f"scale_image scale {i}", scale_image, i, img)
        runner.bench_func(f"rotate_image scale {i}", rotate_image, i, img)
        runner.bench_func(f"fog_mapping scale {i}", fog_mapping, i, img)
        runner.bench_func(f"splatter_mapping scale {i}", splatter_mapping, i, img)
        runner.bench_func(
            f"dotted_lines_mapping scale {i}", dotted_lines_mapping, i, img
        )
        runner.bench_func(f"zigzag_mapping scale {i}", zigzag_mapping, i, img)
        runner.bench_func(f"canny_edges_mapping scale {i}", canny_edges_mapping, i, img)
        runner.bench_func(
            f"speckle_noise_filter scale {i}", speckle_noise_filter, i, img
        )
        runner.bench_func(f"false_color_filter scale {i}", false_color_filter, i, img)
        runner.bench_func(f"high_pass_filter scale {i}", high_pass_filter, i, img)
        runner.bench_func(f"low_pass_filter scale {i}", low_pass_filter, i, img)
        runner.bench_func(f"phase_scrambling scale {i}", phase_scrambling, i, img)
        runner.bench_func(
            f"histogram_equalisation scale {i}", histogram_equalisation, i, img
        )
        runner.bench_func(f"reflection_filter scale {i}", reflection_filter, i, img)
        runner.bench_func(
            f"white_balance_filter scale {i}", white_balance_filter, i, img
        )
        runner.bench_func(f"sharpen_filter scale {i}", sharpen_filter, i, img)
        runner.bench_func(f"grayscale_filter scale {i}", grayscale_filter, i, img)
        runner.bench_func(f"posterize_filter scale {i}", posterize_filter, i, img)
        runner.bench_func(f"cutout_filter scale {i}", cutout_filter, i, img)
        runner.bench_func(
            f"sample_pairing_filter scale {i}", sample_pairing_filter, i, img
        )
        runner.bench_func(f"gaussian_blur scale {i}", gaussian_blur, i, img)
        runner.bench_func(f"saturation_filter scale {i}", saturation_filter, i, img)
        runner.bench_func(
            f"saturation_decrease_filter scale {i}", saturation_decrease_filter, i, img
        )
        runner.bench_func(f"fog_filter scale {i}", fog_filter, i, img)
        runner.bench_func(f"frost_filter scale {i}", frost_filter, i, img)
        runner.bench_func(f"snow_filter scale {i}", snow_filter, i, img)


def benchmark_donkey_image_shape():
    # create objects to benchmark
    img = img = np.random.rand(120, 160, 3) * 255
    # Typically, images should be numpy arrays with a dtype of uint8 for most common operations.
    img = img.astype(np.uint8)
    perturbation = ImagePerturbation(scale=0)

    # Create a Runner object to manage benchmarks
    runner = pyperf.Runner(values=10)

    for i in range(5):
        runner.bench_func(f"gaussian_noise scale {i}", gaussian_noise, i, img)
        runner.bench_func(f"poisson_noise scale {i}", poisson_noise, i, img)
        runner.bench_func(f"impulse_noise scale {i}", impulse_noise, i, img)
        runner.bench_func(f"defocus_blur scale {i}", defocus_blur, i, img)
        runner.bench_func(f"glass_blur scale {i}", glass_blur, i, img)
        runner.bench_func(f"motion_blur scale {i}", motion_blur, i, img)
        runner.bench_func(f"zoom_blur scale {i}", zoom_blur, i, img)
        runner.bench_func(f"increase_brightness scale {i}", increase_brightness, i, img)
        runner.bench_func(f"contrast scale {i}", contrast, i, img)
        runner.bench_func(f"elastic scale {i}", elastic, i, img)
        runner.bench_func(f"pixelate scale {i}", pixelate, i, img)
        runner.bench_func(f"jpeg_filter scale {i}", jpeg_filter, i, img)
        runner.bench_func(f"fog_filter scale {i}", fog_filter, i, img)
        runner.bench_func(f"frost_filter scale {i}", frost_filter, i, img)
        runner.bench_func(f"snow_filter scale {i}", snow_filter, i, img)
        runner.bench_func(f"object_overlay scale {i}", object_overlay, i, img)

        perturbation.increment_scale()


if __name__ == "__main__":
    """
    Creates a benchmark for all image perturbations in all severities
    """
    benchmark()
