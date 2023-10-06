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
        runner.bench_func(f"fog_filter scale {i}", fog_filter, i, img)
        runner.bench_func(f"frost_filter scale {i}", frost_filter, i, img)
        runner.bench_func(f"snow_filter scale {i}", snow_filter, i, img)
        runner.bench_func(f"object_overlay scale {i}", object_overlay, i, img)


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
