import pyperf
import numpy as np
from perturbationdrive import ImagePerturbation


def benchmark_demo():
    # create objects to benchmark
    img = img = np.random.rand(256, 256, 3) * 255
    perturbation = ImagePerturbation(scale=0)

    # Create a Runner object to manage benchmarks
    runner = pyperf.Runner(values=10)

    for i in range(5):
        runner.bench_func(f"gaussian_noise scale {i}", perturbation.gaussian_noise, img)
        perturbation.increment_scale()


def benchmark():
    # create objects to benchmark
    img = img = np.random.rand(256, 256, 3) * 255
    # Typically, images should be numpy arrays with a dtype of uint8 for most common operations.
    img = img.astype(np.uint8)
    perturbation = ImagePerturbation(scale=0)

    # Create a Runner object to manage benchmarks
    runner = pyperf.Runner(values=10)

    for i in range(5):
        runner.bench_func(f"gaussian_noise scale {i}", perturbation.gaussian_noise, img)
        runner.bench_func(f"poisson_noise scale {i}", perturbation.poisson_noise, img)
        runner.bench_func(f"impulse_noise scale {i}", perturbation.impulse_noise, img)
        runner.bench_func(f"defocus_blur scale {i}", perturbation.defocus_blur, img)
        runner.bench_func(f"glass_blur scale {i}", perturbation.glass_blur, img)
        runner.bench_func(f"motion_blur scale {i}", perturbation.motion_blur, img)
        runner.bench_func(f"zoom_blur scale {i}", perturbation.zoom_blur, img)
        runner.bench_func(
            f"increase_brightness scale {i}", perturbation.increase_brightness, img
        )
        runner.bench_func(f"contrast scale {i}", perturbation.contrast, img)
        runner.bench_func(f"elastic scale {i}", perturbation.elastic, img)
        runner.bench_func(f"pixelate scale {i}", perturbation.pixelate, img)
        runner.bench_func(f"jpeg_filter scale {i}", perturbation.jpeg_filter, img)
        runner.bench_func(f"fog_filter scale {i}", perturbation.fog_filter, img)
        runner.bench_func(f"frost_filter scale {i}", perturbation.frost_filter, img)
        runner.bench_func(f"snow_filter scale {i}", perturbation.snow_filter, img)
        runner.bench_func(f"object_overlay scale {i}", perturbation.object_overlay, img)

        perturbation.increment_scale()


def benchmark_donkey_image_shape():
    # create objects to benchmark
    img = img = np.random.rand(120, 160, 3) * 255
    # Typically, images should be numpy arrays with a dtype of uint8 for most common operations.
    img = img.astype(np.uint8)
    perturbation = ImagePerturbation(scale=0)

    # Create a Runner object to manage benchmarks
    runner = pyperf.Runner(values=10)

    for i in range(5):
        runner.bench_func(f"gaussian_noise scale {i}", perturbation.gaussian_noise, img)
        runner.bench_func(f"poisson_noise scale {i}", perturbation.poisson_noise, img)
        runner.bench_func(f"impulse_noise scale {i}", perturbation.impulse_noise, img)
        runner.bench_func(f"defocus_blur scale {i}", perturbation.defocus_blur, img)
        runner.bench_func(f"glass_blur scale {i}", perturbation.glass_blur, img)
        runner.bench_func(f"motion_blur scale {i}", perturbation.motion_blur, img)
        runner.bench_func(f"zoom_blur scale {i}", perturbation.zoom_blur, img)
        runner.bench_func(
            f"increase_brightness scale {i}", perturbation.increase_brightness, img
        )
        runner.bench_func(f"contrast scale {i}", perturbation.contrast, img)
        runner.bench_func(f"elastic scale {i}", perturbation.elastic, img)
        runner.bench_func(f"pixelate scale {i}", perturbation.pixelate, img)
        runner.bench_func(f"jpeg_filter scale {i}", perturbation.jpeg_filter, img)
        runner.bench_func(f"fog_filter scale {i}", perturbation.fog_filter, img)
        runner.bench_func(f"frost_filter scale {i}", perturbation.frost_filter, img)
        runner.bench_func(f"snow_filter scale {i}", perturbation.snow_filter, img)
        runner.bench_func(f"object_overlay scale {i}", perturbation.object_overlay, img)

        perturbation.increment_scale()


if __name__ == "__main__":
    """
    Creates a benchmark for all image perturbations in all severities
    """
    benchmark_donkey_image_shape()
