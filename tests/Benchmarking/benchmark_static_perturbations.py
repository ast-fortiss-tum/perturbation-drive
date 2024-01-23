import pyperf
import numpy as np
from perturbationdrive import ImagePerturbation


def benchmark_static():
    perturbation_funcs = [
        "static_snow_filter",
        "static_rain_filter",
        "static_object_overlay",
        "static_sun_filter",
        "static_lightning_filter",
        "static_smoke_filter",
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
    benchmark_static()
