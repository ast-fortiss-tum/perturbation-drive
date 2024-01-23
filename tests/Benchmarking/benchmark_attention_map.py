import pyperf
import numpy as np
from perturbationdrive import ImagePerturbation
from tensorflow.keras.models import load_model


def benchmark_style_transfer():
    perturbation_funcs = [
        "gaussian_noise",
    ]

    # create objects to benchmark
    img = np.random.rand(240, 320, 3) * 255
    img = img.astype(np.uint8)

    # Create a Runner object to manage benchmarks
    model = load_model("./examples/models/generated_dave_90k_v4.h5", compile=False)
    model.compile(loss="sgd", metrics=["mse"])
    #attention_map = {
    #    "map": "grad_cam",
    #    "model": model,
    #    "threshold": 0.5,
    #    "layer": "conv2d_5",
    #}
    #perturbation_obj = ImagePerturbation(
    #    funcs=perturbation_funcs, attention_map=attention_map
    #)

    runner = pyperf.Runner(values=10, loops=5)

    #for i in range(5):
    #    for func in perturbation_funcs:
    #        runner.bench_func(
    #            f"{func} attention_map grad_cam scale {i}",
    #            perturbation_obj.perturbation,
    #            img,
    #            func,
    #            i,
    #        )

    attention_map = {
        "map": "vanilla",
        "model": model,
        "threshold": 0.5,
        "layer": "conv2d_5",
    }

    perturbation_obj = ImagePerturbation(
        funcs=perturbation_funcs, attention_map=attention_map
    )

    for i in range(5):
        for func in perturbation_funcs:
            runner.bench_func(
                f"{func} attention_map vanilla scale {i}",
                perturbation_obj.perturbation,
                img,
                func,
                i,
            )


if __name__ == "__main__":
    benchmark_style_transfer()
