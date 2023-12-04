import pyperf
import numpy as np
from perturbationdrive import Sim2RealGen, NeuralStyleTransfer
import tensorflow as tf
# we need tfs here because the model uses tfa.layers.InstanceNormalization
# and without the import we cannot compile the model
import tensorflow_addons as tfa
import cv2

# Instantiate classes outside the benchmark function
cycleGenerativeModels = Sim2RealGen()
neuralStyleModels = NeuralStyleTransfer(
    [
        "perturbationdrive/NeuralStyleTransfer/models/instance_norm/candy.t7",
        "perturbationdrive/NeuralStyleTransfer/models/eccv16/composition_vii.t7",
    ]
)


def benchmark(cycleGenerativeModels, neuralStyleModels):
    img = np.random.rand(320, 240, 3) * 255
    img = img.astype(np.uint8)

    # Create a Runner object to manage benchmarks
    runner = pyperf.Runner(values=30, loops=1)

    for i in range(5):
        runner.bench_func(f"sim2real {i}", cycleGenerativeModels.toSim, img)
        runner.bench_func(f"sim2sim {i}", cycleGenerativeModels.sim2sim, img)
        runner.bench_func(f"candy {i}", neuralStyleModels.transferStyle, img, "candy")
        runner.bench_func(
            f"composition_vii {i}",
            neuralStyleModels.transferStyle,
            img,
            "composition_vii",
        )


def benchmark_generative_components(cycleGenerativeModels):
    sim2real = tf.keras.models.load_model(
        "perturbationdrive/Generative/sim2real_e25.h5"
    )
    img = np.random.rand(320, 240, 3) * 255
    img = img.astype(np.uint8)
    (h, w) = img.shape[:2]

    image_tensor_batch = cycleGenerativeModels.preprocess_image(img)
    generated_real = sim2real.predict(image_tensor_batch)
    generated_real = np.clip(generated_real * 255, 0, 255).astype(np.uint8)

    runner = pyperf.Runner(values=30, loops=1)

    for i in range(3):
        runner.bench_func(f"preprocess {i}", cycleGenerativeModels.preprocess_image, img)
        runner.bench_func(f"cycle {i}", sim2real.predict, image_tensor_batch)

if __name__ == "__main__":
    benchmark_generative_components(cycleGenerativeModels)
