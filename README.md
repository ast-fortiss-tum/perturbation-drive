# Perturbation Drive

A library to test the robstuness of Self-Driving-Cars via image perturbations.

This library is split up into two sections:

1) A collection of the most common perturbation functions which can be used by itself.
2) A benchmarking class which can be used to benchmark the robustness of models to common image perturbations.

## Installation

You can install the library using pip

```Shell
pip install perturbationdrive
```

## Performing Image Perturbations

Apply common image perturbations and corruptions to images.
Each perturbation needs an input image and the scale of the perturbation as input.
The scale is in the range from 0 to 4.

```Python
from perturbationdrive import poisson_noise

perturbed_image = poisson_noise(image, 0)

```

## Benchmarking Self-Driving Cars

View the examples folder for examples on how to run this benchmarking library with different simulators.

### Create the ImagePerturbation Object

On initialization the perturbation class expects the following parameters:

- `funcs=[]`: A string list of perturbation functions which shall be used for the benchmarking process. If this list is empty, a prefedined set of functions will be used.
- `log_dir="logs.csv"`: The string path to the file which shall be used to log the benchmarking.
- `overwrite_logs=True`: If the old log should be overwriten, if there is already a file at the `log_dir` place.
- `image_size=(240,320)`: The size of the input and output image.
- `drop_boundary=3`: The worst case performance boundary for the used performance measure. If this boundary is exceeded the perturbation will not be run again in a future lap. The default perfirnabce measure is Cross-Track-Error.

```Python
from perturbationdrive import ImagePerturbation

# Instantiate a perturbation object
funcs = [
    "elastic", 
    "defocus_blur",
    "dynamic_snow_filter"
]
perturbation = ImagePerturbation(funcs)
```

### Perturbate the image

The perturbation object expects the image and a dict containing the following entries:

- `lap`: The current lap of the car.
- `sector`: The sector of the track that the car is currenlty in
- `xte`: The Cross-Track-Error of the car which serves as imediate performance measure of the car. If wanted one can
supply another performance measure, however, the performance measure needs to be supplied in the entry `XTE`.
- `pos_x`: The x position of the car.
- `pos_y`: The y position of the car.
- `pos_z`: The z position of the car

The perturbation object will automatically select the perturbation and the perturbation scale based on the lap, position
and past performance.

The `perturbate` function returns a dict containing the perturbated image and an instruction what to do next

- `image`: The perturbated image.
- `func`: The next instruction for the simulator as a string. Possible values are:
  - `reset_car`: Reset the car to the starting position.
  - `quit_app`: The benchmarking process has finished and we need to quit the simulation.
  - `update`: We have perturbated an image which should be supplied to our model and we need to send control parameters to the simulator.

```Python
from perturbationdrive import ImagePerturbation

# Instantiate a perturbation object
perturbation = ImagePerturbation()

# Apply an image perturbation to an incoming object
info_data = {
    "lap": data["lap"],
    "sector": data["sector"],
    "xte": data["cte"],
    "pos_x": data["pos_x"],
    "pos_y": data["pos_x"],
    "pos_z": data["pos_x"],
}
message = perturbation.peturbate(image, info_data)
perturbated_image = message["image"]
instruction = message["func"]

# Print final output of the perturbation benchmark
perturbation.on_stop()
```

### Measure the Steering Angle Performance

If you also want to measure the steering angle difference of your model on the perturbated image and the non-perturbated image you can optionally call the `ImagePerturbation` object and measure this metric.

```Python
# calculate the steering angle difference
diff = abs(steering_angle - unchanged_steering_angle)
# update the metrics
perturbation.udpateSteeringPerformance(diff)
```

## Neural Style Transfer

If you want to perturb your images using neural style transfer based on the ideas and models of `Perceptual Losses for Real-Time Style Transfer and Super- Resolution, Johnson et al., 2016`, you first need to download the models via a setup script.

1) Navigate into the directory via `cd perturbationdrive/NeuralStyleTransfer`
2) Make the setup script executable `chmod +x setup.sh`
3) Execute the setup script `./setup.sh`. This will create the folders `perturbationdrive/NeuralStyleTransfer/models/instance_norm` and `perturbationdrive/NeuralStyleTransfer/models/eccv16` with all relevant Neural Style Transfer models.

Happy Testing!

## Local setup

To set this library up locally, navigate into the folder of this library.

### Contributing and extending this library

Make sure you have all requirements for this library setup.

```Shell
pip install -r requirements.txt
```

### Installing this library locally

```Shell
pip install .
```
