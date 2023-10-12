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

The perturbation object will automatically select the perturbation and the perturbation scale.

```Python
from perturbationdrive import ImagePerturbation

# Instantiate a perturbation object
perturbation = ImagePerturbation()

# Apply an image perturbation to an incoming object
image = perturbation.peturbate(image)

# Print final output of the perturbation benchmark
perturbation.on_stop()
```

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
