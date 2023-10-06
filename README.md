# Perturbation Drive

A library to test the robstuness of Self-Driving-Cars via image perturbations

## Installation

You can install the library using pip

```Shell
pip install perturbationdrive
```

## Get started

View the examples folder for examples on how to run this benchmarking library with different simulators.

The perturbation object wukk automatically select the perturbation and the perturbation scale.

```Python
from perturbationdrive import ImagePerturbation

# Instantiate a perturbation object
perturbation = ImagePerturbation()

# Apply an image perturbation to an incoming object
image = perturbation.peturbate(image)

# Print final output of the perturbation benchmark
perturbation.on_stop()
```
