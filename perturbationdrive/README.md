# PerturbationDrive Docs

This ReadMe provides documentation over all functionalities in the perturbation drive folder.

## Table of Contents

- [Image Perturbations](#image-perturbations)
  - [ImagePerturbation Controller](#imageperturbation-controller)
  - [Example](#image-perturbation-example)
- [PerturbationDrive Controller](#perturbationdrive-controller)
- [Simulator](#simulator)
  - [Scenario](#scenario)
  - [ScenarioOutcome](#scenariooutcome)
- [Automated Driving System](#ads)

## Image Perturbations

The file `perturbationfuncs.py` implements all image perturbations used in this work. Note, that this library uses [OpenCV Python](https://pypi.org/project/opencv-python/) for performing perturbations and hence each method expects an image per OpenCV specification. This is an image with 3 color channels and the dtype `uint8`.
Each perturbation needs an input image and the scale of the perturbation as input. The scale is in the range from 0 to 4. The following table details all perturbations of this library.

| Function Name | Description |
| --------------- | --------------- |
| Row 1, Column 1 | Row 1, Column 2 |
| Row 2, Column 1 | Row 2, Column 2 |
| Row 3, Column 1 | Row 3, Column 2 |

### ImagePerturbation Controller

The class `ImagePerturbation` provides a class interface for performing perturbations on images. This is also the controller used in this framework to provide easy to access perturbations. Note, that this class also provides access to the more advanced perturbations, such as Dynamic Perturbations, Generative Perturbations and Perturbations based on the Attention Map.

By creating a subclass, one can extend the perturbations used in this library.

### Image Perturbation Example

```Python
from perturbationdrive import poisson_noise, gaussian_noise
import cv2

height, width = 300, 300
random_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

poisson_img = poisson_noise(image, 0)
cv2.imshow(poisson_img)

gaussian_img = gaussian_noise(image, 4)
cv2.imshow(gaussian_img)

# used the controller for perturbation

# perturb the image based on the attention map


```

## PerturbationDrive Controller

## Simulator

### Scenario

### ScenarioOutcome

## ADS
