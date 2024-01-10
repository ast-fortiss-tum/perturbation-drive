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
| gaussian_noise          | Statistical noise having the probability density function of the normal distribution     |
| poisson_noise           | Statistical noise having the probability density function of the Poisson distribution     |
| impulse_noise           | Random, sharp, and sudden disturbances, taking the form of scattered bright or dark pixel     |
| defocus_blur            | Simulates the effect of the lens being out of focus via circular disc-shaped kernels     |
| glass_blur              | Simulates the effect of viewing an image through a frosted glass     |
| motion_blur             | Simulates of streaking effect in one direction of the image     |
| zoom_blur               | Simulates a radial blur which emanates from a central point of the image     |
| increase_brightness     | Simulates increased brightness by altering the images value channel     |
| contrast                | Increases the difference in luminance on the image     |
| elastic                 | Moves each image pixel by a random offset derived from a Gaussian distribution     |
| pixelate                | Divides the image into square regions and all pixels in a region get assigned the average pixel value of the region     |
| jpeg_filter             | JPEG compression artifacts     |
| shear_image             | Horizontally replaces each point if a fixed direction by an amount proportional to its signed distance from a given line parallel to that direction     |
| translate_image         | Moves every pixel of the image by the same distance into a certain direction     |
| scale_image             | Increases or decreases the size of an image by a certain factor     |
| rotate_image            | Rotates the image by a certain angle in the euclidean space     |
| stripe_mapping          | a     |
| fog_mapping             | a     |
| splatter_mapping        | Randomly adds black patches of varying size on the image     |
| dotted_lines_mapping    | Randomly adds straight dotted lines on the image     |
| zigzag_mapping          | Randomly adds black zig-zag lines on the image     |
| canny_edges_mapping     | Applies Canny edge detection to highlight images and lay them over the image     |
| speckle_noise_filter    | Granular noise texture degrading the quality of the image     |
| false_color_filter      | Swaps color channels of the image, inverts each color channel or average each color channel with the other channels     |
| high_pass_filter        | Retains high frequency information in the image while reducing the low frequency information in the image, resulting in sharpened image     |
| low_pass_filter         | Calculates the average of each pixel to its neighbors     |
| phase_scrambling        | Also called power scrambling, scrambles all image channels by using Fast Fourier Transform     |
| power_equalisation      | a     |
| histogram_equalisation  | Spreads out the pixel intensity in an image via the images histogram resulting in enhancing the contrast of the image     |
| reflection_filter       | Creates a mirror effect to the input image and appends the mirrored image to the bottom of the image     |
| white_balance_filter    | Globally adjusts the intensity of image colors to render white surfaces correctly     |
| sharpen_filter          | Enhances local regions and removes blurring by using the sharpen kernel      |
| grayscale_filter        | Converts all colors to gray tones     |
| silhouette_filter       | a     |
| invert_filter           | Inverts all color channels of the image separately     |
| solarite_filter         | a     |
| posterize_filter        | Reduces the number of distinct colors while maintaining essential image features by quantization of color channels     |
| cutout_filter           | Inserts random black rectangular shapes over the image     |
| sample_pairing_filter   | Randomly samples two regions of the image together. The sampled regions are blended together with a varying alpha value     |
| gaussian_blur           | Blurs the image by applying the Gaussian function on the image     |
| saturation_filter       | Increases or decreases the saturation of the image by increasing or decreasing the saturation channel of the image in the HSV (hue, saturation, lightness) representation of the image     |
| saturation_decrease_filter | Increases or decreases the saturation of the image by increasing or decreasing the saturation channel of the image in the HSV (hue, saturation, lightness) representation of the image  |
| fog_filter              | Simulates fog by reducing the image's contrast and saturation     |
| frost_filter            | Simulates the appearance of frost patterns which form on surfaces during cold conditions     |
| snow_filter             | Simulates the effect of snow falling by artificially inserting snow crystals     |

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
