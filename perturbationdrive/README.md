# PerturbationDrive Docs

This ReadMe provides documentation over all functionalities in the perturbation drive folder.

## Table of Contents

- [Image Perturbations](#image-perturbations)
  - [ImagePerturbation Controller](#imageperturbation-controller)
  - [Example](#image-perturbation-example)
- [Simulator](#simulator)
  - [Scenario](#scenario)
  - [ScenarioOutcome](#scenariooutcome)
- [Automated Driving System](#ads)
- [PerturbationDrive Controller](#perturbationdrive-controller)

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

All of these functions share same parameters and return value:

- Parameters
  - scale: int. Perturbation intensity on a range from 0 to 4.
  - image: ndarray[Any, dtype[dtype=uint8]]. Image which should be perturbed.
- Returns
  - image: ndarray[Any, dtype[dtype=uint8]]. Perturbed image.

### ImagePerturbation Controller

The class `ImagePerturbation` provides a class interface for performing perturbations on images. This is also the controller used in this framework to provide easy to access perturbations. Note, that this class also provides access to the more advanced perturbations, such as Dynamic Perturbations, Generative Perturbations and Perturbations based on the Attention Map.

#### ImagePerturbation Class

When the class is initialized, all models for generative perturbations (such as Neural Style Transfer or CycleGAN) are loaded into memory. Furthermore the buffer for applying dynamic perturbations is initialized and all frames are stored in the buffer. Only when the user specified generative perturbations or dynamic perturbations, this preprocessing step is applied.
This class has the following parameters:

- `funcs` (`List[str]`, default: `[]`): List of the function names we want to use as perturbations. If this list is empty, all perturbations from the table above are used.
- `attention_map` (`dict(map: str, model: tf.model, threshold: float, layer: str)`, default: `{}`): States if we perturbated the input based on the attention map and which attention map to use. Possible arguments for map are `grad_cam` or `vanilla`. If you want to perturb based on the attention map you will need to speciy the model, attention threshold as well as the map type here. You can use either the vanilla saliency map or the Grad Cam attention map. If this dict is empty we do not perturb based on the saliency regions. The treshold can be empty and is 0.5 per default. The default layer for the GradCam Map is `conv2d_5`.
- `image_size` (`Tuple[float, float]`: default: `(240, 320)`). Input image size for all perturbations.

By creating a subclass, one can extend the perturbations used in this library. Note, that the minimum requirement for the subclass are implenting the `perturbation` function.

The table below details all function names of the generative and dynamic perturbations

| Function Name | Description |
| --------------- | --------------- |
| candy          |  Applies Neural Styling in this style   |
| la_muse          |  Applies Neural Styling in this style    |
| mosaic          |   Applies Neural Styling in this style   |
| feathers          |  Applies Neural Styling in this style    |
| the_scream          |  Applies Neural Styling in this style    |
| udnie          |  Applies Neural Styling in this style    |
| sim2real          |  Converts images from the SDSandbox Donkey USCII Track to the domain of real world images   |
| dynamic_snow_filter          | Adds artificial snow fall to the input image sequence    |
| dynamic_rain_filter          | Adds artificial rain fall to the input image sequence    |
| dynamic_sun_filter          | Artificially moves a sun across the input image sequence    |
| dynamic_lightning_filter          | Generates multiple lightning strikes over the input image sequence    |
| dynamic_smoke_filter          | Adds artificial smoke clouds to the input image sequence    |

#### ImagePerturbation.perturbation

Perturbs the input image based on the function name given. This class has the following parameters:

- `image` (`ndarray[Any, dtype[dtype=uint8]]`): Input image
- `perturbation_name` (`str`): Name of the perturbation to apply. If the string is empty, no perturbation will be appliesd. All possible perturbation names are detailed in the perturbation tables of this seection.
- `intensity`: (`int`). Perturbation intensity on a range from 0 to 4.

Returns:

- `ndarray[Any, dtype[dtype=uint8]]` The perturbed image resized to the `image_size` dimensions.

### Image Perturbation Example

```Python
from perturbationdrive import poisson_noise, gaussian_noise, ImagePerturbation
import cv2

height, width = 300, 300
random_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

# perform perturbations
poisson_img = poisson_noise(image, 0)
cv2.imshow(poisson_img)

gaussian_img = gaussian_noise(image, 4)
cv2.imshow(gaussian_img)

# this example will fail because the intensity is out of bounds
gaussian_img = gaussian_noise(image, -1)

# used the controller for perturbation

controller1 = ImagePerturbation(funcs=[candy, poisson_noise])
candy_img = controller1.perturbation("candy", 2)

# perturb the image based on the attention map
import tensorflow as tf

demo_model = tf.keras.Model(inputs=inputs, outputs=outputs)
controller2 = ImagePerturbation(funcs=[candy, poisson_noise], attention_map={"map": "grad_cam", "model": demo_model, "threshold": 0.4})
poisson_img = controller1.perturbation("poisson_noise", 2)

# this example will result in an exception because the controller does not know this perturbation
_ = controller1.perturbation("gaussian_noise", 2)
# this example will fail, because the intensity is out of bounds
__ = controller1.perturbation("gaussian_noise", 5)
```

## Simulator

### Scenario

### ScenarioOutcome

## ADS

## PerturbationDrive Controller
