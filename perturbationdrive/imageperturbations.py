import numpy as np
import cv2
import os
import itertools
import random
import skimage.exposure
import logging
import datetime
from perturbationdrive.perturbationfuncs import (
    gaussian_noise,
    poisson_noise,
    impulse_noise,
    defocus_blur,
    glass_blur,
    motion_blur,
    zoom_blur,
    increase_brightness,
    contrast,
    elastic,
    pixelate,
    jpeg_filter,
    shear_image,
    translate_image,
    scale_image,
    rotate_image,
    fog_mapping,
    splatter_mapping,
    dotted_lines_mapping,
    zigzag_mapping,
    canny_edges_mapping,
    speckle_noise_filter,
    false_color_filter,
    high_pass_filter,
    low_pass_filter,
    phase_scrambling,
    histogram_equalisation,
    reflection_filter,
    white_balance_filter,
    sharpen_filter,
    grayscale_filter,
    posterize_filter,
    cutout_filter,
    sample_pairing_filter,
    gaussian_blur,
    saturation_filter,
    saturation_decrease_filter,
    fog_filter,
    frost_filter,
    snow_filter,
    dynamic_snow_filter,
    dynamic_rain_filter,
    object_overlay,
    dynamic_object_overlay,
    dynamic_sun_filter,
    dynamic_lightning_filter,
    dynamic_smoke_filter,
    perturb_high_attention_regions,
)
from perturbationdrive.road_generator import RoadGenerator
from .utils.data_utils import CircularBuffer
from .utils.logger import CSVLogHandler
from .utils.timeout import timeout_func
import types
import importlib
from .NeuralStyleTransfer.NeuralStyleTransfer import NeuralStyleTransfer
from .SaliencyMap import gradCam, getActivationMap
from typing import Any, Union
from .Generative.Sim2RealGen import Sim2RealGen
from typing import List, Tuple


class ImagePerturbation:
    """
    Instanciates an image perturbation class

    :param funcs: List of the function names we want to use as perturbations
    :type funcs: list string
    :default funcs: If this list is empty we use all perturbations which are quick enough for
        the simultation

    :param attention_map: States if we perturbated the input based on the attention map and which attention map to use. Possible arguments for map are
        `grad_cam` or `vanilla`.
        If you want to perturb based on the attention map you will need to speciy the model, attention threshold as well as the map type here.
        You can use either the vanilla saliency map or the Grad Cam attention map. If this dict is empty we do not perturb based on the saliency regions
    :type attention_map: dict(map: str, model: tf.model, threshold: float, layer: str).
    :default attention_map={}: The treshold can be empty and is 0.5 per default. The default layer for the GradCam Map is `conv2d_5`
    """

    def __init__(
        self,
        funcs: List[str] = [],
        attention_map={},
        image_size: Tuple[float, float] = (240, 320),
    ):
        # Build list of all perturbation functions
        if len(funcs) == 0:
            self._fns = get_functions_from_module("perturbationdrive.perturbationfuncs")
        else:
            # the user has given us perturbations to use
            self._fns = _convertStringToPertubation(funcs)

        # Load iterators for dynamic masks if we have dynamic masks
        height, width = image_size
        self.height = height
        self.width = width
        for filter, (path, iterator_name) in FILTER_PATHS.items():
            if filter in self._fns:
                frames = _loadMaskFrames(path, height, width)
                setattr(self, iterator_name, itertools.cycle(frames))

        self.neuralStyleModels = NeuralStyleTransfer(getNeuralModelPaths(funcs))
        # check if we use cycle gans
        if self.useGenerativeModels(funcs):
            self.cycleGenerativeModels = Sim2RealGen()
        # init perturbating saliency regions
        self.attention_func = mapSaliencyNameToFunc(attention_map.get("map", None))
        self.model = attention_map.get("model", None)
        self.saliency_threshold = attention_map.get("threshold", 0.5)
        self.grad_cam_layer = attention_map.get("layer", "conv2d_5")

        print(f"{5* '-'} Finished Perturbation-Controller set up {5* '-'}")

    def perturbation(
        self,
        image,
        perturbation_name: str,
        intensity: int,
    ) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """
        Perturbs the image based on the function name given
        """
        if perturbation_name == "":
            return cv2.resize(image, (self.width, self.height))

        # continue with the main logic
        func = FUNCTION_MAPPING[perturbation_name]
        iterator_name = ITERATOR_MAPPING.get(func, "")
        if iterator_name != "":
            iterator = getattr(self, ITERATOR_MAPPING.get(func, ""))
            pertub_image = func(intensity, image, iterator)
        elif "styling" in func.__name__ or "sim2" in func.__name__:
            # apply either style transfer or cycle gan
            pertub_image = func(self, intensity, image)
        elif self.attention_func != None:
            # preprocess image and get map
            img_array = preprocess_image_saliency(image)
            map = self.attention_func(self.model, img_array, self.grad_cam_layer)
            # perturb regions of image which have high values
            pertub_image = perturb_high_attention_regions(
                map, image, func, self.saliency_threshold, intensity
            )
        else:
            pertub_image = func(intensity, image)
        return cv2.resize(pertub_image, (self.width, self.height))

    def candy_styling(self, scale, image):
        alpha = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
        styled = self.neuralStyleModels.transferStyle(image, "candy").astype(np.uint8)
        return cv2.addWeighted(styled, alpha, image, (1 - alpha), 0)

    def la_muse_styling(self, scale, image):
        alpha = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
        styled = self.neuralStyleModels.transferStyle(image, "la_muse").astype(np.uint8)
        return cv2.addWeighted(styled, alpha, image, (1 - alpha), 0)

    def mosaic_styling(self, scale, image):
        alpha = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
        styled = self.neuralStyleModels.transferStyle(image, "mosaic").astype(np.uint8)
        return cv2.addWeighted(styled, alpha, image, (1 - alpha), 0)

    def feathers_styling(self, scale, image):
        alpha = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
        styled = self.neuralStyleModels.transferStyle(image, "feathers").astype(
            np.uint8
        )
        return cv2.addWeighted(styled, alpha, image, (1 - alpha), 0)

    def the_scream_styling(self, scale, image):
        alpha = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
        styled = self.neuralStyleModels.transferStyle(image, "the_scream").astype(
            np.uint8
        )
        return cv2.addWeighted(styled, alpha, image, (1 - alpha), 0)

    def udnie_styling(self, scale, image):
        alpha = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
        styled = self.neuralStyleModels.transferStyle(image, "udnie").astype(np.uint8)
        return cv2.addWeighted(styled, alpha, image, (1 - alpha), 0)

    def the_wave_styling(self, scale, image):
        alpha = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
        styled = self.neuralStyleModels.transferStyle(image, "the_wave").astype(
            np.uint8
        )
        return cv2.addWeighted(styled, alpha, image, (1 - alpha), 0)

    def starry_night_styling(self, scale, image):
        alpha = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
        styled = self.neuralStyleModels.transferStyle(image, "starry_night").astype(
            np.uint8
        )
        return cv2.addWeighted(styled, alpha, image, (1 - alpha), 0)

    def composition_vii_styling(self, scale, image):
        alpha = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
        styled = self.neuralStyleModels.transferStyle(image, "composition_vii").astype(
            np.uint8
        )
        return cv2.addWeighted(styled, alpha, image, (1 - alpha), 0)

    def sim2real(self, scale, image):
        alpha = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
        styled = self.cycleGenerativeModels.toReal(image)
        return cv2.addWeighted(styled, alpha, image, (1 - alpha), 0)

    def sim2sim(self, scale, image):
        alpha = [0.2, 0.4, 0.6, 0.8, 1.0][scale]
        styled = self.cycleGenerativeModels.sim2sim(image)
        return cv2.addWeighted(styled, alpha, image, (1 - alpha), 0)

    def useGenerativeModels(self, func_names):
        return True if ("sim2real" in func_names or "sim2sim" in func_names) else False


def _loadMaskFrames(path: str, isGreenScreen=True, height=240, width=320) -> list:
    """
    Loads all frames for a given mp4 video. Appends alpha channel and optionally sets
    the alpha channel of the greenscreen background to 0

    Parameters:
        - path str: Path to mp4 video
        - isGreenScreen boolean=True: States if the greenscreen background is removed
        - height int=240: Desired height of the frames
        - widht int=320: Desired widht of the frames

    Returns: list MatLike
    """
    cap = cv2.VideoCapture(path)

    # extract frames
    frames = []
    while True:
        # the image is rgb so we convert it to rgba
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv2.resize(frame, (width, height))
        if frame.shape[2] != 4:
            # append alpha channel
            alpha_channel = np.ones(frame.shape[:2], dtype=frame.dtype) * 255
            frame = cv2.merge((frame, alpha_channel))
        # option to remove greenscreen by default
        if isGreenScreen:
            frame = _removeGreenScreen(frame)
        frames.append(frame)
    cap.release()
    return frames


def _removeGreenScreen(image):
    """
    Removes green screen background by setting transparency to 0 using LAB channels

    Returns: Mathlike
    """
    # convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # extract A channel
    A = lab[:, :, 1]
    # threshold A channel
    thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # blur threshold image
    blur = cv2.GaussianBlur(
        thresh, (0, 0), sigmaX=5, sigmaY=5, borderType=cv2.BORDER_DEFAULT
    )
    # stretch so that 255 -> 255 and 127.5 -> 0
    mask = skimage.exposure.rescale_intensity(
        blur, in_range=(127.5, 255), out_range=(0, 255)
    ).astype(np.uint8)
    # add mask to image as alpha channel
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image[:, :, 3] = mask
    return image


# Mapping of function names to function objects
FUNCTION_MAPPING = {
    "gaussian_noise": gaussian_noise,
    "poisson_noise": poisson_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "glass_blur": glass_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "increase_brightness": increase_brightness,
    "contrast": contrast,
    "elastic": elastic,
    "pixelate": pixelate,
    "jpeg_filter": jpeg_filter,
    "shear_image": shear_image,
    "translate_image": translate_image,
    "scale_image": scale_image,
    "rotate_image": rotate_image,
    "fog_mapping": fog_mapping,
    "splatter_mapping": splatter_mapping,
    "dotted_lines_mapping": dotted_lines_mapping,
    "zigzag_mapping": zigzag_mapping,
    "canny_edges_mapping": canny_edges_mapping,
    "speckle_noise_filter": speckle_noise_filter,
    "false_color_filter": false_color_filter,
    "high_pass_filter": high_pass_filter,
    "low_pass_filter": low_pass_filter,
    "phase_scrambling": phase_scrambling,
    "histogram_equalisation": histogram_equalisation,
    "reflection_filter": reflection_filter,
    "white_balance_filter": white_balance_filter,
    "sharpen_filter": sharpen_filter,
    "grayscale_filter": grayscale_filter,
    "fog_filter": fog_filter,
    "frost_filter": frost_filter,
    "snow_filter": snow_filter,
    "dynamic_snow_filter": dynamic_snow_filter,
    "dynamic_rain_filter": dynamic_rain_filter,
    "object_overlay": object_overlay,
    "dynamic_object_overlay": dynamic_object_overlay,
    "dynamic_sun_filter": dynamic_sun_filter,
    "dynamic_lightning_filter": dynamic_lightning_filter,
    "dynamic_smoke_filter": dynamic_smoke_filter,
    "perturb_high_attention_regions": perturb_high_attention_regions,
    "posterize_filter": posterize_filter,
    "cutout_filter": cutout_filter,
    "sample_pairing_filter": sample_pairing_filter,
    "gaussian_blur": gaussian_blur,
    "saturation_filter": saturation_filter,
    "saturation_decrease_filter": saturation_decrease_filter,
    "candy": ImagePerturbation.candy_styling,
    "la_muse": ImagePerturbation.la_muse_styling,
    "mosaic": ImagePerturbation.mosaic_styling,
    "feathers": ImagePerturbation.feathers_styling,
    "the_scream": ImagePerturbation.the_scream_styling,
    "udnie": ImagePerturbation.udnie_styling,
    "the_wave": ImagePerturbation.the_wave_styling,
    "starry_night": ImagePerturbation.starry_night_styling,
    "la_muse": ImagePerturbation.la_muse_styling,
    "composition_vii": ImagePerturbation.composition_vii_styling,
    "sim2real": ImagePerturbation.sim2real,
    "sim2sim": ImagePerturbation.sim2sim,
}

# mapping of dynamic perturbation functions to their image path and iterator name
FILTER_PATHS = {
    dynamic_snow_filter: (
        "./perturbationdrive/OverlayMasks/snow.mp4",
        "_snow_iterator",
    ),
    dynamic_lightning_filter: (
        "./perturbationdrive/OverlayMasks/lightning.mp4",
        "_lightning_iterator",
    ),
    dynamic_rain_filter: (
        "./perturbationdrive/OverlayMasks/rain.mp4",
        "_rain_iterator",
    ),
    dynamic_object_overlay: (
        "./perturbationdrive/OverlayMasks/birds.mp4",
        "_bird_iterator",
    ),
    dynamic_smoke_filter: (
        "./perturbationdrive/OverlayMasks/smoke.mp4",
        "_smoke_iterator",
    ),
    dynamic_sun_filter: ("./perturbationdrive/OverlayMasks/sun.mp4", "_sun_iterator"),
}

# mapping of dynamic perturbation functions to their iterator name
ITERATOR_MAPPING = {
    dynamic_snow_filter: "_snow_iterator",
    dynamic_rain_filter: "_rain_iterator",
    dynamic_sun_filter: "_sun_iterator",
    dynamic_lightning_filter: "_lightning_iterator",
    dynamic_object_overlay: "_bird_iterator",
    dynamic_smoke_filter: "_smoke_iterator",
}


def _convertStringToPertubation(func_names):
    """
    Converts a list of function names into a list of perturbation functions
    """
    ret = []
    for name in func_names:
        if name in FUNCTION_MAPPING:
            ret.append(FUNCTION_MAPPING[name])
    return ret


def get_functions_from_module(module_name):
    """
    Import all functions from a module and return them in a list.

    Args:
    - module_name (str): The name of the module to import.

    Returns:
    - List[types.FunctionType]: A list of functions from the module.
    """
    module = importlib.import_module(module_name)

    functions_list = [
        getattr(module, attr_name)
        for attr_name in dir(module)
        if isinstance(getattr(module, attr_name), types.FunctionType)
        and getattr(module, attr_name).__module__ == module_name
    ]
    functions_list = [
        func
        for func in functions_list
        if func.__name__ != "perturb_high_attention_regions"
        and func.__name__ != "high_pass_filter"
        and func.__name__ != "fog_mapping"
        and func.__name__ != "zoom_blur"
    ]
    return functions_list


def getNeuralModelPaths(style_names: [str]):
    paths = [
        "perturbationdrive/NeuralStyleTransfer/models/instance_norm/candy.t7",
        "perturbationdrive/NeuralStyleTransfer/models/eccv16/composition_vii.t7",
        "perturbationdrive/NeuralStyleTransfer/models/instance_norm/feathers.t7",
        "perturbationdrive/NeuralStyleTransfer/models/instance_norm/la_muse.t7",
        "perturbationdrive/NeuralStyleTransfer/models/instance_norm/mosaic.t7",
        "perturbationdrive/NeuralStyleTransfer/models/eccv16/starry_night.t7",
        "perturbationdrive/NeuralStyleTransfer/models/instance_norm/the_scream.t7",
        "perturbationdrive/NeuralStyleTransfer/models/eccv16/the_wave.t7",
        "perturbationdrive/NeuralStyleTransfer/models/instance_norm/udnie.t7",
    ]
    style_names = [style for style in style_names if any(style in s for s in paths)]
    lookup_dict = {os.path.splitext(os.path.basename(path))[0]: path for path in paths}
    return [lookup_dict[key] for key in style_names]


def mapSaliencyNameToFunc(name: Union[str, None]):
    if name == None:
        return None
    elif name == "grad_cam":
        return gradCam
    elif name == "vanilla":
        return getActivationMap
    else:
        return None


def preprocess_image_saliency(img):
    img_arr = np.asarray(img, dtype=np.float32)
    return img_arr.reshape((1,) + img_arr.shape)
