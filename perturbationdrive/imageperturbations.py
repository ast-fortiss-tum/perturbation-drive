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
    stripe_mapping,
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
    power_equalisation,
    histogram_equalisation,
    reflection_filter,
    white_balance_filter,
    sharpen_filter,
    grayscale_filter,
    silhouette_filter,
    invert_filter,
    solarite_filter,
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
from .utils.data_utils import CircularBuffer
from .utils.logger import CSVLogHandler
import types
import importlib
from .NeuralStyleTransfer.NeuralStyleTransfer import NeuralStyleTransfer


class ImagePerturbation:
    """
    Instanciates an image perturbation class

    :param scale: The scale of the perturbation in the range [1;5].
    :type scale: int

    :param funcs: List of the function names we want to use as perturbations
    :type funcs: list string
    :default funcs: If this list is empty we use all perturbations which are quick enough for
        the simultation

    :param image_size: Tuple of height and width of the image
    :type image_size: Tuple(int, int)
    :default image_size=(240,320): If this list is empty we use all perturbations
    """

    def __init__(
        self,
        funcs=[],
        log_dir="logs.csv",
        overwrite_logs=True,
        image_size=(240, 320),
        drop_boundary=3.0,
    ):
        # marks which perturbation is selected next
        self._index = 0
        self._lap = 1
        self._sector = 0
        self.scale = 0
        self.drop_boundary = drop_boundary
        self.is_stopped = False
        # setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self._csv_handler = CSVLogHandler(
            log_dir, mode=("w" if overwrite_logs else "a")
        )
        self.logger.addHandler(self._csv_handler)

        # for the first scale we randomly shuffle the filters
        # after the first scale we select the filter next with the loweset xte
        # we only iterate to the next filter if the average xte for this filter is
        # less than x, where we set x here to 2, but plan on having x as param
        # later on
        if len(funcs) == 0:
            self._fns = get_functions_from_module("perturbationdrive.perturbationfuncs")
        else:
            # the user has given us perturbations to use
            self._fns = _convertStringToPertubation(funcs)
        self._shuffle_perturbations()
        # measures for all perturbations
        self.measures = {}
        for func in self._fns:
            self.measures[func.__name__] = {
                "pertubation_name": func.__name__,
                "xte": 0,
                "steering_diff": 0,
                "frames": 0,
                "highest_scale": 0,
                "is_dropped": False,
            }
        # Load iterators for dynamic masks if we have dynamic masks
        height, width = image_size
        for filter, (path, iterator_name) in FILTER_PATHS.items():
            if filter in self._fns:
                frames = _loadMaskFrames(path, height, width)
                setattr(self, iterator_name, itertools.cycle(frames))
        # circular buffer to stop after 10 frames of crash
        self._crash_buffer = CircularBuffer(10)
        self.neuralStyleModels = NeuralStyleTransfer(getNeuralModelPaths(funcs))
        self.logger.info(
            [
                "pertubation_name",
                "xte",
                "frames",
                "highest_scale",
                "is_dropped",
                "sector",
                "lap",
                "x_pos",
                "y_pos",
                "timestamp",
                "steering_diff",
            ]
        )
        self._csv_handler.flush_row()
        print("finished setup")

    def peturbate(self, image, data: dict):
        """
        Perturbates an image based on the current perturbation
        If we the car is stuck, we reset the lap and move on to the next perturbation.
        If at any time the XTE is larger than 3, the car is fully of the raod and we will not
        move on to the next intensity stage for this perturbation

        :param image: The input image for the perturbation
        :type image: MatLike
        :param data: The necessary information from the simulator to update the perturbation object.
            It needs to contains the xte, the current lap
        :type dict:

        :return: the perturbed image
        :rtype: MatLike
        :return: states if we stop the benchmark because the car is stuck or done
        :rtype: bool
        """
        self._csv_handler.flush_row()
        if self.is_stopped:
            return {"image": image, "func": "quit_app"}
        self._crash_buffer.add((data["pos_x"], data["pos_y"]))
        if self._crash_buffer.all_elements_equal() and self._crash_buffer.length() > 9:
            print("Crash buffer is full")
            return {"image": image, "func": "reset_car"}
        # check if we have finished the lap
        if self._lap != data["lap"] or self._sector > data["sector"]:
            self._sector = data["sector"]
            self._lap = data["lap"]
            # we need to move to the next perturbation
            self._index = (self._index + 1) % len(self._fns)
            print(f"Moving on to perturbation {self._fns[self._index].__name__}")
            # check if we should increment the scale
            if self._index == 0:
                self._increment_scale()
                # print summary when incrementing scale
                self.print_performance()
                return {"image": image, "func": "reset_car"}
            else:
                return {"image": image, "func": "reset_car"}

        self._sector = data["sector"]
        self._lap = data["lap"]
        # calculate the filter index
        func = self._fns[self._index]
        # if we have a special dynamic overlay we need to pass the iterator as param
        iterator_name = ITERATOR_MAPPING.get(func, "")
        if iterator_name != "":
            iterator = getattr(self, ITERATOR_MAPPING.get(func, ""))
            pertub_image = func(self.scale, image, iterator)
        elif "styling" in func.__name__:
            pertub_image = func(self, self.scale, image)
        else:
            pertub_image = func(self.scale, image)
        # update xte
        curr_xte = self.measures[func.__name__]["xte"]
        num_perturbations = self.measures[func.__name__]["frames"]
        curr_xte = (curr_xte * num_perturbations + data["xte"]) / (
            num_perturbations + 1
        )
        self.measures[func.__name__]["xte"] = curr_xte
        self.measures[func.__name__]["frames"] = num_perturbations + 1
        if np.abs(data["xte"]) > self.drop_boundary:
            self.measures[func.__name__]["highest_scale"] = self.scale
            self.measures[func.__name__]["is_dropped"] = True
        self.logger.info(
            [
                func.__name__,
                data["xte"],
                self.measures[func.__name__]["frames"],
                self.measures[func.__name__]["highest_scale"],
                self.measures[func.__name__]["is_dropped"],
                data["sector"],
                data["lap"],
                data["pos_x"],
                data["pos_y"],
                datetime.datetime.now().timestamp(),
            ]
        )
        return {"image": pertub_image, "func": "update"}

    def updateSteeringPerformance(self, steeringAngleDiff):
        if self._index < len(self._fns):
            # calculate the filter index
            funcName = self._fns[self._index].__name__
            # update steering angle diff
            curr_diff = self.measures[funcName]["steering_diff"]
            num_differences = self.measures[funcName]["frames"]
            num_differences = num_differences if num_differences > 0 else 1
            curr_diff = (curr_diff * num_differences + steeringAngleDiff) / (
                num_differences
            )
            self.measures[funcName]["steering_diff"] = curr_diff
            self.logger.info(steeringAngleDiff)

    def _increment_scale(self):
        """
        Increments the scale by one and drops all perturbations which have been droped
        """
        self._sort_perturbations()
        self._update_fncs_scale()
        if len(self._fns) == 0:
            self.on_stop()
        if self.scale < 4:
            self.scale += 1
        else:
            # we are done
            self.on_stop()

    def on_stop(self):
        """
        Prints summary of the image perturbation

        :rtype: void
        """
        print("\n" + "=" * 45)
        print("    STOPED BENCHMARKING")
        print("=" * 45 + "\n")
        self.print_performance()
        self.is_stopped = True

    def _update_fncs_scale(self):
        for fnc in self._fns:
            self.measures[fnc.__name__]["highest_scale"] = self.scale

    def print_performance(self):
        """Command line output for the xte measures of all funcs"""
        print("\n" + "=" * 45)
        print(f"    AVERAGE PERFORMANCE ON SCALE {self.scale}")
        print("=" * 45 + "\n")
        total_average_xte = 0
        count = 0
        total_average_sad = 0
        for key, value_dict in self.measures.items():
            count += 1
            curr_xte = value_dict["xte"]
            total_average_xte += curr_xte
            steering_diff = value_dict["steering_diff"]
            total_average_sad += steering_diff
            is_droped = value_dict["is_dropped"]
            highest_scale = value_dict["highest_scale"]
            print(f"Perturbation: {key}")
            print(f"Average XTE: {curr_xte:.4f}")
            print(f"Average Steering Angle Diff: {steering_diff:.4f}")
            print(f"Is Dropped: {is_droped}")
            print(f"Highest scale achieved: {highest_scale}")
            print("-" * 45)
        total_average_xte = total_average_xte / count
        print(f"Total average XTE: {total_average_xte:.4f}")
        print(f"Total average Steering Angle Diff: {total_average_sad:.4f}")
        print("=" * 45 + "\n")

    def _shuffle_perturbations(self):
        """
        Randomly shuffles the perturbations
        """
        random.shuffle(self._fns)

    def _perturbation_dropout(self, pert_func):
        return not self.measures[pert_func.__name__]["is_dropped"]

    def _sort_perturbations(self):
        """
        Drops perturbations which are dropped due to high xte
        """
        self._fns = list(filter(self._perturbation_dropout, self._fns))

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
            print("failed to read frame")
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
    "stripe_mapping": stripe_mapping,
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
    "power_equalisation": power_equalisation,
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
    "silhouette_filter": silhouette_filter,
    "invert_filter": invert_filter,
    "solarite_filter": solarite_filter,
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
