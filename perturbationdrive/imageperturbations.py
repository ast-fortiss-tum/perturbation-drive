import numpy as np
import cv2
import itertools
import random
from perturbationdrive.perturbationfuncs import (
    dynamic_snow_filter,
    dynamic_snow_filter,
    poisson_noise,
    jpeg_filter,
    motion_blur,
    frost_filter,
    fog_filter,
    contrast,
    elastic,
    object_overlay,
    glass_blur,
    gaussian_noise,
    dynamic_rain_filter,
    snow_filter,
    pixelate,
    increase_brightness,
    impulse_noise,
    defocus_blur,
)


class ImagePerturbation:
    """
    Instanciates an image perturbation class

    :param scale: The scale of the perturbation in the range [1;5].
    :type scale: int

    :param funcs: List of the function names we want to use as perturbations
    :type funcs: list string
    :default funcs: If this list is empty we use all perturbations
    """

    def __init__(self, scale: int, funcs=[]):
        self.scale = scale
        self._totalPerturbations = 0
        # fot the first scale we randomly shuffle the filters
        # after the first scale we select the filter next with the loweset xte
        # we only iterate to the next filter if the average xte for this filter is
        # less than x, where we set x here to 2, but plan on having x as param
        # later on
        if len(funcs) == 0:
            self._fns = [
                dynamic_snow_filter,
                poisson_noise,
                jpeg_filter,
                motion_blur,
                frost_filter,
                fog_filter,
                contrast,
                elastic,
                object_overlay,
                glass_blur,
                gaussian_noise,
                dynamic_rain_filter,
            ]
        else:
            # the user has given us perturbations to use
            self._fns = _convertStringToPertubation(funcs)
        self._shuffle_perturbations()
        # init xte for all perturbations as 0
        self.xte = {}
        # init steering angle diffs for all perturbations as 0
        self.steering_angle = {}
        for func in self._fns:
            # tupple of average xte and amount of perturbations
            self.xte[func.__name__] = (0, 0)
            self.steering_angle[func.__name__] = (0, 0)
        # we create an infinite iterator over the snow frames
        snow_frames = _loadSnowFrames()
        rain_frames = _loadRainFrames()
        self._snow_iterator = itertools.cycle(snow_frames)
        self._rain_iterator = itertools.cycle(rain_frames)

    def peturbate(self, image, prev_xte=0.0):
        """
        Perturbates an image based on the current perturbation

        :param image: The input image for the perturbation
        :type image: MatLike
        :param prev_xte: The cross track error of the car, provided by the simulator
        :type prev_xte: Float (default is 0)

        :return: the perturbed image
        :rtype: MatLike
        :return: states if we stop the benchmark because the car is stuck or done
        :rtype: bool
        """
        # calculate the filter index
        index = int(self._totalPerturbations / 100)
        func = self._fns[index]
        # if we have a special dynamic overlay we need to pass the iterator as param
        if func is dynamic_snow_filter:
            image = func(self.scale, image, self._snow_iterator)
        elif func is dynamic_rain_filter:
            image = func(self.scale, image, self._rain_iterator)
        else:
            image = func(self.scale, image)
        # update xte
        curr_xte, num_perturbations = self.xte[func.__name__]
        curr_xte = (curr_xte * num_perturbations + prev_xte) / (num_perturbations + 1)
        self.xte[func.__name__] = (curr_xte, num_perturbations + 1)
        # we only move to the next perturbation if the xte is below or equal to 2
        if (self._totalPerturbations + 1) % 100 == 0:
            if np.abs(curr_xte) <= 2:
                self._totalPerturbations += 1
        else:
            self._totalPerturbations += 1
        # check if we increment the scale
        if self._totalPerturbations == len(self._fns) * 100:
            self._totalPerturbations = 0
            self._increment_scale()
            # print summary when incrementing scale
            # we have ~20 fps, so we incremente the scale approx every 55 seconds
            self.print_xte()
        return image

    def updateSteeringPerformance(self, steeringAngleDiff):
        # calculate the filter index
        index = int((self._totalPerturbations - 1) / 100)
        funcName = self._fns[index].__name__
        # update steering angle diff
        curr_diff, num_differences = self.steering_angle[funcName]
        curr_diff = (curr_diff * num_differences + steeringAngleDiff) / (
            num_differences + 1
        )
        self.steering_angle[funcName] = (curr_diff, num_differences + 1)

    def _increment_scale(self):
        """Increments the scale by one"""
        if self.scale < 4:
            self.scale += 1

    def on_stop(self):
        """
        Prints summary of the image perturbation

        :rtype: void
        """
        print("\n" + "=" * 45)
        print("    STOPED BENCHMARKING")
        print("=" * 45 + "\n")
        self.print_xte()

    def print_xte(self):
        """Command line output for the xte measures of all funcs"""
        print("\n" + "=" * 45)
        print(f"    AVERAGE XTE ON SCALE {self.scale}")
        print("=" * 45 + "\n")
        total_average_xte = 0
        count = 0
        total_average_sad = 0
        for key, value in self.xte.items():
            count += 1
            curr_xte, _ = value
            total_average_xte += curr_xte
            steering_diff, _ = self.steering_angle[key]
            total_average_sad += steering_diff
            print(f"Average XTE for {key}: {curr_xte:.4f}")
            print(f"Average Steering Angle Diff for {key}: {steering_diff:.4f}")
            print("-" * 45)
        total_average_xte = total_average_xte / count
        print(f"Total average XTE: {total_average_xte:.4f}")
        print(f"Total average Steering Angle Diff: {total_average_sad:.4f}")
        print("=" * 45 + "\n")

    def _shuffle_perturbations(self):
        """randomly shuffles the perturbations"""
        random.shuffle(self._fns)

    def _sort_perturbations(self):
        """sorts the perturbations according to their xte"""
        self._fns = sorted(self._fns, key=lambda f: self.xte[f.__name__][0])


def _loadSnowFrames():
    """
    Helper method to load all snow frames for quicker mask overlay later

    This mask has a total of 69 frames

    Credit for the mask
    <a href="https://www.vecteezy.com/video/1803396-falling-snow-overlay-loop">Falling Snow Overlay Loop Stock Videos by Vecteezy</a>
    """
    cap = cv2.VideoCapture("./perturbationdrive/OverlayMasks/snowfall.mp4")

    # extract frames
    frames = []
    while True:
        # the image is rgb so we convert it to rgba
        ret, frame = cap.read()
        if not ret or frame is None:
            print("failed to read frame")
            break
        alpha_channel = np.ones(frame.shape[:2], dtype=frame.dtype) * 255
        frame = cv2.merge((frame, alpha_channel))
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _loadRainFrames():
    """
    Helper method to load all rain frames for quicker mask overlay later

    This mask has a total of x frames

    Credit for the mask
    <a href="https://www.vecteezy.com/video/9265242-green-screen-rain-effect">Green Screen Rain Effect Stock Videos by Vecteezy</a>
    """
    cap = cv2.VideoCapture("./perturbationdrive/OverlayMasks/rain.mp4")

    # extract frames
    frames = []
    while True:
        # the image is rgb so we convert it to rgba
        ret, frame = cap.read()
        if not ret or frame is None:
            print("failed to read frame")
            break
        if frame.shape[2] != 4:
            # append alpha channel
            alpha_channel = np.ones(frame.shape[:2], dtype=frame.dtype) * 255
            frame = cv2.merge((frame, alpha_channel))
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# Mapping of function names to function objects
function_mapping = {
    "dynamic_snow_filter": dynamic_snow_filter,
    "poisson_noise": poisson_noise,
    "jpeg_filter": jpeg_filter,
    "motion_blur": motion_blur,
    "frost_filter": frost_filter,
    "fog_filter": fog_filter,
    "contrast": contrast,
    "elastic": elastic,
    "object_overlay": object_overlay,
    "glass_blur": glass_blur,
    "gaussian_noise": gaussian_noise,
    "dynamic_rain_filter": dynamic_rain_filter,
    "snow_filter": snow_filter,
    "pixelate": pixelate,
    "increase_brightness": increase_brightness,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "pixelate": pixelate,
}


def _convertStringToPertubation(func_names):
    """Converts a list of function names into a list of perturbation functions"""
    ret = []
    for name in func_names:
        if name in function_mapping:
            ret.append(function_mapping[name])
    return ret
