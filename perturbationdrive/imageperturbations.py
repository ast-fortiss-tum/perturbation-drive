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
    zoom_blur,
    dynamic_smoke_filter,
    dynamic_lightning_filter,
    dynamic_sun_filter,
    dynamic_object_overlay,
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
        # marks which perturbation is selected next
        self._index = 0
        self._lap = 1
        self._sector = 1
        self.scale = 0
        # fot the first scale we randomly shuffle the filters
        # after the first scale we select the filter next with the loweset xte
        # we only iterate to the next filter if the average xte for this filter is
        # less than x, where we set x here to 2, but plan on having x as param
        # later on
        if len(funcs) == 0:
            self._fns = [
                dynamic_snow_filter,
                dynamic_snow_filter,
                poisson_noise,
                jpeg_filter,
                motion_blur,
                frost_filter,
                fog_filter,
                contrast,
                elastic,
                glass_blur,
                gaussian_noise,
                dynamic_rain_filter,
                snow_filter,
                pixelate,
                increase_brightness,
                impulse_noise,
                defocus_blur,
                dynamic_smoke_filter,
                dynamic_lightning_filter,
                dynamic_sun_filter,
                dynamic_object_overlay,
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
        snow_frames = _loadMaskFramesGreenScreen("./perturbationdrive/OverlayMasks/snowfall.mp4")
        rain_frames = _loadMaskFramesGreenScreen("./perturbationdrive/OverlayMasks/rain.mp4")
        bird_frames = _loadMaskFramesGreenScreen("./perturbationdrive/OverlayMasks/birds.mp4")
        lightning_frames = _loadMaskFramesGreenScreen("./perturbationdrive/OverlayMasks/lightning.mp4")
        smoke_frames = _loadMaskFramesGreenScreen("./perturbationdrive/OverlayMasks/smoke.mp4")
        sun_frames = _loadMaskFramesGreenScreen("./perturbationdrive/OverlayMasks/sun.mp4")

        self._snow_iterator = itertools.cycle(snow_frames)
        self._rain_iterator = itertools.cycle(rain_frames)
        self._bird_iterator = itertools.cycle(bird_frames)
        self._lightning_iterator = itertools.cycle(lightning_frames)
        self._smoke_iterator = itertools.cycle(smoke_frames)
        self._sun_iterator = itertools.cycle(sun_frames)

    def peturbate(self, image, data: dict):
        """
        Perturbates an image based on the current perturbation

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
        # check if we have finished the lap
        if self._lap != data["lap"]:
            self._index += 1
        elif self._sector > data["sector"]:
            self._index += 1
        self._sector = data["sector"]
        self._lap = data["lap"]
        # calculate the filter index
        func = self._fns[self._index]
        # if we have a special dynamic overlay we need to pass the iterator as param
        if func is dynamic_snow_filter:
            image = func(self.scale, image, self._snow_iterator)
        elif func is dynamic_rain_filter:
            image = func(self.scale, image, self._rain_iterator)
        elif func is dynamic_sun_filter:
            image = func(self.scale, image, self._sun_iterator)
        elif func is dynamic_lightning_filter:
            image = func(self.scale, image, self._lightning_iterator)
        elif func is dynamic_object_overlay:
            image = func(self.scale, image, self._bird_iterator)
        elif func is dynamic_smoke_filter:
            image = func(self.scale, image, self._smoke_iterator)
        else:
            image = func(self.scale, image)
        # update xte
        curr_xte, num_perturbations = self.xte[func.__name__]
        curr_xte = (curr_xte * num_perturbations + data["xte"]) / (
            num_perturbations + 1
        )
        self.xte[func.__name__] = (curr_xte, num_perturbations + 1)
        # check if we increment the scale
        if self._index == len(self._fns) - 1:
            self._index = 0
            self._increment_scale()
            # print summary when incrementing scale
            self.print_xte()
        return image

    def updateSteeringPerformance(self, steeringAngleDiff):
        # calculate the filter index
        funcName = self._fns[self._index].__name__
        # update steering angle diff
        curr_diff, num_differences = self.steering_angle[funcName]
        curr_diff = (curr_diff * num_differences + steeringAngleDiff) / (
            num_differences + 1
        )
        self.steering_angle[funcName] = (curr_diff, num_differences + 1)

    def _increment_scale(self):
        """Increments the scale by one"""
        if self.scale == 0:
            self._sort_perturbations()
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


def _loadMaskFrames(path: str) -> list:
    """
    Helper method to load all rain frames for quicker mask overlay later

    Credits for video masks
    <a href="https://www.vecteezy.com/video/9265242-green-screen-rain-effect">Green Screen Rain Effect Stock Videos by Vecteezy</a>
    <a href="https://www.vecteezy.com/video/1803396-falling-snow-overlay-loop">Falling Snow Overlay Loop Stock Videos by Vecteezy</a>
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
        if frame.shape[2] != 4:
            # append alpha channel
            alpha_channel = np.ones(frame.shape[:2], dtype=frame.dtype) * 255
            frame = cv2.merge((frame, alpha_channel))

        frames.append(frame)
    cap.release()
    return frames


def _loadMaskFramesGreenScreen(path: str) -> list:
    """
    Helper method to load all rain frames for quicker mask overlay later

    Credits for video masks
    - <a href="https://www.vecteezy.com/video/25444944-flying-black-birds-flock-animation-on-green-screen-background">Flying black birds flock animation on green screen background Stock Videos by Vecteezy</a>
    - <a href="https://www.vecteezy.com/video/9265242-green-screen-rain-effect">Green Screen Rain Effect Stock Videos by Vecteezy</a>
    - <a href="https://www.vecteezy.com/video/1803396-falling-snow-overlay-loop">Falling Snow Overlay Loop Stock Videos by Vecteezy</a>
    - <a href="https://www.vecteezy.com/video/29896285-fly-through-dark-cloud-or-smoke-effect-animation-moving-forward-through-dark-cloud-or-smoke-effect-on-green-screen-background">Fly through dark cloud or smoke effect animation, moving forward through dark cloud or smoke effect on green screen background Stock Videos by Vecteezy</a>
    - <a href="https://www.vecteezy.com/video/16627335-realistic-lightning-strike-on-green-screen-background-blue-lightning-thunderstorm-effect-over-green-background-for-video-projects-3d-loop-animation-of-electric-thunderstorm-lightning-strike-multi">Realistic Lightning Strike On Green Screen Background , Blue Lightning Thunderstorm Effect Over Green Background For Video Projects,3d Loop Animation Of Electric Thunderstorm Lightning Strike, Multi Stock Videos by Vecteezy</a>
    - <a href="https://www.vecteezy.com/video/20614326-green-lens-flare-red-bright-glow-sun-light-lens-flares-art-animation-on-green-free-video">Green lens flare red bright glow, sun light lens flares art animation on green Free video Stock Videos by Vecteezy</a>
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
        # Define a range for the green color and create a mask.
        lower_green = np.array([35, 40, 40])  # Lower bound for the green color
        upper_green = np.array([90, 255, 255])  # Upper bound for the green color
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # Invert the mask to get the non-green areas
        mask_inv = cv2.bitwise_not(mask)

        # Split the original frame into its R, G, and B channels
        r, g, b = cv2.split(frame)

        # Create a new 4-channel image (B, G, R, alpha)
        rgba = [b, g, r, mask_inv]
        frame_with_alpha = cv2.merge(rgba, 4)

        frames.append(frame_with_alpha)
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
    "zoom_blur": zoom_blur,
    "dynamic_smoke_filter": dynamic_smoke_filter,
    "dynamic_lightning_filter": dynamic_lightning_filter,
    "dynamic_sun_filter": dynamic_sun_filter,
    "dynamic_object_overlay": dynamic_object_overlay,
}


def _convertStringToPertubation(func_names):
    """Converts a list of function names into a list of perturbation functions"""
    ret = []
    for name in func_names:
        if name in function_mapping:
            ret.append(function_mapping[name])
    return ret
