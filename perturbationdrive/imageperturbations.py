import numpy as np
import cv2
import os
import itertools
import skimage.exposure
from perturbationdrive.AttentionMasks.raindrops_generator.raindrop.dropgenerator import generateDrops, generate_label
import random

from perturbationdrive.AttentionMasks.raindrops_generator.raindrop.config import cfg
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
    dynamic_raindrop_filter,
    object_overlay,
    dynamic_object_overlay,
    dynamic_sun_filter,
    dynamic_lightning_filter,
    dynamic_smoke_filter,
    perturb_high_attention_regions,
    perturb_highest_n_attention_regions,
    perturb_lowest_n_attention_regions,
    perturb_random_n_attention_regions,
    effects_attention_regions,
    static_lightning_filter,
    static_smoke_filter,
    static_sun_filter,
    static_rain_filter,
    static_snow_filter,
    static_smoke_filter,
    static_object_overlay,
)
from perturbationdrive.RoadGenerator.RoadGenerator import RoadGenerator
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
        The image can either be perturbed on the absolut value or the highest / lowest n-percent of the attention map by specifying the attention_perturbation argument..
        Default attention_perturbation is `perturb_high_attention_regions`, other possible values are `perturb_highest_n_attention_regions`, `perturb_lowest_n_attention_regions` and `perturb_random_n_attention_regions`.
    :type attention_map: dict(map: str, model: tf.model, threshold: float, layer: str, attention_perturbation: str).
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
        for filter, (path, iterator_name, color, thres) in FILTER_PATHS.items():
            if filter in self._fns:
                print(
                    f"{5* '-'} Loading Dynamic Masks - This can take some time {5* '-'}"
                )
                frames = _loadMaskFrames(
                    path=path,
                    isGreenScreen=True,
                    height=height,
                    width=width,
                    green_screen_color=color,
                    threshold=thres,
                )
                setattr(self, iterator_name, itertools.cycle(frames))
        for filter, (path, mask_name, color, thres) in STATIC_PATHS.items():
            if filter in self._fns:
                # set image without green screen as mask
                raw_image = cv2.imread(path)
                # move to rbg
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                # add alpha channel
                if raw_image.shape[2] != 4:
                    # append alpha channel
                    alpha_channel = (
                        np.ones(raw_image.shape[:2], dtype=raw_image.dtype) * 255
                    )
                    raw_image = cv2.merge((raw_image, alpha_channel))
                image = _remove_green_pixels(image, color, thres)
                image = cv2.resize(image, (width, height))
                setattr(self, mask_name, image)

        self.neuralStyleModels = NeuralStyleTransfer(getNeuralModelPaths(funcs))
        # check if we use cycle gans
        if self.useGenerativeModels(funcs):
            self.cycleGenerativeModels = Sim2RealGen()
        # init perturbating saliency regions
        self.attention_func = mapSaliencyNameToFunc(attention_map.get("map", None))
        self.model = attention_map.get("model", None)
        self.saliency_threshold = attention_map.get("threshold", 0.5)
        self.grad_cam_layer = attention_map.get("layer", "conv2d_5")
        self.attention_perturbation = attention_map.get(
            "attention_perturbation", "no_usage"
        )
        self.iteration=0
        self.previous_points=[]
        self.previous_sizes=[]
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
        mask_name = MASK_MAPPING.get(func, "")
        if iterator_name != "":
            iterator = getattr(self, ITERATOR_MAPPING.get(func, ""))
            pertub_image = func(intensity, image, iterator)
        elif mask_name != "":
            mask = getattr(self, MASK_MAPPING.get(func, ""))
            pertub_image = func(intensity, image, mask)
        elif "styling" in func.__name__ or "sim2" in func.__name__:
            # apply either style transfer or cycle gan
            pertub_image = func(self, intensity, image)
        elif "attention" in func.__name__ and self.attention_func != None and "dynamic" in func.__name__:
            img_array = preprocess_image_saliency(image)
            map = self.attention_func(self.model, img_array, self.grad_cam_layer)
            pertub_image = func(self, map, intensity, image, perturbation_name)
        elif "attention" in func.__name__ and self.attention_func != None:
            img_array = preprocess_image_saliency(image)
            map = self.attention_func(self.model, img_array, self.grad_cam_layer)
            pertub_image = func(map, intensity, image,perturbation_name)
        elif "attention" in func.__name__:
            print("ERROR NO ATTENTION ON ATTENTION BASED PERTURBATION")
            return cv2.resize(image, (self.width, self.height))
        elif "effects"  in func.__name__ and "dynamic" in func.__name__:
            pertub_image = func(self, intensity, image,perturbation_name)
        elif self.attention_perturbation != "no_usage":
            # preprocess image and get map
            img_array = preprocess_image_saliency(image)
            map = self.attention_func(self.model, img_array, self.grad_cam_layer)
            if self.attention_perturbation == "perturb_highest_n_attention_regions":
                pertub_image = perturb_highest_n_attention_regions(
                    map, image, func, (self.saliency_threshold * 100), intensity
                )
            elif self.attention_perturbation == "perturb_lowest_n_attention_regions":
                pertub_image = perturb_lowest_n_attention_regions(
                    map, image, func, (self.saliency_threshold * 100), intensity
                )
            elif self.attention_perturbation == "perturb_random_n_attention_regions":
                pertub_image = perturb_random_n_attention_regions(
                    map, image, func, (self.saliency_threshold * 100), intensity
                )
            else:
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



    def effects_attention_regions_dynamic(
        self,saliency_map,scale, image,name
    ):
        cfg = {
            'maxR': 30,
            'minR': 1,
            'maxDrops': 100,
            'minDrops': 1,
            'edge_darkratio': 0.4,
            'return_label': False,
            'label_thres': 128,
            'A': (1, 4.5),
            'B': (3, 1),
            'C': (1, 3),
            'D': (3, 3)
        }
        coords_tuples=[]
        shapes = None
        sizes = None
        
        sizes_static = [2,3,4,3,4,5,2,3,1]
        coords_tuples_static=[(0,200),(10,130),(30,80),(70,40),(140,180),(200,30),(250,190),(300,40),(150,80)]

        coords_tuples=[]
        sizes=[]
        for i in range((1+scale)*2):
            coords_tuples.append((random.randint(1,300),random.randint(0,200)))
            sizes.append(random.randint(1,(1+scale)*5))
        coords_tuples=coords_tuples+coords_tuples_static
        sizes=sizes+sizes_static
        
        
        if len(self.previous_points)!=0 and len(self.previous_points)==len(coords_tuples):
            
            shapes=self.previous_shapes
            sizes = self.previous_sizes
            
            if self.iteration%2==0:
                coords_tuples=[]
                direction_x=random.randint(-1,1)
                for i,value in enumerate(self.previous_points):
                    x,y=value
                    x=x+direction_x
                    direction_y=random.randint(0,1)
                    if sizes[i]<6:
                        y=y+direction_y*3
                    elif sizes[i]<12:
                        y=y+direction_y
                    else:
                        y=y+direction_y*2
                    coords_tuples.append((x,y))
            else:
                coords_tuples=self.previous_points

            for i,value in enumerate(coords_tuples):
                x,y=value
                size=sizes[i]
                if x>310 or x<2:
                    x=random.randint(30,290)
                if y>235:
                    if i<(1+scale)*2:
                        max_coords = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
                        x=max_coords[0]
                        y=max_coords[1]
                        size=random.randint((1+scale)+10,30)
                    else:
                        y=random.randint(0,40)
                        size=random.randint(1,5)
                elif y>200:
                    if random.randint(0,5)==4:
                        size=max(1,min(size+random.randint(-2,0),30))
                elif y>180:
                    if random.randint(0,5)==4:
                        size=max(1,min(size+random.randint(-1,0),30))
                coords_tuples[i]=(x,y)
                sizes[i]=size

            if self.iteration%30==0:
                    sizes = []
                    for i,value in enumerate(self.previous_sizes):
                        sizes.append(max(1,min(value+random.randint(-1,1),30)))
        List_of_Drops, self.previous_shapes, self.previous_sizes  = generate_label(image.shape[0], image.shape[1], coords_tuples,cfg,shapes,sizes)
        output_image = generateDrops(image, cfg, List_of_Drops)
        self.previous_points=coords_tuples
        self.iteration+=1
        return output_image
    

        


    
    def effects_regions_dynamic(
        self,scale, image,name
    ):
        cfg = {
            'maxR': 30,
            'minR': 1,
            'maxDrops': 100,
            'minDrops': 1,
            'edge_darkratio': 0.4,
            'return_label': False,
            'label_thres': 128,
            'A': (1, 4.5),
            'B': (3, 1),
            'C': (1, 3),
            'D': (3, 3)
        }
        coords_tuples=[]
        shapes = None
        sizes = None
        if "snowflake" in name:
            sizes=[]
            coords_tuples=[]
            for i in range(100*(scale+1)):
                sizes.append(1)
                coords_tuples.append((random.randint(20,300),random.randint(20,220)))
        else:
            sizes = [2,3,4,3,4,5,2,3,1]
            coords_tuples=[(0,200),(10,130),(30,80),(70,40),(140,180),(200,30),(250,190),(300,40),(150,80)]
        # coords_tuples=[(300,1)]
        
        if len(self.previous_points)!=0 and len(self.previous_points)==len(coords_tuples):
            
            shapes=self.previous_shapes
            sizes = self.previous_sizes
            
            if self.iteration%1==0:
                coords_tuples=[]
                direction_x=random.randint(-1,1)
                for i,value in enumerate(self.previous_points):
                    if "snowflake" not in name:
                        x,y=value
                        x=x+direction_x
                        direction_y=random.randint(0,1)
                        if sizes[i]<6:
                            y=y+direction_y*3
                        elif sizes[i]<12:
                            y=y+direction_y
                        else:
                            y=y+direction_y*2
                    else:
                        x,y=value
                        if x<160:
                            x=x-5
                        elif x>160:
                            x=x+5
                        elif x==160:
                            x=x+random.randint(-1,1)
                        if y<120:
                            y=y-4
                        elif y>120:
                            y=y+4
                        else:
                            y=y+random.randint(-5,5)
                    coords_tuples.append((x,y))
            else:
                coords_tuples=self.previous_points

            for i,value in enumerate(coords_tuples):
                x,y=value
                size=sizes[i]
                if "snowflake" not in name:
                    if x>310 or x<2:
                        x=random.randint(30,290)
                    if y>235:
                        y=random.randint(0,40)
                        size=random.randint(1,(scale+1)*5)
                    elif y>200:
                        if random.randint(0,5)==4:
                            size=max(1,min(size+random.randint(-2,0),30))
                    elif y>180:
                        if random.randint(0,5)==4:
                            size=max(1,min(size+random.randint(-1,0),30))
                else:
                    if (x>310 or x<2) and (y>239 or y<2):
                        x,y = (random.randint(20,300),random.randint(20,220))
                        
                coords_tuples[i]=(x,y)
                sizes[i]=size

            if self.iteration%30==0:
                    sizes = []
                    for i,value in enumerate(self.previous_sizes):
                        if "snowflake" not in name:
                            sizes.append(max(1,min(value+random.randint(-1,1),30)))
                        else:
                            sizes.append(max(1,min(value+random.randint(-1,1),3)))
        
        List_of_Drops, self.previous_shapes, self.previous_sizes  = generate_label(image.shape[0], image.shape[1], coords_tuples,cfg,shapes,sizes)
        output_image = generateDrops(image, cfg, List_of_Drops)
        self.previous_points=coords_tuples
        self.iteration+=1
        return output_image


def find_third_point(p1, p2, n):
    
    # Convert points to NumPy arrays for easier calculations
    p1 = np.array(p1)
    p2 = np.array(p2)

    p2[0]=p2[0]+random.randint(-10,10)
    p2[1]=p2[1]+random.randint(-10,10)

    # Calculate the direction vector from p1 to p2
    direction = p2 - p1

    # Calculate the length of the direction vector
    length = np.linalg.norm(direction)

    # Normalize the direction vector to get the unit vector
    unit_vector = direction / length

    # Scale the unit vector by the distance n
    scaled_vector = unit_vector * n

    # Calculate the third point by adding the scaled vector to p1
    p3 = p1 + scaled_vector
    p3=[int(p3[0]),int(p3[1])]
    return tuple(p3)

def _loadMaskFrames(
    path: str,
    isGreenScreen=True,
    height=240,
    width=320,
    green_screen_color: List[int] = [66, 193, 5],
    threshold=40,
) -> list:
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
        # convert to rgba
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        # option to remove greenscreen by default
        if frame.shape[2] != 4:
            # append alpha channel
            alpha_channel = np.ones(frame.shape[:2], dtype=frame.dtype) * 255
            frame = cv2.merge((frame, alpha_channel))
        if isGreenScreen:
            frame = _remove_green_pixels(frame, green_screen_color, threshold=threshold)
        frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    cap.release()
    return frames


def _remove_green_pixels(image, target_green_rgb, threshold=40):
    """
    Remove green pixels around a target color value.

    """
    # Convert the target green color to HSV since it's better for color thresholding
    target_green_hsv = cv2.cvtColor(np.uint8([[target_green_rgb]]), cv2.COLOR_RGB2HSV)[
        0
    ][0]

    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for the target green color
    lower_green = np.array(
        [
            max(target_green_hsv[0] - threshold, 0),
            max(target_green_hsv[1] - threshold, 0),
            max(target_green_hsv[2] - threshold, 0),
        ]
    )
    upper_green = np.array(
        [
            min(target_green_hsv[0] + threshold, 255),
            min(target_green_hsv[1] + threshold, 255),
            min(target_green_hsv[2] + threshold, 255),
        ]
    )

    # Create a mask for the green color
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    if image.shape[2] == 3:
        # Create an alpha channel
        alpha_channel = np.ones(image.shape[:2], dtype=image.dtype) * 255
        # Add the alpha channel to make it 4-channel
        image = cv2.merge([image, alpha_channel])

    if image.shape[2] == 4:
        mask_4channel = cv2.merge([mask, mask, mask, np.zeros_like(mask)])
        mask_indices = np.where(mask_4channel == 255)

        image[mask_indices[0], mask_indices[1], :] = (
            255,
            255,
            255,
            0,
        )  # Replacing with transparent
    else:
        mask_3channel = cv2.merge([mask, mask, mask])
        mask_indices = np.where(mask_3channel == 255)
        image[mask_indices[0], mask_indices[1], :] = (
            255,
            255,
            255,
            0,
        )  # Replacing with transparent

    return image


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
    "dynamic_raindrop_filter": dynamic_raindrop_filter,
    "object_overlay": object_overlay,
    "dynamic_object_overlay": dynamic_object_overlay,
    "dynamic_sun_filter": dynamic_sun_filter,
    "dynamic_lightning_filter": dynamic_lightning_filter,
    "dynamic_smoke_filter": dynamic_smoke_filter,
    "static_snow_filter": static_snow_filter,
    "static_rain_filter": static_rain_filter,
    "static_object_overlay": static_object_overlay,
    "static_sun_filter": static_sun_filter,
    "static_lightning_filter": static_lightning_filter,
    "static_smoke_filter": static_smoke_filter,
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
    "effects_attention_regions": effects_attention_regions,
    "effects_attention_regions_dynamic": ImagePerturbation.effects_attention_regions_dynamic,
    "effects_regions_dynamic": ImagePerturbation.effects_regions_dynamic
}

# mapping of dynamic perturbation functions to their image path and iterator name
FILTER_PATHS = {
    dynamic_snow_filter: (
        "./perturbationdrive/OverlayMasks/snow.mp4",
        "_snow_iterator",
        [8, 255, 18],
        45,
    ),
    dynamic_lightning_filter: (
        "./perturbationdrive/OverlayMasks/lightning.mp4",
        "_lightning_iterator",
        [32, 91, 10],
        45,
    ),
    dynamic_rain_filter: (
        "./perturbationdrive/OverlayMasks/rain.mp4",
        "_rain_iterator",
        [3, 129, 8],
        40,
    ),
    dynamic_raindrop_filter: (
        "./perturbationdrive/OverlayMasks/test.mp4",
        "_raindrop_iterator",
        [8, 255, 18],
        45,
    ),
    dynamic_object_overlay: (
        "./perturbationdrive/OverlayMasks/birds.mp4",
        "_bird_iterator",
        [66, 193, 5],
        40,
    ),
    dynamic_smoke_filter: (
        "./perturbationdrive/OverlayMasks/smoke.mp4",
        "_smoke_iterator",
        [37, 149, 59],
        75
    ),
    dynamic_sun_filter: (
        "./perturbationdrive/OverlayMasks/sun.mp4",
        "_sun_iterator",
        [9, 166,  56],
        60
    ),
}

STATIC_PATHS = {
    static_snow_filter: (
        "./perturbationdrive/OverlayMasks/static_snow.png",
        "_snow_mask",
        [8, 255, 18],
        45.0,
    ),
    static_lightning_filter: (
        "./perturbationdrive/OverlayMasks/static_light.png",
        "_lightning_mask",
        [32, 91, 10],
        45,
    ),
    static_rain_filter: (
        "./perturbationdrive/OverlayMasks/static_rain.png",
        "_rain_mask",
        [3, 129, 8],
        40,
    ),
    static_object_overlay: (
        "./perturbationdrive/OverlayMasks/static_birds.png",
        "_bird_mask",
        [66, 193, 5],
        40,
    ),
    static_smoke_filter: (
        "./perturbationdrive/OverlayMasks/static_smoke.png",
        "_smoke_mask",
        [37, 149, 59],
        75
    ),
    static_sun_filter: (
        "./perturbationdrive/OverlayMasks/static_sun.png",
        "_sun_mask",
        [9, 166,  56],
        60
    ),
}

# mapping of dynamic perturbation functions to their iterator name
ITERATOR_MAPPING = {
    dynamic_snow_filter: "_snow_iterator",
    dynamic_rain_filter: "_rain_iterator",
    dynamic_raindrop_filter: "_raindrop_iterator",
    dynamic_sun_filter: "_sun_iterator",
    dynamic_lightning_filter: "_lightning_iterator",
    dynamic_object_overlay: "_bird_iterator",
    dynamic_smoke_filter: "_smoke_iterator",
}

MASK_MAPPING = {
    static_snow_filter: "_snow_mask",
    static_rain_filter: "_rain_mask",
    static_sun_filter: "_sun_mask",
    static_lightning_filter: "_lightning_mask",
    static_object_overlay: "_bird_mask",
    static_smoke_filter: "_smoke_mask",
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


def getNeuralModelPaths(style_names: List[str]):
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
    style_names = [
        style for style in style_names if any(style in s for s in paths) and style != ""
    ]
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
