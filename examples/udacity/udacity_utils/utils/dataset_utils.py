import cv2
import numpy as np
from examples.udacity.udacity_utils.config import IMAGE_WIDTH, IMAGE_HEIGHT


def crop(image: np.ndarray) -> np.ndarray:
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]  # remove the sky and the car front (from SelfOracle)


def resize(image: np.ndarray) -> np.ndarray:
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def bgr2yuv(image: np.ndarray) -> np.ndarray:
    """
    Convert the image from BGR to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Combine all preprocess functions into one
    """
    image = crop(image=image)
    image = resize(image=image)
    image = bgr2yuv(image=image)
    return image


