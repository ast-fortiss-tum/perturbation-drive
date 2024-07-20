from perturbationdrive import ADS

from typing import List, Any
from numpy import dtype, ndarray, uint8
import numpy as np


class ExampleAgent(ADS):
    """
    Example agent using Dave2 architecture trained on SDSandBox Sim Data
    """

    def __init__(self):
        self.name = "random agent"

    def action(self, _: ndarray[Any, dtype[uint8]]) -> List:
        """
        Takes one action step given the input, here the input is a cv2 image.
        This method also contains the preparation for the underlying model
        """
        random_steering = np.random.uniform(-0.1, 0.1)
        random_acceleration = np.random.uniform(0, 0.1)
        return [[random_steering, random_acceleration]]

    def name(self) -> str:
        """
        Returns the name of the ADS
        """
        return self.name
