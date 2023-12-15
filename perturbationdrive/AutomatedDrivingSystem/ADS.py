from abc import ABC, abstractmethod
from typing import List

from numpy import dtype, ndarray, Any, float32


class ADS(ABC):
    """
    Simulated the behavior of a automated driving system
    """

    @abstractmethod
    def action(self, input: ndarray[Any, dtype[float32]]) -> List:
        """
        Takes one action step given the input, here the input is a cv2 image.
        This method also contains the preparation for the underlying model
        """
        pass
