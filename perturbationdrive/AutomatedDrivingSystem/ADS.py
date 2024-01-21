from abc import ABC, abstractmethod
from typing import List, Any

from numpy import dtype, ndarray, uint8


class ADS(ABC):
    """
    Simulated the behavior of a automated driving system
    """

    @abstractmethod
    def action(self, input: ndarray[Any, dtype[uint8]]) -> List:
        """
        Takes one action step given the input, here the input is a cv2 image.
        This method also contains the preparation for the underlying model
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the ADS
        """
        pass
