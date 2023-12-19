import random
from abc import ABC, abstractmethod
from typing import Union, List, Tuple
from RoadGenerator.Roads.road import Road


class RoadGenerator(ABC):
    """
    Generates new roads
    """

    @abstractmethod
    def generate(self, *args, **kwargs) -> Union[str, None]:
        """
        Generates a new road and returns it as string representation in tuples
        """
        pass
