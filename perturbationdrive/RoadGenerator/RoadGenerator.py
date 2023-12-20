import random
from abc import ABC, abstractmethod
from typing import Union, List, Tuple
from .Roads.road import Road


class RoadGenerator(ABC):
    """
    Generates new roads
    """

    @abstractmethod
    def generate(self, *args, **kwargs) -> Union[str, None]:
        """
        Generates a new road and returns it as string representation in tuples.

        kwargs needs to contain the initial staring pos as arg `starting_pos`
        """
        pass
