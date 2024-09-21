import random
from abc import ABC, abstractmethod
from typing import Union


class RoadGenerator(ABC):
    """
    Generates new roads
    """

    @abstractmethod
    def generate(
        self,
        *args,
        **kwargs,
    ) -> Union[str, None]:
        """
        Generates a new road and returns it as string representation. Example road is `1.0,1.0,1.0@2.0,2.0,2.0@3.0,3.0,2.0`.

        kwargs needs to at least contain the initial staring pos as arg `starting_pos`

        ---
        Within the grid-search control flow this method is called with the following key word arguments:
        - `starting_pos` (float, float, float, float): The perturbation function to apply to the road
        - `angles` List[int]: The angles to apply to the road
        - `seg_lengths` List[int]: The segment lengths to apply to the road
        - `prior_results` List[ScenarioOutcome]: All prior simulation results
        ---

        :return: The generated road as string representation
        :rtype: Union[str, None]
        """
        pass
