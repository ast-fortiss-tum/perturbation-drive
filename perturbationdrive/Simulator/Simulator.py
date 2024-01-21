from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple
from ..AutomatedDrivingSystem import ADS
from .Scenario import Scenario, ScenarioOutcome
from ..imageperturbations import ImagePerturbation


class PerturbationSimulator(ABC):
    """
    Adapter for a simulator
    """

    def __init__(
        self,
        max_xte: float = 2.0,
        simulator_exe_path: str = "",
        host: str = "127.0.0.1",
        port: int = 9091,
        initial_pos: Union[Tuple[float, float, float, float], None] = None,
    ):
        self.max_xte = max_xte
        self.simulator_exe_path = simulator_exe_path
        self.host = host
        self.port = port
        self.initial_pos: Union[Tuple[float, float, float, float], None] = initial_pos

    @abstractmethod
    def connect(self):
        """
        Connects to the simulator
        """
        pass

    @abstractmethod
    def simulate_scanario(
        self, agent: ADS, scenario: Scenario, perturbation_controller: ImagePerturbation
    ) -> ScenarioOutcome:
        """
        Simulates the given list of scenarions.
        This method needs to handle the different formats of the waypoints of a scenario (e.g. angles, points or none).
        If the images should be displayed or saves, this is the method to use.

        For all given scenarios the following steps are repreated
        - Resets the simulator
        - Builds the scenarion (in terms of waypoints)
        - Runs an action loop
            - Get the observation from the simulator
            - Perturb the observation given a method by the perturbation controller
            - Feed the perturbed observation into the ADS
            - Perform the actions of the ADS
        """
        pass

    @abstractmethod
    def tear_down(self):
        """
        Tears the connection to the simulator down
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the simulator
        """
        pass
