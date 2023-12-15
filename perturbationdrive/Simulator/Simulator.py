from abc import ABC, abstractmethod, property
from typing import Dict, List, Union
from ..AutomatedDrivingSystem import ADS
from Scenario import Scenario, ScenarioOutcome
from ..imageperturbations import ImagePerturbation


class PerturbationSimulator(ABC):
    """
    Adapter for a simulator
    """

    @property
    @abstractmethod
    def max_xte(self) -> float:
        """
        Returns the maximum xte before the car is considered to be off the road
        """
        pass

    @property
    @abstractmethod
    def simulator_exe_path(self) -> Union[str, None]:
        """
        Path to exe of simulator. If this path is none, the user has to start the sim manually
        """
        pass

    @abstractmethod
    def connect(sefl):
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
