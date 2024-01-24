from abc import ABC
from typing import Callable, List, Union, Tuple
from dataclasses import dataclass

from numpy import ndarray, uint8


@dataclass
class Scenario:
    """
    Models a scenario in terms of road, perturbation and perturbation scale
    """

    waypoints: Union[str, None]
    perturbation_function: str
    perturbation_scale: int


@dataclass
class ScenarioOutcome:
    """
    Models the outcome of a scenario
    """

    frames: List[int]
    pos: List[Tuple[float, float, float]]
    xte: List[float]
    speeds: List[float]
    actions: List[List[float]]
    scenario: Union[Scenario, None]
    isSuccess: bool
    timeout: bool


@dataclass
class OfflineScenarioOutcome:
    """
    Models the outcome of a offline testing scenario
    """

    image_file_name: str
    json_file_name: str
    perturbation_function: str
    perturbation_scale: int
    ground_truth_actions: List[float]
    perturbed_image_actions: List[float]
    normal_image_actions: List[float]
