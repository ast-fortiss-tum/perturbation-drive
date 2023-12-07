from abc import ABC, abstractmethod
from udacity_utils.driving.road import Road
from typing import List, Tuple


class TestGenerator(ABC):
    def __init__(self, map_size: int):
        self.map_size = map_size
        self.road_to_generate = None

    @abstractmethod
    def generate(self, mutation_point: int = None, angles: List[int] = []) -> Road:
        """
        Generates a new road given a list of waypoints containing x and y coordinates
        """
        raise NotImplemented("Not implemented")

    @abstractmethod
    def set_max_angle(self, max_angle: int) -> None:
        raise NotImplemented("Not implemented")

    # once the road is generated the road_to_generate parameter is set to None by the concrete implementations
    def set_road_to_generate(self, road: Road) -> None:
        self.road_to_generate = road
