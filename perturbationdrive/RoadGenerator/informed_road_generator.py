from .RoadGenerator import RoadGenerator
from .Roads.road import Road
from .CustomRoadGenerator import CustomRoadGenerator

import copy
import random
import numpy as np
from typing import List


class InformedRoadGenerator(RoadGenerator):
    """Generate random roads given the configuration parameters"""

    NUM_INITIAL_SEGMENTS_THRESHOLD = 2
    NUM_UNDO_ATTEMPTS = 20

    def __init__(
        self,
        num_control_nodes=8,
        max_angle=90,
        seg_length=25,
        num_spline_nodes=20,
        initial_node=(0.0, 0.0, 0.0, 0.0),
    ):
        assert num_control_nodes > 1 and num_spline_nodes > 0
        assert 0 <= max_angle <= 360
        assert seg_length > 0
        assert len(initial_node) == 4
        self.num_control_nodes = num_control_nodes
        self.num_spline_nodes = num_spline_nodes
        self.initial_node = initial_node
        self.max_angle = max_angle
        self.seg_length = seg_length
        self.road_to_generate = None

        self.previous_road: Road = None

        self._road_gen = CustomRoadGenerator(
            num_control_nodes=num_control_nodes,
            max_angle=max_angle,
            seg_length=seg_length,
            num_spline_nodes=num_spline_nodes,
            initial_node=initial_node,
        )

    def generate(self, *args, **kwargs) -> str:
        """
        Generates a road with the given amount of turns and smoothness.

        Takes params
        - starting_pos: Tuple[float, float, float, float]
        - num_turns: int
        - avg_smoothness: float
        """
        assert "starting_pos" in kwargs, "Must provide starting_pos"
        assert "num_turns" in kwargs, "Must provide num_turns"
        assert "avg_smoothness" in kwargs, "Must provide avg_smoothness"
        assert (
            0 <= kwargs["num_turns"] <= self.num_control_nodes
        ), "Number of turns must be between 0 and the number of control nodes"
        assert 0 <= kwargs["avg_smoothness"] <= 1, "Smoothness must be between 0 and 1"
        assert len(kwargs["starting_pos"]) == 4, "Starting pos must be of length 4"

        if self.road_to_generate is not None:
            road_to_generate = copy.deepcopy(self.road_to_generate)
            self.road_to_generate = None
            return road_to_generate

        starting_pos = kwargs["starting_pos"]
        num_turnes = kwargs["num_turns"]
        avg_smoothness = kwargs["avg_smoothness"]

        generates_angles = self._angle_generator(
            num_turnes,
            avg_smoothness,
        )
        # use the angles to generate a road
        _ = self._road_gen.generate(
            starting_pos=starting_pos,
            angles=generates_angles,
            seg_lengths=None,
        )

        self.previous_road = self._road_gen.previous_road

        return self.previous_road.serialize_concrete_representation(
            cr=self.previous_road.get_concrete_representation()
        )

    def _angle_generator(
        self,
        num_turns: int,
        avg_smoothness: int,
    ):
        angles = [0]
        turns_remaining = num_turns
        entries_remaining = self.num_control_nodes - 1

        # get the starting curvature by randomly taking 1 or -1
        curve_orientation = random.choice([-1, 1])

        # conflicting generation, 0 num_turns and avg_smoothness > 5
        if num_turns == 0 and avg_smoothness >= 0.005:
            print(
                "Warning: conflicting generation, 0 num_turns and avg_smoothness > 0.005"
            )
            # return a list of random entries betwen -3 and 3
            return [random.randint(-3, 3) for _ in range(self.num_control_nodes)]

        while len(angles) < self.num_control_nodes:
            odds_of_turn = turns_remaining / entries_remaining
            if odds_of_turn > 1:
                odds_of_turn = 1
            # get the odds of generating a turn
            outcome = random.choices(
                ["turn", "staight"], weights=[odds_of_turn, 1 - odds_of_turn], k=1
            )[0]

            if outcome == "turn":
                # switch the curve orientation
                curve_orientation *= -1
                # decrement the number of turns remaining
                turns_remaining -= 1
            # search for the next best entry
            next_best_angle = self._find_next_best_angle(
                min_angle=(25 if outcome == "turn" else 0),
                smoothness_aim=avg_smoothness,
                prev_angles=angles,
                curve_orientation=curve_orientation,
            )
            angles.append(next_best_angle * curve_orientation)
            # decrement the number of entries remaining
            entries_remaining -= 1
            # check if we had an accidental turn
            if outcome != "turn" and angles[-2] == 0 and abs(angles[-1]) >= 20:
                turns_remaining -= 1

        return angles

    def _find_next_best_angle(
        self,
        min_angle: int,
        smoothness_aim: float,
        prev_angles: List[int],
        curve_orientation: int,
    ):
        best_match = None
        best_diff = np.inf

        for i in range(min_angle, self.max_angle):
            # generate a road and check for smoothness and curve values
            gen = CustomRoadGenerator(
                num_control_nodes=len(prev_angles) + 1,
            )
            angles_copy = copy.deepcopy(prev_angles)
            angles_copy.append(i * curve_orientation)
            _ = gen.generate(
                starting_pos=(0, 0, 0, 4),
                angles=angles_copy,
                seg_lengths=None,
            )
            road = gen.previous_road
            # get smoothness and curve values
            smoothness = road.curvature()
            curve = road.num_turns()
            # check if we found a better match
            if abs(smoothness - smoothness_aim) < best_diff:
                best_match = i
                best_diff = abs(smoothness - smoothness_aim)
        return best_match
