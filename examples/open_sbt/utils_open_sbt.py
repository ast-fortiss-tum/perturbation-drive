# related to open_sbt
from model_ga.individual import Individual

import numpy as np
from typing import List, Union, Tuple, Dict, Any
import json

# related to perturbation drive
from perturbationdrive import (
    Scenario,
    CustomRoadGenerator,
)


def _load_config() -> Dict[str, Any]:
    """
    Load the perturbation_config.json file and return it as a dictionary
    """
    with open("./examples/open_sbt/perturbation_config.json") as json_file:
        data = json.load(json_file)
    return data


def individualToScenario(
    individual: Individual,
    variable_names: List[str],
    road_generator: CustomRoadGenerator,
    starting_pos: Tuple[float, float, float],
) -> Scenario:
    angles: List[str] = []
    perturbation_scale: int = 0
    perturbation_function_int: int = 1
    perturbation_function: str = ""
    seg_lengths: List[str] = []
    for variables_name, value in zip(variable_names, individual):
        # Check if the current item is the perturbation scale
        if variables_name == "perturbation_scale":
            perturbation_scale = int(value)
        elif variables_name == "perturbation_function":
            perturbation_function_int = int(value)
        elif variables_name.startswith("angle"):
            new_angle = int(value)
            angles.append(new_angle)
        elif variables_name.startswith("seg_length"):
            seg_length = int(value)
            seg_lengths.append(seg_length)

    # generate the road string from the configuration
    seg_lengths: Union[List[str], None] = seg_lengths if len(seg_lengths) > 0 else None
    road_str: str = road_generator.generate(
        starting_pos=starting_pos, angles=angles, seg_lengths=seg_lengths
    )
    # load function mapping from perturbation_config.json
    function_mapping = _load_config()
    # map the function
    amount_keys = len(list(function_mapping.keys()))
    if perturbation_function_int > 0 and perturbation_function_int <= amount_keys:
        perturbation_function = function_mapping[str(perturbation_function_int)]
        print(
            f"IndividualToScenario: Function is {function_mapping[str(perturbation_function_int)]}/{perturbation_function}"
        )
    else:
        perturbation_function = function_mapping["1"]
        print(
            f"IndividualToScenario: Perturbation function not found for values {perturbation_function_int}, using default: {function_mapping['1']}"
        )

    # return the scenario
    scenario = Scenario(
        waypoints=road_str,
        perturbation_function=perturbation_function,
        perturbation_scale=perturbation_scale,
    )
    return scenario


def calculate_velocities(
    positions: List[Tuple[float, float, float]], speeds: List[float]
) -> Tuple[float, float, float]:
    """
    Calculate velocities given a list of positions and corresponding speeds.
    """
    if len(positions) != len(speeds) or len(speeds) <= 1:
        return []
    velocities = []
    for i in range(len(positions) - 1):
        displacement = np.array(positions[i + 1]) - np.array(positions[i])
        displacement_norm = np.linalg.norm(displacement)
        # avoid division by zero
        if displacement_norm > 0:
            displacement_norm += 0.001
        else:
            displacement_norm -= 0.001
        direction = displacement / displacement_norm
        velocity = direction * speeds[i]
        velocities.append(velocity)
    return velocities
