# related to open_sbt
from model_ga.individual import Individual

import numpy as np
from typing import List, Union, Tuple, Dict, Any
import json
import hashlib

# related to perturbation drive
from perturbationdrive import (
    Scenario,
    CustomRoadGenerator,
    InformedRoadGenerator,
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


def individualsToName(
    individuals: List[Individual],
    variable_names: List[str],
    prefix: str = "",
) -> str:
    """
    Generates a name for the individual based on the values of the genes
    """
    res = {}
    name = ""
    for i, individual in enumerate(individuals):
        temp = {}
        for name, value in zip(variable_names, individual):
            name += f"{str(value)}:"
            temp[name] = value
        res[i] = temp
        name += "_"
    hased_name = _hash_string_to_20_chars(name)
    # write res as json to hash_name.json
    with open(
        f"./logs/open_sbt/sdsandbox/{prefix}individual_{hased_name}.json", "w"
    ) as outfile:
        json.dump(res, outfile)
    return hased_name


def _hash_string_to_20_chars(input_string):
    # Hash the string using SHA-1
    hash_obj = hashlib.sha1(input_string.encode())
    hash_hex = hash_obj.hexdigest()

    # Truncate to 20 characters
    return hash_hex[:20]


def shortIndividualToScenario(
    individual: Individual,
    variable_names: List[str],
    road_generator: InformedRoadGenerator,
    starting_pos: Tuple[float, float, float],
) -> Scenario:
    num_turns: int = 0
    avg_smoothness: float = 0
    perturbation_scale: int = 0
    perturbation_function_int: int = 1
    perturbation_function: str = ""

    for variables_name, value in zip(variable_names, individual):
        # Check if the current item is the perturbation scale
        if variables_name == "perturbation_scale":
            perturbation_scale = int(value)
        elif variables_name == "perturbation_function":
            perturbation_function_int = int(value)
        elif variables_name == "num_turns":
            num_turns = int(value)
        elif variables_name == "avg_smoothness":
            avg_smoothness = float(value)

    # generate the road string from the configuration
    road_str: str = road_generator.generate(
        starting_pos=starting_pos,
        num_turns=num_turns,
        avg_smoothness=avg_smoothness,
    )
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
    return Scenario(
        waypoints=road_str,
        perturbation_function=perturbation_function,
        perturbation_scale=perturbation_scale,
    )


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
