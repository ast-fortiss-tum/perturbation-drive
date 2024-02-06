# related to open_sbt
from model_ga.individual import Individual
from simulation.simulator import SimulationOutput

import numpy as np
from typing import List, Union, Tuple, Dict, Any
import json
import hashlib
import math

# related to perturbation drive
from perturbationdrive import (
    Scenario,
    CustomRoadGenerator,
    InformedRoadGenerator,
    ScenarioOutcome,
)


def _load_config() -> Dict[str, Any]:
    """
    Load the perturbation_config.json file and return it as a dictionary
    """
    with open("./examples/open_sbt/perturbation_dave2_config.json") as json_file:
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
    sim_folder: str = "sdsandbox",
    prefix: str = "",
) -> str:
    """
    Generates a name for the individual based on the values of the genes
    """
    res = {}
    fileName = ""
    for i, individual in enumerate(individuals):
        temp = {}
        for name, value in zip(variable_names, individual):
            fileName += f"{str(value)}:"
            temp[name] = int(value)
        res[i] = temp
        name += "_"
    hased_name = _hash_string_to_20_chars(fileName)
    # write res as json to hash_name.json
    with open(
        f"./logs/open_sbt/{sim_folder}/{prefix}individual_{hased_name}.json", "w"
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
            avg_smoothness = float(value) / 1000

    # generate the road string from the configuration
    road_str: str = road_generator.generate(
        starting_pos=starting_pos,
        num_turns=num_turns,
        avg_smoothness=avg_smoothness,
    )
    function_mapping = _load_config()

    # map the function
    amount_keys = len(list(function_mapping.keys()))
    if perturbation_function_int >= 0 and perturbation_function_int <= amount_keys:
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


def findClosestPoint(
    point: Tuple[float, float, float], pointsStr: str
) -> Tuple[int, int]:
    # point is a tuple of (x, y, z)
    # pointsStr is a string of a list of points seperated by @
    stringPoints = pointsStr.split("@")
    point = [float(value) for value in point]
    if math.floor(point[2] * 10) / 10 == 0.5:
        point[1], point[2] = point[2], point[1]
    # cast each strring from 0.1,0.2,0.3 to [0.1, 0.2, 0.3]
    points = [list(map(float, point.split(","))) for point in stringPoints]
    # find closest match of point in points
    closest = min(points, key=lambda p: sum((a - b) ** 2 for a, b in zip(p, point)))
    # return index of closest point and total length of points
    return (points.index(closest), len(points))


def distanceToLastRoadPoint(point: Tuple[float, float, float], pointsStr: str) -> float:
    point = [float(value) for value in point]
    stringPoints = pointsStr.split("@")
    waypoints = [list(map(float, point.split(","))) for point in stringPoints]
    x = [point_[0] for point_ in waypoints]
    y = [point_[2] for point_ in waypoints]
    z = [point_[1] for point_ in waypoints]
    lastPoint = (x[-1], y[-1], z[-1])
    if math.floor(lastPoint[1] * 10) / 10 == 0.5:
        lastPoint[1], lastPoint[2] = lastPoint[2], lastPoint[1]
    if math.floor(point[1] * 10) / 10 == 0.5:
        point[1], point[2] = point[2], point[1]
    return sum((a - b) ** 2 for a, b in zip(lastPoint, point))


def mapOutComeToSimout(outcome: ScenarioOutcome) -> SimulationOutput:
    """
    Maps the outcome of a scenario to a SimulationOutput object. Other parameters are
    xte, timeout, isSuccess, and ttf.

    Args:
        outcome (ScenarioOutcome): The outcome of a scenario.

    Returns:
        SimulationOutput: The mapped simulation output.

    """
    posLast = outcome.pos[-1]
    # swap the y and z values
    posLast[1], posLast[2] = posLast[2], posLast[1]

    # find the closest point to the last value in pos
    closestPoint, totalPoints = findClosestPoint(posLast, outcome.scenario.waypoints)
    quickness = 1 - (closestPoint / totalPoints)

    # get distance to last road point
    distance = distanceToLastRoadPoint(posLast, outcome.scenario.waypoints)
    abs_xte = [abs(xte) for xte in outcome.xte]
    isSuccess = max(abs_xte) < 2.0 and distance <= 6.0

    return SimulationOutput(
        simTime=float(len(outcome.frames)),
        times=outcome.frames,
        location={"ego": [(x[0], x[1]) for x in outcome.pos]},
        velocity={"ego": calculate_velocities(outcome.pos, outcome.speeds)},
        speed={"ego": outcome.speeds},
        acceleration={"ego": []},
        yaw={
            "ego": [],
        },
        collisions=[],
        actors={
            1: "ego",
        },
        otherParams={
            "xte": outcome.xte,
            "timeout": outcome.timeout,
            "isSuccess": isSuccess,
            "ttf": quickness,
        },
    )
