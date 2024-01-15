# related to open_sbt
from model_ga.individual import Individual

import numpy as np
from typing import List, Union, Tuple

# related to perturbation drive
from perturbationdrive import (
    Scenario,
    CustomRoadGenerator,
)


def individualToScenario(
    individual: Individual,
    variable_names: List[str],
    road_generator: CustomRoadGenerator,
    starting_pos: Tuple[float, float, float],
) -> Scenario:
    instance_values = [v for v in zip(variable_names, individual)]
    angles: List[str] = []
    perturbation_scale: int = 0
    perturbation_function_int: int = 1
    perturbation_function: str = ""
    seg_lengths: List[str] = []
    for i in range(0, len(instance_values) + 1):
        # Check if the current item is the perturbation scale
        if instance_values[i][0].startswith("perturbation_scale"):
            perturbation_scale = int(instance_values[i][1])
            break
        elif instance_values[i][0].startswith("perturbation_function"):
            perturbation_function_int = int(instance_values[i][1])
            break
        elif instance_values[i][0].startswith("angle"):
            new_angle = int(instance_values[i][1])
            angles.append(new_angle)
        elif instance_values[i][0].startswith("seg_length"):
            seg_length = int(instance_values[i][1])
            seg_lengths.append(seg_length)

    # generate the road string from the configuration
    seg_lengths: Union[List[str], None] = seg_lengths if len(seg_lengths) > 0 else None
    road_str: str = road_generator.generate(
        starting_pos=starting_pos, angles=angles, seg_lengths=seg_lengths
    )
    # map the function
    amount_keys = len(list(FUNCTION_MAPPING.keys()))
    if perturbation_function_int > 0 and perturbation_function_int <= amount_keys:
        perturbation_function = FUNCTION_MAPPING[perturbation_function_int]
    else:
        perturbation_function = FUNCTION_MAPPING[1]
        print(
            f"IndividualToScenario: Perturbation function not found for values {perturbation_function_int}, using default"
        )

    # return the scenario
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


FUNCTION_MAPPING = {
    1: "gaussian_noise",
    2: "poisson_noise",
    3: "impulse_noise",
    4: "defocus_blur",
    5: "glass_blur",
    6: "increase_brightness",
}
