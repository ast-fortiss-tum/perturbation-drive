# imports related to OpenSBT
from simulation.simulator import Simulator, SimulationOutput
from model_ga.individual import Individual

# all other imports
from typing import List, Dict, Tuple
import logging as log
import numpy as np
import gym
import cv2
from PIL import Image
from numpy.typing import NDArray
from numpy import uint8
from perturbationdrive import gaussian_blur
from tensorflow.keras.models import load_model
import time

# related to this simulator
from udacity_utils.generators.road_generator import RoadGenerator
from udacity_utils.envs.udacity.udacity_gym_env import (
    UdacityGymEnv_RoadGen,
)
from udacity_utils.config import NUM_CONTROL_NODES


class UdacitySimulator(Simulator):
    @staticmethod
    def simulate(
        list_individuals: List[Individual],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float,
        do_visualize: bool = False,
    ) -> List[SimulationOutput]:
        """
        Runs all indicidual simulations and returns simulation outputs
        """
        results = []
        # load ADS system
        model = load_model(
            "./examples/sdsandbox_perturbations/generatedRoadModel.h5",
            compile=False,
        )

        for ind in list_individuals:
            try:
                speeds = []
                pos = []
                xte = []

                instance_values = [v for v in zip(variable_names, ind)]
                # get the perturbation scale, the instance values are in the format [x1, y1, x2, y2, ..., scale]

                (
                    angles,
                    perturbation_scale,
                ) = UdacitySimulator._process_simulation_vars(instance_values)

                test_generator = RoadGenerator(map_size=250)
                env = UdacityGymEnv_RoadGen(
                    seed=1,
                    test_generator=test_generator,
                    exe_path="./examples/udacity/udacity_utils/sim/udacity_sim.app",
                )

                # set up of params
                done = False

                obs: NDArray[uint8] = env.reset(skip_generation=False, angles=angles)
                while not done:
                    # resize to fit model
                    obs = cv2.resize(obs, (320, 240), cv2.INTER_NEAREST)
                    # perturb image and preprocess it
                    img = gaussian_blur(perturbation_scale, obs)
                    img_arr = np.asarray(img, dtype=np.float32)
                    # add a batch dimension
                    img_arr = img_arr.reshape((1,) + img_arr.shape)

                    actions = model(img_arr, training=False)

                    # clip action to avoid out of bound errors
                    if isinstance(env.action_space, gym.spaces.Box):
                        actions = np.clip(
                            actions, env.action_space.low, env.action_space.high
                        )

                    # obs is the image, info contains the road and the position of the car
                    obs, done, info = env.step(actions)

                    speeds.append(info["speed"])
                    pos.append(info["pos"])
                    xte.append(info["cte"])
                # morph values into SimulationOutput Object
                result = SimulationOutput(
                    simTime=float(len(speeds)),
                    times=[x for x in range(len(speeds))],
                    location={
                        "ego": [(x[0], x[1]) for x in pos],  # cut of z value
                    },
                    velocity={
                        "ego": UdacitySimulator._calculate_velocities(pos, speeds),
                    },
                    speed={
                        "ego": speeds,
                    },
                    acceleration={"ego": []},
                    yaw={
                        "ego": [],
                    },
                    collisions=[],
                    actors={
                        1: "ego",
                    },
                    otherParams={"xte": xte},
                )

                results.append(result)
            except Exception as e:
                print(f"Received exception during simulation {e}")

                raise e
            finally:
                # time.sleep(2)
                env.close()

        return results

    @staticmethod
    def _calculate_velocities(positions, speeds) -> Tuple[float, float, float]:
        """
        Calculate velocities given a list of positions and corresponding speeds.
        """
        velocities = []

        for i in range(len(positions) - 1):
            displacement = np.array(positions[i + 1]) - np.array(positions[i])
            direction = displacement / np.linalg.norm(displacement)
            velocity = direction * speeds[i]
            velocities.append(velocity)

        return velocities

    @staticmethod
    def _process_simulation_vars(
        instance_values: List[Tuple[str, float]],
    ) -> Tuple[List[int], int]:
        angles = []
        perturbation_scale = None

        # Iterate over the data list
        for i in range(0, len(instance_values)):
            # Check if the current item is the perturbation scale
            if instance_values[i][0].startswith("perturbation"):
                perturbation_scale = int(instance_values[i][1])
                break

            # Extract and pair x and y coordinates
            new_angle = int(instance_values[i][1])
            angles.append(new_angle)

        return angles, perturbation_scale

    @staticmethod
    def create_scenario_instance_xosc(filename: str, _: Dict, __=None):
        """
        Dummy method
        """
        return filename
