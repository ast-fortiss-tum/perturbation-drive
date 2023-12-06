# imports related to OpenSBT
from simulation.simulator import Simulator, SimulationOutput
from model_ga.individual import Individual

# all other imports
from typing import List, Dict
import logging as log
import numpy as np
import gym

# related to this simulator
from udacity_utils.generators.road_generator import RoadGenerator
from examples.udacity.udacity_utils.envs.udacity.udacity_gym_env import UdacityGymEnv


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
        try:
            results = []
            for ind in list_individuals:
                speed = []
                pos = []
                xte = []

                instance_values = [v for v in zip(variable_names, ind)]

                test_generator = RoadGenerator(map_size=250)
                env = UdacityGymEnv(
                    seed=1,
                    test_generator=test_generator,
                    exe_path=args.udacity_exe_path,  # TODO: Set this here
                )

                # set up of params
                speed = 0.0
                done = False

                # This build the path, here we need to inject our instance values
                obs = env.reset(skip_generation=False)
                while not done:
                    # sample a action for now, but later on we will use an agent here
                    actions = env.action_space.sample()
                    # clip action to avoid out of bound errors
                    if isinstance(env.action_space, gym.spaces.Box):
                        actions = np.clip(
                            actions, env.action_space.low, env.action_space.high
                        )

                    # obs is the image, info contains the road and the position of the car
                    _, done, info = env.step(actions)

                    speed.append(info["speed"])
                    pos.append(info["pos"])
                    xte.append(info("cte"))
                # morph values into SimulationOutput Object
                result = SimulationOutput(
                    simTime=float(len(speed)),
                    times=[x for x in range(len(speed))],
                    location={
                        "ego": pos,
                    },
                    velocity={
                        "ego": UdacitySimulator._calculate_velocities(pos, speed),
                    },
                    speed={
                        "ego": speed,
                    },
                    actors={
                        1: "ego",
                    },
                    otherParams={"xte": xte},
                )
                result.append(result)

        except Exception as e:
            print(f"Received exception during simulation {e}")

            raise e
        finally:
            log.info("++  finished simulation ++")
        return results

    @staticmethod
    def _calculate_velocities(positions, speeds):
        """
        Calculate velocities given a list of positions and corresponding speeds.
        """
        velocities = []

        for i in range(len(positions) - 1):
            displacement = np.array(positions[i + 1]) - np.array(positions[i])
            direction = displacement / np.linalg.norm(displacement)
            velocity = direction * speeds[i]
            velocity += 0.0  #  we append z dimensio
            velocities.append(velocity)

        return velocities

    @staticmethod
    def create_scenario_instance_xosc(filename: str, _: Dict, __=None):
        """
        Dummy method
        """
        return filename
