# Original author: Roma Sokolkov
# Edited by Antonin Raffin
import os
import time
from typing import Optional, Tuple, Dict

import gym
import numpy as np
from gym import spaces

from envs.udacity.config import BASE_PORT, MAX_STEERING, INPUT_DIM
from envs.udacity.core.udacity_sim import UdacitySimController
from envs.unity_proc import UnityProcess
from global_log import GlobalLog
from test_generators.test_generator import TestGenerator


class UdacityGymEnv(gym.Env):
    """
    Gym interface for DonkeyCar with support for using
    a VAE encoded observation instead of raw pixels if needed.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
            self,
            seed: int,
            test_generator: TestGenerator = None,
            headless: bool = False,
            exe_path: str = None,
    ):

        self.seed = seed
        self.exe_path = exe_path
        self.logger = GlobalLog('UdacityGymEnv')
        self.test_generator = test_generator
        if headless:
            self.logger.warn("Headless mode not supported with Udacity")
        self.headless = False
        self.port = BASE_PORT

        self.unity_process = None
        if self.exe_path is not None:
            self.logger.info("Starting UdacityGym env")
            assert os.path.exists(self.exe_path), 'Path {} does not exist'.format(self.exe_path)
            # Start Unity simulation subprocess if needed
            self.unity_process = UnityProcess()
            self.unity_process.start(sim_path=self.exe_path, headless=headless, port=self.port)
            time.sleep(2)  # wait for the simulator to start and the scene to be selected

        self.executor = UdacitySimController(port=self.port, test_generator=test_generator)

        # steering + throttle, action space must be symmetric
        self.action_space = spaces.Box(
            low=np.array([-MAX_STEERING, -1]),
            high=np.array([MAX_STEERING, 1]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=0, high=255, shape=INPUT_DIM, dtype=np.uint8)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, bool, Dict]:
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        # action[0] is the steering angle
        # action[1] is the throttle

        self.executor.take_action(action=action)
        observation, done, info = self.observe()

        return observation, done, info

    def reset(self, mutation_point: int = None, skip_generation: bool = False) -> np.ndarray:

        self.executor.reset(mut_point=mutation_point, skip_generation=skip_generation)
        observation, done, info = self.observe()

        return observation

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        if mode == 'rgb_array':
            return self.executor.image_array
        return None

    def observe(self) -> Tuple[np.ndarray, bool, Dict]:
        return self.executor.observe()

    def close(self) -> None:
        if self.unity_process is not None:
            self.unity_process.quit()
