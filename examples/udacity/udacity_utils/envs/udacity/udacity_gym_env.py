# Original author: Roma Sokolkov
# Edited by Antonin Raffin
import os
import time
from typing import Optional, Tuple, Dict, Union

import gym
import numpy as np
from gym import spaces

from examples.udacity.udacity_utils.envs.udacity.config import BASE_PORT, MAX_STEERING, INPUT_DIM
from examples.udacity.udacity_utils.envs.udacity.core.udacity_sim import UdacitySimController
from examples.udacity.udacity_utils.envs.unity_proc import UnityProcess
from examples.udacity.udacity_utils.global_log import GlobalLog


class UdacityGymEnv_RoadGen(gym.Env):
    """
    Gym interface for udacity simulator
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        seed: int,
        headless: bool = False,
        exe_path: str = None,
    ):
        self.seed = seed
        self.exe_path = exe_path
        self.logger = GlobalLog("UdacityGymEnv_RoadGen")
        if headless:
            self.logger.warn("Headless mode not supported with Udacity")
        self.headless = False
        self.port = BASE_PORT

        self.unity_process = None
        if self.exe_path is not None:
            self.logger.info("Starting UdacityGym env")

            # remove if it works
            current_path = os.getcwd()
            # print(f"Current Directory: {current_path}")
            # files = [f for f in os.listdir("./examples/udacity/udacity_utils/sim")]
            # print(f"Files in the directory: {files}")

            # assert os.path.exists(self.exe_path), "Path {} does not exist".format(
            #     self.exe_path
            # )
            # Start Unity simulation subprocess if needed
            self.unity_process = UnityProcess()
            self.unity_process.start(
                sim_path=self.exe_path, headless=headless, port=self.port
            )
            time.sleep(
                2
            )  # wait for the simulator to start and the scene to be selected

        self.executor = UdacitySimController(port=self.port)

        # steering + throttle, action space must be symmetric
        self.action_space = spaces.Box(
            low=np.array([-MAX_STEERING, 0]),
            high=np.array([MAX_STEERING, 1]),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0, high=255, shape=INPUT_DIM, dtype=np.uint8
        )

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

    def reset(
        self,
        skip_generation: bool = False,
        track_string: Union[str, None] = None,
    ) -> np.ndarray:
        self.executor.reset(
            skip_generation=skip_generation,
            track_string=track_string,
        )
        observation, done, info = self.observe()
        time.sleep(2)
        return observation

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "rgb_array":
            return self.executor.image_array
        return None

    def observe(self) -> Tuple[np.ndarray, bool, Dict]:
        return self.executor.observe()
    
    def weather(self, weather_string: str = "Sun", intensity_in: int = 90) -> Optional[np.ndarray]:
        self.executor.weather(weather_string=weather_string, intensity_in=intensity_in)
        return None

    def close(self) -> None:
        if self.unity_process is not None:
            self.unity_process.quit()
