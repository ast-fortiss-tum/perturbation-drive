# used modules from perturbation drive
from numpy import ndarray, uint8
from perturbationdrive import (
    PerturbationSimulator,
    ADS,
    Scenario,
    ScenarioOutcome,
    ImageCallBack,
    ImagePerturbation,
    GlobalLog as Gl,
)
import traceback

# used libraries
from examples.udacity.udacity_utils.envs.udacity.udacity_gym_env import (
    UdacityGymEnv_RoadGen,
)
from typing import Union
import cv2
import gym
import numpy as np
import time


class UdacitySimulator(PerturbationSimulator):
    def __init__(
        self,
        simulator_exe_path: str = "./examples/udacity/udacity_utils/sim/udacity_sim.app",
        host: str = "127.0.0.1",
        port: int = 9091,
    ):
        # udacity road is 8 units wide
        super().__init__(
            max_xte=4.0,
            simulator_exe_path=simulator_exe_path,
            host=host,
            port=port,
            initial_pos=None,
        )
        self.client: Union[UdacityGymEnv_RoadGen, None] = None
        self.logger = Gl("UdacitySimulator")

    def connect(self):
        super().connect()
        self.client = UdacityGymEnv_RoadGen(
            seed=1,
            exe_path=self.simulator_exe_path,
        )
        self.client.reset()
        time.sleep(2)
        self.logger.info("Connected to Udacity Simulator")
        # set initial pos
        obs, done, info = self.client.observe()
        x, y, z = info["pos"]
        if self.initial_pos is None:
            self.initial_pos = (x, y, z, 2 * self.max_xte)
        self.logger.info(f"Initial pos: {self.initial_pos}")

    def simulate_scanario(
        self, agent: ADS, scenario: Scenario, perturbation_controller: ImagePerturbation
    ) -> ScenarioOutcome:
        try:
            waypoints = scenario.waypoints
            perturbation_function_string = scenario.perturbation_function
            perturbation_scale = scenario.perturbation_scale

            # set up image monitor
            # monitor = ImageCallBack()
            # monitor.display_waiting_screen()
            self.logger.info(f"{5 * '-'} Starting udacity scenario {5 * '_'}")

            # set all params for init loop
            actions = [[0.0, 0.0]]
            perturbed_image = None

            # set up params for saving data
            pos_list = []
            xte_list = []
            actions_list = []
            speed_list = []
            isSuccess = False
            done = False
            timeout = False

            # reset the scene to match the scenario
            # Road generatior ir none because we currently do not build random roads
            obs: ndarray[uint8] = self.client.reset(
                skip_generation=False, track_string=waypoints
            )
            start_time = time.time()

            # action loop
            while not done:
                if time.time() - start_time > 62:
                    self.logger.info("SDSandBox: Timeout after 120s")
                    timeout = True
                    break

                obs = cv2.resize(obs, (320, 240), cv2.INTER_NEAREST)

                # perturb the image
                perturbed_image = perturbation_controller.perturbation(
                    obs, perturbation_function_string, perturbation_scale
                )

                # agent makes a move, the agent also selects the dtype and adds a batch dimension
                actions = agent.action(perturbed_image)

                # clip action to avoid out of bound errors
                if isinstance(self.client.action_space, gym.spaces.Box):
                    actions = np.clip(
                        actions,
                        self.client.action_space.low,
                        self.client.action_space.high,
                    )

                # monitor.display_img(
                #    perturbed_image,
                #    f"{actions[0][0]}",
                #    f"{actions[0][1]}",
                #    perturbation_function_string,
                # )
                # obs is the image, info contains the road and the position of the car
                obs, done, info = self.client.step(actions)

                # log new info
                pos_list.append(info["pos"])
                xte_list.append(info["cte"])
                speed_list.append(info["speed"])
                actions_list.append(actions)

            # determine if we were successful
            isSuccess = max([abs(xte) for xte in xte_list]) < self.max_xte
            self.logger.info(
                f"{5 * '-'} Finished udacity scenario: {isSuccess} {5 * '_'}"
            )
            # monitor.display_disconnect_screen()
            # monitor.destroy()

            # reset for the new track
            _ = self.client.reset(skip_generation=False, track_string=waypoints)
            # return the scenario output
            return ScenarioOutcome(
                frames=[x for x in range(len(pos_list))],
                pos=pos_list,
                xte=xte_list,
                speeds=speed_list,
                actions=actions_list,
                scenario=scenario,
                isSuccess=isSuccess,
                timeout=timeout,
            )
        except Exception as e:
            # close the simulator
            self.tear_down()
            traceback.print_stack()
            # throw the exception
            raise e

    def tear_down(self):
        self.client.close()

    def name(self) -> str:
        return "UdacitySimualtorAdapter"
