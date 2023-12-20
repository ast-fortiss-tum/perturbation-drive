# used modules from perturbation drive
from numpy import ndarray, uint8
from perturbationdrive import PerturbationSimulator
from perturbationdrive.AutomatedDrivingSystem.ADS import ADS
from perturbationdrive.Simulator.Scenario import Scenario, ScenarioOutcome
from perturbationdrive.imageperturbations import ImagePerturbation

# used libraries
from udacity_utils.envs.udacity.udacity_gym_env import UdacityGymEnv_RoadGen
from typing import Union
import cv2
import gym
import numpy as np
import pygame


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

    def connect(self):
        super().connect()
        self.client = UdacityGymEnv_RoadGen(
            seed=1,
            exe_path=self.simulator_exe_path,
        )
        # set initial pos
        obs, done, info = self.client.observe()
        x, y, z = info["pos"]
        self.initial_pos = (x, y, z, 2 * self.max_xte)

    def simulate_scanario(
        self, agent: ADS, scenario: Scenario, perturbation_controller: ImagePerturbation
    ) -> ScenarioOutcome:
        waypoints = scenario.waypoints
        perturbation_function_string = scenario.perturbation_function
        perturbation_scale = scenario.perturbation_scale

        # set up image monitor
        monitor = ImageCallBack()

        # set all params for init loop
        actions = [[0.0, 0.0]]
        perturbed_image = None

        # set up params for saving data
        pos_list = []
        xte_list = []
        actions_list = []
        isSuccess = False
        done = False

        # reset the scene to match the scenario
        # Road generatior ir none because we currently do not build random roads
        obs: ndarray[uint8] = self.client.reset(
            skip_generation=False, track_string=waypoints
        )
        # TODO: Resetting does not work!

        # action loop
        while not done:
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
                    actions, self.client.action_space.low, self.client.action_space.high
                )

            monitor.display_img(perturbed_image, actions[0][0], actions[0][1], perturbation_function_string)
            # obs is the image, info contains the road and the position of the car
            obs, done, info = self.client.step(actions)

            # log new info
            pos_list.append(info["pos"])
            xte_list.append(info["cte"])
            actions_list.append(actions)

        # determine if we were successful
        isSuccess = max([abs(xte) for xte in xte_list]) < self.max_xte
        print(f"{5 * '-'} Finished udacity scenario: {isSuccess} {5 * '_'}")
        # reset for the new track
        _ = self.client.reset(
            skip_generation=False, track_string=waypoints
        )
        # return the scenario output
        return ScenarioOutcome(
            frames=[x for x in range(len(pos_list))],
            pos=pos_list,
            xte=xte_list,
            speeds=[],
            actions=actions_list,
            scenario=scenario,
            isSuccess=isSuccess,
        )

    def tear_down(self):
        self.client.close()


class ImageCallBack:
    def __init__(self):
        pygame.init()
        ch, row, col = 3, 240, 320

        size = (col * 2, row * 2)
        pygame.display.set_caption("udacity sim image monitor")
        self.screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
        self.camera_surface = pygame.surface.Surface((col, row), 0, 24).convert()
        self.myfont = pygame.font.SysFont("monospace", 15)

    def screen_print(self, x, y, msg, screen):
        label = self.myfont.render(msg, 1, (255, 255, 0))
        screen.blit(label, (x, y))

    def display_img(self, img, steering, throttle, perturbation):
        # swap image axis
        img = img.swapaxes(0, 1)
        # draw frame
        pygame.surfarray.blit_array(self.camera_surface, img)
        camera_surface_2x = pygame.transform.scale2x(self.camera_surface)
        self.screen.blit(camera_surface_2x, (0, 0))
        # steering and throttle value
        self.screen_print(10, 10, "NN(steering): " + str(steering), self.screen)
        self.screen_print(10, 25, "NN(throttle): " + str(throttle), self.screen)
        self.screen_print(10, 40, "Perturbation: " + perturbation, self.screen)
        pygame.display.flip()
