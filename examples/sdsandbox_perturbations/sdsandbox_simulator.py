# used modules from perturbation drive
from perturbationdrive import PerturbationSimulator
from perturbationdrive.AutomatedDrivingSystem.ADS import ADS
from perturbationdrive.Simulator.Scenario import Scenario, ScenarioOutcome
from perturbationdrive.imageperturbations import ImagePerturbation

# used libraries
from gym_donkeycar.core.fps import FPSTimer
from gym_donkeycar.core.message import IMesgHandler
from gym_donkeycar.core.sim_client import SimClient
from typing import Any, Union, List, Tuple, Dict
import pygame
import time
import base64
import cv2
import numpy as np


class SDSandboxSimulator(PerturbationSimulator):
    def __init__(
        self, simulator_exe_path: str = "", host: str = "127.0.0.1", port: int = 9091
    ):
        super().__init__(
            max_xte=2.0,
            simulator_exe_path=simulator_exe_path,
            host=host,
            port=port,
            initial_pos=None,
        )
        self.client: Union[DonkeySimMsgHandler, None] = None

    def connect(self):
        super().connect()
        address = (self.host, self.port)
        handler = DonkeySimMsgHandler()
        self.client = SimClient(address, handler)

        # wait for the first observation here
        while len(self.client.msg_handler.sim_data) == 0:
            print("SDSandBoxSimulator Waiting for inital obs")
            time.sleep(0.02)
        # the last value if the road width, which should be equivalent to max_xte * 2
        print("We are at ", self.client.msg_handler.sim_data)
        self.initial_pos = (
            self.client.msg_handler.sim_data["pos_x"],
            self.client.msg_handler.sim_data["pos_y"],
            self.client.msg_handler.sim_data["pos_z"],
            self.max_xte * 2,
        )

    def simulate_scanario(
        self, agent: ADS, scenario: Scenario, perturbation_controller: ImagePerturbation
    ) -> ScenarioOutcome:
        waypoints = scenario.waypoints
        perturbation_function_string = scenario.perturbation_function
        perturbation_scale = scenario.perturbation_scale

        # set all params for init loop
        actions = [[0.0, 0.0]]
        perturbed_image = None

        # set up params for saving data
        pos_list = []
        xte_list = []
        actions_list = []
        speed_list = []
        isSuccess = False

        # reset the scene to match the scenario
        self.client.msg_handler.reset_scenario(waypoints)

        # run the scenario
        while self._client_connected(self.client):
            try:
                # TODO: Play around with this value
                time.sleep(0.02)
                # we provide the actions and perturbed image here
                obs: Dict[str, Any] = self.client.msg_handler.update(
                    actions, perturbed_image, perturbation_function_string
                )
                # check if we are done
                if obs["done"]:
                    isSuccess = True
                    break
                elif obs["xte"] > self.max_xte:
                    break

                # perturb the image
                perturbed_image = perturbation_controller.perturbation(
                    obs["image"],
                    perturbation_name=perturbation_function_string,
                    intensity=perturbation_scale,
                )
                # get ads actions
                actions = agent.action(perturbed_image)

                # save data for output
                pos_list.append([obs["pos_x"], obs["pos_y"], obs["pos_z"]])
                xte_list.append(obs["xte"])
                speed_list.append(obs["speed"])
                actions_list.append(actions)

            except KeyboardInterrupt:
                print(f"{5 * '+'} SDSandBox Simulator Got Interrupted {5 * '+'}")
                self.client.stop()
                raise KeyboardInterrupt

        # send reset to sim client
        self.client.msg_handler.reset_car()

        # return the resul of this simulation
        return ScenarioOutcome(
            frames=[x for x in range(len(pos_list))],
            pos=pos_list,
            xte=xte_list,
            speeds=speed_list,
            actions=[(f"{action[0][0]}", f"{action[0][0]}") for action in actions_list],
            scenario=scenario,
            isSuccess=isSuccess,
        )

    def tear_down(self):
        self.client.stop()

    def _client_connected(self, client: SimClient) -> bool:
        """
        Retruns true if the client is still connected
        """
        return client.is_connected()


class DonkeySimMsgHandler(IMesgHandler):
    """
    This class needs to implement `on_connect`, `on_recv_message` and `on_close`
    """

    STEERING = 0
    THROTTLE = 1

    def __init__(
        self,
        rand_seed=0,
    ):
        self.client = None
        self.timer = FPSTimer()
        self.sim_data: Dict = {}
        # we need this if we want to measure the diff in steering angles
        self.unchanged_img_arr = None
        # set the image call back to monitor the data
        image_cb = ImageCallBack()
        self.image_cb = image_cb.display_img
        self.steering_angle = 0.0
        self.throttle = 0.0
        self.rand_seed = rand_seed
        # TcpCarHandler has the calls `telemetry`, `car_loaded`, `scene_selection_ready`
        self.fns = {
            "telemetry": self.on_telemetry,
            "car_loaded": self.on_car_created,
            "on_disconnect": self.on_disconnect,
            "aborted": self.on_aborted,
            "update": self.update,
        }

    def on_connect(self, client):
        self.client = client
        self.timer.reset()

    def on_aborted(self, msg):
        print(f"{5 * '+'} Warning: Donkey Sim Aborted {5 * '+'}")
        self.stop()

    def on_disconnect(self):
        print(f"{5 * '+'} Warning: Donkey Sim Disconnected {5 * '+'}")

    def on_car_created(self, data):
        print(f"{5 * '-'} Car Created With Data {data} {5 * '-'}")

    def on_recv_message(self, message):
        """
        This function receives data from the simulator and then runs the tasks here
        """
        self.timer.on_frame()
        if not "msg_type" in message:
            print(
                f"{5 * '+'} Warning: Donkey Sim Message Type not Present {message} {5 * '+'}"
            )
            return

        msg_type = message["msg_type"]
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print(
                f"{5 * '+'} Warning: Donkey Sim Unknwon Message Type {msg_type} {5 * '+'}"
            )

    def on_telemetry(self, data):
        imgString = data["image"]
        # the sandbox mixes y and z value up, so we fix it here
        img_data = base64.b64decode(imgString)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # stack the image into the image array so we can use it later
        self.sim_data = {
            "xte": data["cte"],
            "pos_x": data["pos_x"],
            "pos_y": data["pos_z"],
            "pos_z": data["pos_y"],
            "speed": data["speed"],
            "done": False,  # TODO: This needs to be send by the simulator
            "image": image,
        }

    def update(
        self,
        actions: List[List[float]],
        perturbed_image: Union[any, None],
        perturbation: str = "",
    ) -> Dict[str, Any]:
        """
        We take a action, send the action to the client and then return the latest telemetry data
        """
        # take the output here to avoid race conditions
        output = self.sim_data

        self.steering_angle = actions[0][0]
        self.throttle = actions[0][1]
        msg = {
            "msg_type": "control",
            "steering": f"{self.steering_angle}",
            "throttle": f"{self.throttle}",
            "brake": "0.0",
        }

        self.client.queue_message(msg)
        # run the image call back to inspect the image
        if perturbed_image is not None:
            self.image_cb(
                perturbed_image,
                f"{self.steering_angle}",
                f"{self.throttle}",
                perturbation,
            )
        # return the sim_data so we can perturb it in the main loop and get a control action
        return output

    def reset_scenario(self, waypoints: Union[str, None]):
        """
        Sends a new road to the sim
        """

        msg = {
            "msg_type": "regen_road",
            "wayPoints": waypoints.__str__(),
        }

        self.client.queue_message(msg)

    def reset_car(self):
        """
        Resets the car for a new scenario
        """
        # TODO: Test this
        msg = {"msg_type": "quit_app"}
        self.client.queue_message(msg)

    def on_close(self):
        print(f"{5 * '+'} Warning: Donkey Sim Message Closed by SimClient {5 * '+'}")

    def __del__(self):
        self.stop()


class ImageCallBack:
    def __init__(self):
        pygame.init()
        ch, row, col = 3, 240, 320

        size = (col * 2, row * 2)
        pygame.display.set_caption("sdsandbox image monitor")
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
        self.screen_print(10, 10, "NN(steering): " + steering, self.screen)
        self.screen_print(10, 25, "NN(throttle): " + throttle, self.screen)
        self.screen_print(10, 40, "Perturbation: " + perturbation, self.screen)
        pygame.display.flip()
