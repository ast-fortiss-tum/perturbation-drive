# used modules from perturbation drive
from perturbationdrive import (
    GlobalLog,
    ImageCallBack,
)

# used libraries
from gym_donkeycar.core.fps import FPSTimer
from gym_donkeycar.core.message import IMesgHandler
from typing import Any, Union, List, Dict
import base64
import cv2
import numpy as np


class DonkeySimMsgHandler(IMesgHandler):
    """
    This class is used to handle the messages from the simulator and send messages to the simulator.

    This class needs to implement `on_connect`, `on_recv_message` and `on_close`
    """

    STEERING = 0
    THROTTLE = 1

    def __init__(
        self,
        rand_seed=0,
        show_image_cb=True,
    ):
        """
        :param rand_seed:  The random seed used for the simulator
        """
        self.client = None
        self.timer = FPSTimer()
        self.sim_data: Dict = {}
        # we need this if we want to measure the diff in steering angles
        self.unchanged_img_arr = None
        # set the image call back to monitor the data
        if show_image_cb:
            self.image_cb = ImageCallBack()
            # display waiting screen
            self.image_cb.display_waiting_screen()
        else:
            self.image_cb = None
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
        self.logger = GlobalLog("DonkeySimMsgHandler")

    def on_connect(self, client):
        """
        Called when the client connects to the simulator
        """
        self.client = client
        self.timer.reset()

    def on_aborted(self, msg):
        """
        Called when the simulator is aborted
        """
        self.logger.critical(
            f"DonkeySimMsgHandler: {5 * '+'} Warning: Donkey Sim Aborted {5 * '+'}"
        )
        if self.image_cb is not None:
            self.image_cb.display_disconnect_screen()
        self.stop()

    def on_disconnect(self):
        """
        Called when the simulator disconnects
        """
        if self.image_cb is not None:
            self.image_cb.display_disconnect_screen()
        msg = {
            "msg_type": "disconnect",
        }
        self.client.queue_message(msg)
        self.logger.info(
            f"DonkeySimMsgHandler: {5 * '+'} Warning: Donkey Sim Disconnected {5 * '+'}"
        )

    def on_car_created(self, data):
        """
        Called when the car is created
        """
        self.logger.info(
            f"DonkeySimMsgHandler: {5 * '-'} Car Created With Data {data} {5 * '-'}"
        )

    def on_recv_message(self, message):
        """
        This function receives data from the simulator and then runs the tasks here
        """
        self.timer.on_frame()
        if not "msg_type" in message:
            self.logger.info(
                f"DonkeySimMsgHandler: {5 * '+'} Warning: Donkey Sim Message Type not Present {message} {5 * '+'}"
            )
            return

        msg_type = message["msg_type"]
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            self.logger.info(
                f"DonkeySimMsgHandler: {5 * '+'} Warning: Donkey Sim Unknwon Message Type {msg_type} {5 * '+'}"
            )

    def on_telemetry(self, data):
        """
        Called when the simulator sends telemetry data. We use this to get the image and the telemetry data.

        :param data: The telemetry data
        """
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
            "done": data["done"],
            "image": image,
        }

    def update(
        self,
        actions: List[List[float]],
        perturbed_image: Union[any, None],
        perturbation: str = "",
    ) -> Dict[str, Any]:
        """
        We take a action, send the action to the client and then return the latest telemetry data.

        :param actions: The actions to take
        :param perturbed_image: The perturbed image
        :param perturbation: The perturbation function name
        :return: The telemetry data
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
        if perturbed_image is not None and self.image_cb is not None:
            self.image_cb.display_img(
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
        # display waiting screen
        if self.image_cb is not None:
            self.image_cb.display_waiting_screen()

    def reset_car(self):
        """
        Resets the car for a new scenario
        """
        msg = {"msg_type": "reset_car"}
        self.client.queue_message(msg)
        # display waiting screen
        if self.image_cb is not None:
            self.image_cb.display_waiting_screen()

    def on_close(self):
        """
        Called when the simulator closes
        """
        if self.image_cb is not None:
            self.image_cb.display_disconnect_screen()
        self.logger.info(
            f"DonkeySimMsgHandler: {5 * '+'} Warning: Donkey Sim Message Closed by SimClient {5 * '+'}"
        )

    def stop(self):
        """
        Called when the simulator stops
        """
        if self.image_cb is not None:
            self.image_cb.display_disconnect_screen()
        self.logger.info(
            f"DonkeySimMsgHandler: {5 * '+'} Warning: Donkey Sim Message Stoped {5 * '+'}"
        )

    def __del__(self):
        self.stop()
