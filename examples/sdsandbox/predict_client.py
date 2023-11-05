"""
Predict Server
Create a server to accept image inputs and run them against a trained neural network.
This then sends the steering output back to the client.

I advise using the model_1_11.h5 as it achieves the best performance

Author: Tawn Kramer
"""
from __future__ import print_function
import argparse
import time
import base64

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from gym_donkeycar.core.fps import FPSTimer
from gym_donkeycar.core.message import IMesgHandler
from gym_donkeycar.core.sim_client import SimClient
from perturbationdrive import ImagePerturbation

import conf
import models


if tf.__version__ == "1.13.1":
    from tensorflow import ConfigProto, Session

    # Override keras session to work around a bug in TF 1.13.1
    # Remove after we upgrade to TF 1.14 / TF 2.x.
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = Session(config=config)
    keras.backend.set_session(session)


class DonkeySimMsgHandler(IMesgHandler):
    STEERING = 0
    THROTTLE = 1

    def __init__(
        self,
        model,
        constant_throttle,
        image_cb=None,
        rand_seed=0,
        pert_funcs=[],
        attention={},
    ):
        self.model = model
        if attention:
            attention["model"] = model
        self.perturbation = ImagePerturbation(pert_funcs, attention_map=attention)
        self.constant_throttle = constant_throttle
        self.client = None
        self.timer = FPSTimer()
        self.img_arr = None
        # we need this if we want to measure the diff in steering angles
        self.unchanged_img_arr = None
        self.image_cb = image_cb
        self.steering_angle = 0.0
        self.throttle = 0.0
        self.rand_seed = rand_seed
        self.fns = {
            "telemetry": self.on_telemetry,
            "car_loaded": self.on_car_created,
            "on_disconnect": self.on_disconnect,
            "aborted": self.on_aborted,
            "reset_car": self.on_reset_car,
            "quit_app": self.on_quit_app,
            "update": self.on_enque_image,
        }

    def on_connect(self, client):
        self.client = client
        self.timer.reset()

    def on_aborted(self, msg):
        print("aborted")
        self.stop()

    def on_disconnect(self):
        print("disconnected")

    def on_recv_message(self, message):
        self.timer.on_frame()
        if not "msg_type" in message:
            return

        msg_type = message["msg_type"]
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print("unknown message type", msg_type)

    def on_car_created(self, data):
        if self.rand_seed != 0:
            self.send_regen_road(0, self.rand_seed, 1.0)

    def on_telemetry(self, data):
        imgString = data["image"]
        pert_data = {
            "lap": data["lap"],
            "sector": data["sector"],
            "xte": data["cte"],
            "pos_x": data["pos_x"],
            "pos_y": data["pos_y"],
            "pos_z": data["pos_z"],
        }
        # use opencv because it has faster image manipulation and conversion to numpy than PIL
        img_data = base64.b64decode(imgString)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if we want to measure the diff in steering angle
        unchanged_img_arr = np.asarray(image, dtype=np.float32)
        self.unchanged_img_arr = unchanged_img_arr.reshape(
            (1,) + unchanged_img_arr.shape
        )
        # perturb the image
        message = self.perturbation.peturbate(image, pert_data)
        # unpack the function we need next
        func = self.fns[message["func"]]
        new_image = message["image"]
        img_arr = np.asarray(new_image, dtype=np.float32)
        func(image=img_arr)

    def on_enque_image(self, image, *kwargs):
        self.img_arr = image.reshape((1,) + image.shape)
        if self.image_cb is not None:
            self.image_cb(image, self.steering_angle)

    def update(self):
        if self.img_arr is not None:
            self.predict(self.img_arr)
            self.img_arr = None

    def predict(self, image_array):
        outputs = self.model.predict(image_array)
        if self.unchanged_img_arr is not None:
            unchanged_outputs = self.model.predict(self.unchanged_img_arr)
            return self.parse_outputs(outputs, unchanged_outputs)
        self.parse_outputs(outputs)

    def parse_outputs(self, outputs, unchanged_outputs=[]):
        res = []

        # Expects the model with final Dense(2) with steering and throttle
        for i in range(outputs.shape[1]):
            res.append(outputs[0][i])

        self.on_parsed_outputs(res, unchanged_outputs)

    def on_parsed_outputs(self, outputs, unchanged_outputs):
        self.outputs = outputs

        if len(outputs) > 0:
            self.steering_angle = outputs[self.STEERING]
        if self.constant_throttle != 0.0:
            self.throttle = self.constant_throttle
        elif len(outputs) > 1:
            self.throttle = outputs[self.THROTTLE] * conf.throttle_out_scale

        # get normal output
        if len(unchanged_outputs) > 0:
            unchanged_steering = unchanged_outputs[0][self.STEERING]
            diff = abs(self.steering_angle - unchanged_steering)
            self.perturbation.updateSteeringPerformance(diff)
        # self.throttle = 0.0
        self.send_control(self.steering_angle, self.throttle)

    def send_control(self, steer, throttle):
        # print("send st:", steer, "th:", throttle)
        msg = {
            "msg_type": "control",
            "steering": steer.__str__(),
            "throttle": throttle.__str__(),
            "brake": "0.0",
        }
        self.client.queue_message(msg)

    def send_regen_road(self, road_style=0, rand_seed=0, turn_increment=0.0):
        """
        Regenerate the road, where available. For now only in level 0.
        In level 0 there are currently 5 road styles. This changes the texture on the road
        and also the road width.
        The rand_seed can be used to get some determinism in road generation.
        The turn_increment defaults to 1.0 internally. Provide a non zero positive float
        to affect the curviness of the road. Smaller numbers will provide more shallow curves.
        """
        msg = {
            "msg_type": "regen_road",
            "road_style": road_style.__str__(),
            "rand_seed": rand_seed.__str__(),
            "turn_increment": turn_increment.__str__(),
        }

        self.client.queue_message(msg)

    def on_reset_car(self, **kwargs):
        """
        Reset the lap to the start point to use a new perturbation
        """
        msg = {"msg_type": "reset_car"}
        print(f"\n\nSend Reset Car\n\n")
        self.client.queue_message(msg)

    def on_quit_app(self, **kwargs):
        """
        Quits the app and prints autput
        """
        msg = {"msg_type": "quit_app"}
        self.client.queue_message(msg)
        print(f"\n\nQuit App\n\n")
        self.stop()

    def stop(self):
        self.client.stop()
        print("stoped client")

    def __del__(self):
        self.stop()


def clients_connected(arr):
    for client in arr:
        if not client.is_connected():
            return False
    return True


def go(
    filename,
    address,
    constant_throttle=0,
    num_cars=1,
    image_cb=None,
    rand_seed=None,
    pert_funcs=[],
    attention={},
):
    print("loading model", filename)
    model = load_model(filename, compile=False)

    # In this mode, looks like we have to compile it
    model.compile(loss="sgd", metrics=["mse"])

    clients = []

    for _ in range(0, num_cars):
        # setup the clients
        handler = DonkeySimMsgHandler(
            model,
            constant_throttle,
            image_cb=image_cb,
            rand_seed=rand_seed,
            pert_funcs=pert_funcs,
            attention=attention,
        )
        client = SimClient(address, handler)
        clients.append(client)

    while clients_connected(clients):
        try:
            time.sleep(0.02)
            for client in clients:
                client.msg_handler.update()
        except KeyboardInterrupt:
            # unless some hits Ctrl+C and then we get this interrupt
            print("stopping")
            for client in clients:
                client.stop()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prediction server")
    parser.add_argument("--model", type=str, help="model filename")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="server sim host")
    parser.add_argument("--port", type=int, default=9091, help="bind to port")
    parser.add_argument(
        "--num_cars", type=int, default=1, help="how many cars to spawn"
    )
    parser.add_argument(
        "--constant_throttle", type=float, default=0.0, help="apply constant throttle"
    )
    parser.add_argument(
        "--rand_seed", type=int, default=0, help="set road generation random seed"
    )
    parser.add_argument(
        "--perturbation",
        dest="perturbation",
        action="append",
        type=str,
        default=[],
        help="perturbations to use on the model. by default all are used",
    )
    parser.add_argument(
        "--attention_map", type=str, default="", help="which attention map to use"
    )
    parser.add_argument(
        "--attention_threshold",
        type=float,
        default=0.5,
        help="threshold for attention map perturbation",
    )
    parser.add_argument(
        "--attention_layer",
        type=str,
        default="conv2d_5",
        help="layer for attention map perturbation",
    )

    args = parser.parse_args()
    attention = (
        {}
        if args.attention_map == ""
        else {
            "map": args.attention_map,
            "threshold": args.attention_threshold,
            "layer": args.attention_layer,
        }
    )

    address = (args.host, args.port)
    go(
        args.model,
        address,
        args.constant_throttle,
        num_cars=args.num_cars,
        rand_seed=args.rand_seed,
        pert_funcs=args.perturbation,
        attention=attention,
    )
