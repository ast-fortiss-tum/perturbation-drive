"""
MIT License

Copyright (c) 2018 Roma Sokolkov
Copyright (c) 2018 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Original author: Tawn Kramer

import base64
import time
from io import BytesIO
from threading import Thread
from typing import Tuple, Dict, List, Union

import numpy as np
import socketio
from PIL import Image
from flask import Flask

from examples.udacity.udacity_utils.envs.udacity.config import INPUT_DIM, MAX_CTE_ERROR
from examples.udacity.udacity_utils.envs.udacity.core.client import start_app
from examples.udacity.udacity_utils.global_log import GlobalLog

sio = socketio.Server()
flask_app = Flask(__name__)

last_obs = None
is_connect = False
deployed_track_string = None
generated_track_string = None
steering = 0.0
throttle = 0.0
speed = 0.0
cte = 0.0
angle = 0.0
cte_pid = 0.0
hit = 0.0
done = False
image_array = None
track_sent = False
weather_sent = False
pos_x = 0.0
pos_y = 0.0
pos_z = 0.0
ori_1 = 0.0
ori_2 = 0.0
ori_3 = 0.0
ori_4 = 0.0
ori_5 = 0.0
ori_6 = 0.0
ori_7 = 0.0
udacity_unreactiv = False
weather_recieved = False
weather="Sun"
intensity=90



@sio.on("connect")
def connect(sid, environ) -> None:
    global is_connect
    is_connect = True
    print("Connect to Udacity simulator: {}".format(sid))
    send_control(steering_angle=0, throttle_command=0)


def send_control(steering_angle: float, throttle_command: float) -> None:
    sio.emit(
        "steer",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle_command.__str__(),
        },
        skip_sid=True,
    )
    global udacity_unreactiv
    global speed

    if throttle_command >= 0.01 and round(speed, 1) == 0.0:
        print(f"Warning: Throttle is {throttle_command} but speed is {speed}\n")
        udacity_unreactiv = True
    if speed > 0.0 and udacity_unreactiv:
        print("Warning: Udacity is reactivated\n")
        udacity_unreactiv = False


def send_track(track_string: str) -> None:
    global track_sent
    if not track_sent:
        sio.emit("track", data={"track_string": track_string}, skip_sid=True)
        track_sent = True
        print("SendTrack", end="\n", flush=True)
    else:
        print("Track already sent", end="\n", flush=True)


def send_reset() -> None:
    sio.emit("reset", data={}, skip_sid=True)
    print("Reset", end="\n", flush=True)

def send_weather(track_string,rate) -> None:
    global weather_sent
    if not weather_sent:
        sio.emit("weather", data={"type": track_string,"rate": rate.__str__()}, skip_sid=True)
        weather_sent = True
        print("Weather sent", end="\n", flush=True)
    else:
        print("Weather already sent", end="\n", flush=True)


@sio.on("telemetry")
def telemetry(sid, data) -> None:
    global steering
    global throttle
    global speed
    global cte
    global hit
    global image_array
    global deployed_track_string
    global generated_track_string
    global done
    global cte_pid
    global angle
    global pos_x
    global pos_y
    global pos_z
    global udacity_unreactiv
    global weather_recieved
    global weather_sent
    global weather
    global intensity
    global ori_1
    global ori_2
    global ori_3
    global ori_4
    global ori_5
    global ori_6
    global ori_7

    if data:
        speed = float(data["speed"]) * 3.6  # conversion m/s to km/h
        cte = float(data["cte"])
        cte_pid = float(data["cte_pid"])
        pos_x = float(data["pos_x"])
        pos_y = float(data["pos_y"])
        pos_z = float(data["pos_z"])
        ori_1 = float(data["angle1"])
        ori_2 = float(data["angle2"])
        ori_3 = float(data["angle3"])
        ori_4 = float(data["angle4"])
        ori_5 = float(data["angle5"])
        ori_6 = float(data["angle6"])
        ori_7 = float(data["angle7"])
        hit = data["hit"]
        angle = data["angl"]
        deployed_track_string = data["track"]
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image_array = np.copy(np.array(image))

        if done:
            send_reset()
        elif generated_track_string is not None and not track_sent:
            send_track(track_string=generated_track_string)
            time.sleep(0.5)
        elif weather_recieved and not weather_sent:
            send_weather(weather,intensity)
        else:
            send_control(steering_angle=steering, throttle_command=throttle)
    else:
        print("Wawrning: Udacity data is None")
    if udacity_unreactiv:
        print(f"Warning: Udacity Non Reactive, received {data} from sid {sid}\n")


class UdacitySimController:
    """
    Wrapper for communicating with unity simulation.
    """

    def __init__(
        self,
        port: int,
    ):
        self.port = port
        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM
        self.max_cte_error = MAX_CTE_ERROR

        self.is_success = 0
        self.current_track = None
        self.image_array = np.zeros(self.camera_img_size)

        self.logger = GlobalLog("UdacitySimController")

        self.client_thread = Thread(target=start_app, args=(flask_app, sio, self.port))
        self.client_thread.daemon = True
        self.client_thread.start()
        self.logger = GlobalLog("UdacitySimController")

        while not is_connect:
            time.sleep(0.3)

    def reset(
        self, skip_generation: bool = False, track_string: Union[str, None] = None
    ) -> None:
        global last_obs
        global speed
        global throttle
        global steering
        global image_array
        global hit
        global cte
        global angle
        global cte_pid
        global done
        global generated_track_string
        global track_sent
        global pos_x
        global pos_y
        global pos_z
        global ori_1
        global ori_2
        global ori_3
        global ori_4
        global ori_5
        global ori_6
        global ori_7
        global weather_sent

        last_obs = None
        speed = 0.0
        throttle = 0.0
        steering = 0.0
        self.image_array = np.zeros(self.camera_img_size)
        hit = "none"
        cte = 0.0
        angle = 0.0
        cte_pid = 0.0
        done = False
        generated_track_string = None
        track_sent = False
        pos_x = 0.0
        pos_y = 0.0
        pos_z = 0.0
        ori_1 = 0.0
        ori_2 = 0.0
        ori_3 = 0.0
        ori_4 = 0.0
        ori_5 = 0.0
        ori_6 = 0.0
        ori_7 = 0.0

        weather_sent = False

        self.is_success = 0
        self.current_track = None

        if not skip_generation and track_string is not None:
            generated_track_string = track_string

        time.sleep(1)

    def generate_track(self, track_string: Union[str, None] = None):
        global generated_track_string

        if track_string is not None:
            generated_track_string = track_string

    @staticmethod
    def take_action(action: np.ndarray) -> None:
        global throttle
        global steering
        steering = action[0][0]
        throttle = action[0][1]

    def weather(self, weather_string: str = "Sun", intensity_in: int = 90):
        global weather
        global intensity
        global weather_recieved
        weather_recieved= True
        weather = weather_string
        intensity = intensity_in

    def observe(self) -> Tuple[np.ndarray, bool, Dict]:
        global last_obs
        global image_array
        global done
        global speed
        global cte_pid
        global pos_x
        global pos_y
        global pos_z
        global ori_1
        global ori_2
        global ori_3
        global ori_4
        global ori_5
        global ori_6
        global ori_7
        
        global cte
        global angle

        while last_obs is image_array:
           time.sleep(1.0 / 120.0)
        #    print("Waiting for new image")

        last_obs = image_array
        self.image_array = image_array

        done = self.is_game_over()
        # z and y coordinates appear to be switched
        info = {
            "is_success": self.is_success,
            "track": self.current_track,
            "speed": speed,
            "pos": [pos_x, pos_z, pos_y],
            "orientation": [ori_1, ori_2, ori_3, ori_4],
            "orientation_euler": [ori_5,ori_6,ori_7],
            "cte": cte,
            "cte_pid": cte,
            "angle": angle,
        }

        return last_obs, done, info

    def quit(self):
        self.logger.info("Stopping client")

    def is_game_over(self) -> bool:
        global cte
        global hit
        global speed

        if abs(cte) > self.max_cte_error or hit != "none":
            if abs(cte) > self.max_cte_error:
                self.is_success = 0
            else:
                self.is_success = 1
            return True
        return False
