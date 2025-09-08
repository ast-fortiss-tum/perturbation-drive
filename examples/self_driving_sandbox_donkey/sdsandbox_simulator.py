# used modules from perturbation drive
import tensorflow as tf
from perturbationdrive import (
    PerturbationSimulator,
    ADS,
    Scenario,
    ScenarioOutcome,
    ImagePerturbation,
    GlobalLog,
)

# used libraries
from gym_donkeycar.core.sim_client import SimClient
from typing import Any, Union, List, Tuple, Dict
import time

from examples.self_driving_sandbox_donkey.donkey_exec import DonkeyProcess
from examples.self_driving_sandbox_donkey.donkey_sim_msg_handler import (
    DonkeySimMsgHandler,
)
from typing import Union

import numpy as np
import time
import math
WAYPOINT_THRESHOLD = 1
ANGLE_THRESHOLD = 0
PID=False


class Waypoint_control_utils():
        def __init__(self,treshold,angle_treshold):
            self.waypoint_treshold=treshold
            self.angle_treshold=angle_treshold

        def angle_difference(self,a1, a2):
            diff = a1 - a2
            if diff>=180:
                diff-=360
            elif diff<=-180:
                diff+=360

            
            return diff
        
        def convert_waypoints(self,input_string):
            # Split the input string by the '@' symbol to get individual waypoints
            waypoints = input_string.split('@')
            
            # Initialize an empty list to hold the converted waypoints
            waypoint_list = []
            
            # Iterate through each waypoint string
            for waypoint in waypoints:
                # Split the string by the ',' symbol to get x, y, z values
                x, z, y = waypoint.split(',')
                
                # Convert x, y, z to floats and rearrange to [x, y, z]
                waypoint_list.append([float(x), float(y)])

            # waypoint_list,_=self.generate_road_margins(waypoint_list,1)
            
            return waypoint_list[1:]
        
        def generate_road_margins(self,road_points, offset):
            left_margins = []
            right_margins = []

            num_points = len(road_points)

            # Calculate the direction vectors for each road segment
            direction_vectors = []
            for i in range(num_points - 1):
                dx = road_points[i + 1][0] - road_points[i][0]
                dy = road_points[i + 1][1] - road_points[i][1]
                mag = np.sqrt(dx ** 2 + dy ** 2)
                direction_vectors.append((dx / mag, dy / mag))

            # Average neighboring direction vectors to get smoother normals
            averaged_directions = []
            for i in range(num_points - 1):
                if i == 0:
                    averaged_directions.append(direction_vectors[0])
                elif i == num_points - 2:
                    averaged_directions.append(direction_vectors[-1])
                else:
                    averaged_directions.append(((direction_vectors[i][0] + direction_vectors[i - 1][0]) / 2,
                                                (direction_vectors[i][1] + direction_vectors[i - 1][1]) / 2))

            # Calculate normals and generate margins
            for i in range(num_points - 1):
                dx, dy = averaged_directions[i]
                nx = -dy
                ny = dx

                left_x = road_points[i][0] + offset * nx
                left_y = road_points[i][1] + offset * ny
                right_x = road_points[i][0] - offset * nx
                right_y = road_points[i][1] - offset * ny

                left_margins.append([left_x, left_y])
                right_margins.append([right_x, right_y])

            return left_margins, right_margins
    
        def angle_extraction(self, x1, y1, z1, x2, y2, z2):

            # Calculate the distances between points
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1

            # Calculate the angles on each axis
            angle_x_axis = math.atan2(dy, dz)
            angle_y_axis = math.atan2(dx, dz)
            angle_z_axis = math.atan2(dy, dx)

            # Convert angles from radians to degrees
            angle_x_axis_degrees = math.degrees(angle_x_axis)
            angle_y_axis_degrees = math.degrees(angle_y_axis)
            angle_z_axis_degrees = math.degrees(angle_z_axis)
            return angle_x_axis_degrees, angle_y_axis_degrees, angle_z_axis_degrees
        
        def exponential_increase(self,number, factor):
            return factor * (1 - np.exp(-number))

        def calculate_control(self, x_target, y_target, simulator_pose, simulator_orientation):
            x_cur, y_cur, _ = simulator_pose

            #print(f"Position\nx:{x_cur}, y:{y_cur}")
            _, angle_cur, _ = simulator_orientation
            print(f"Orientation: {round(angle_cur, 3)}, {round(math.radians(angle_cur), 3)} rad")
            distance = math.sqrt((x_target - x_cur)**2 + (y_target - y_cur)**2)
            #print(f"dist {distance}")
            _, angle_y_axis_degrees, _=self.angle_extraction(x_cur, 0.0, y_cur, x_target, 0.0, y_target)
            # print(f"Angle to goal: {angle_y_axis_degrees}")
            angle_difference=self.angle_difference(angle_cur,angle_y_axis_degrees)
            # print(f"angle diff {angle_difference}")
            steering = (math.radians(-angle_difference)*15)
            print(f"angle diff: {angle_difference}, steering: {steering}")
            throttle=distance/10

            return steering, throttle, distance, angle_difference,angle_y_axis_degrees
        
        def calculate_distance(self, x_target, y_target, simulator_pose):
            x_cur, y_cur, _ = simulator_pose
            distance = math.sqrt((x_target - x_cur)**2 + (y_target - y_cur)**2)
            return distance

        def is_waypoint_in_back(self,current_pos, current_orientation, waypoint_x,waypoint_y):
            current_x, current_y, _ = current_pos
            _, angle_cur, _ = current_orientation
            current_orientation=math.radians(angle_cur)
            
            # Calculate vector to the next waypoint
            waypoint_vector = (waypoint_x - current_x, waypoint_y - current_y)
            
            # Calculate vehicle forward vector based on its orientation
            vehicle_forward_vector = (math.cos(current_orientation), math.sin(current_orientation))
            
            # Calculate the dot product
            dot_product = waypoint_vector[0] * vehicle_forward_vector[0] + waypoint_vector[1] * vehicle_forward_vector[1]
            
            # If the dot product is positive, the waypoint is in front; otherwise, it's behind
            return dot_product <= 0


def pid_speed20(road_error, angle_error, speed_error, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error):
    
    road_error=-road_error
    if abs(road_error)>1:
        Kp_road = 0.6
    else:
        Kp_road = 0.45
    
    Ki_road = 0.0
    Kd_road = 0.0001
    
    if angle_error<25:
        Kp_angle = 0.003
        Kd_angle = 0.002
    else:
        Kp_angle = 0.001
        Kd_angle = 0.002

    Ki_angle = 0.0
    

    Kp_speed = 0.1
    Ki_speed = 0.0 
    Kd_speed = 0.0
    
    P_angle = Kp_angle * angle_error
    I_angle = Ki_angle * total_angle_error
    D_angle = Kd_angle * (angle_error - prev_angle_error)

    P_road = Kp_road * road_error
    I_road = Ki_road * total_road_error
    D_road= Kd_road * (road_error - prev_road_error)

    
    
    steering = P_angle + I_angle + D_angle 
    steering =  P_road + I_road + D_road + steering

    steering = max(-1, min(1, steering))

    P_speed = Kp_speed * speed_error
    I_speed = Ki_speed * total_speed_error
    D_speed = Kd_speed * (speed_error - prev_speed_error)
    throttle = P_speed + I_speed + D_speed
    throttle -= 0.6 * abs(road_error)
    throttle = max(0.05, min(0.8, throttle))



    # print(f"s: {steering}, th: {throttle}, kp angle: {P_angle + I_angle + D_angle}, Kp road: {P_road + I_road + D_road}")
    
    prev_road_error=road_error
    prev_angle_error=angle_error
    prev_speed_error=speed_error
    total_road_error+=road_error
    total_angle_error+=angle_error
    total_speed_error+=speed_error
    return throttle, steering, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error

def pid_speed21(road_error, angle_error, speed_error, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error):
    
    road_error=-road_error
    if abs(road_error)>1:
        Kp_road = 0.6
    else:
        Kp_road = 0.45
    
    Ki_road = 0.0
    Kd_road = 0.000
    
    if angle_error<25:
        Kp_angle = 0.003
        Kd_angle = 0.000
    else:
        Kp_angle = 0.001
        Kd_angle = 0.000

    Ki_angle = 0.0
    

    Kp_speed = 0.1
    Ki_speed = 0.0 
    Kd_speed = 0.0
    
    P_angle = Kp_angle * angle_error
    I_angle = Ki_angle * total_angle_error
    D_angle = Kd_angle * (angle_error - prev_angle_error)

    P_road = Kp_road * road_error
    I_road = Ki_road * total_road_error
    D_road= Kd_road * (road_error - prev_road_error)

    
    
    steering = P_angle + I_angle + D_angle 
    steering =  P_road + I_road + D_road + steering

    steering = max(-1, min(1, steering))

    P_speed = Kp_speed * speed_error
    I_speed = Ki_speed * total_speed_error
    D_speed = Kd_speed * (speed_error - prev_speed_error)
    throttle = P_speed + I_speed + D_speed
    throttle -= 0.6 * abs(road_error)
    throttle = max(0.1, min(0.8, throttle))



    # print(f"s: {steering}, th: {throttle}, kp angle: {P_angle + I_angle + D_angle}, Kp road: {P_road + I_road + D_road}")
    
    prev_road_error=road_error
    prev_angle_error=angle_error
    prev_speed_error=speed_error
    total_road_error+=road_error
    total_angle_error+=angle_error
    total_speed_error+=speed_error
    return throttle, steering, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error



def pid_speed25(test,road_error, angle_error, speed_error, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error):
    
    road_error=-road_error
    # if abs(road_error)>1:
    Kp_road = 0.65
    # else:
    #     Kp_road = 0.3
    
    Ki_road = 0.0
    if test:
        Kd_road = 0.0
    else:
        Kd_road = 0.4
    
    Kp_angle = 0.03
    Ki_angle = 0.0
    if test:
        Kd_angle = 0.00
    else:
        Kd_angle = 0.04

    Kp_speed = 0.1
    Ki_speed = 0.0
    if test:
        Kd_speed = 0.0
    else:
        Kd_speed = 0.1
    
    P_angle = Kp_angle * angle_error
    I_angle = Ki_angle * total_angle_error
    D_angle = Kd_angle * (angle_error - prev_angle_error)

    P_road = Kp_road * road_error
    I_road = Ki_road * total_road_error
    D_road= Kd_road * (road_error - prev_road_error)

    
    
    steering = P_angle + I_angle + D_angle 
    steering =  P_road + I_road + D_road + steering

    steering = max(-1, min(1, steering))

    P_speed = Kp_speed * speed_error
    I_speed = Ki_speed * total_speed_error
    D_speed = Kd_speed * (speed_error - prev_speed_error)
    throttle = P_speed + I_speed + D_speed
    throttle -= 0.6 * abs(road_error)
    throttle = max(0.01, min(0.8, throttle))



    # print(f"s: {steering}, th: {throttle}, kp angle: {P_angle + I_angle + D_angle}, Kp road: {P_road + I_road + D_road}")
    
    prev_road_error=road_error
    prev_angle_error=angle_error
    prev_speed_error=speed_error
    total_road_error+=road_error
    total_angle_error+=angle_error
    total_speed_error+=speed_error
    return throttle, steering, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error



class SDSandboxSimulator(PerturbationSimulator):
    def __init__(
        self,
        simulator_exe_path: str = "./sim/donkey-sim.app",
        host: str = "127.0.0.1",
        port: int = 9091,
        show_image_cb=True,
    ):
        super().__init__(
            max_xte=2.0,
            simulator_exe_path=simulator_exe_path,
            host=host,
            port=port,
            initial_pos=None,
        )
        self.port = port
        self.client: Union[DonkeySimMsgHandler, None] = None
        self.process: DonkeyProcess = DonkeyProcess()
        self.logger = GlobalLog("SDSandBoxSimulator")
        self.show_image_cb = show_image_cb

    def connect(self):
        # launch the sim binary here
        self.process.start(self.simulator_exe_path, port=self.port)

        super().connect()
        address = (self.host, self.port)
        handler = DonkeySimMsgHandler(show_image_cb=self.show_image_cb)
        self.client = SimClient(address, handler)

        # wait for the first observation here
        while len(self.client.msg_handler.sim_data) == 0:
            self.logger.info("Waiting for inital obs")
            time.sleep(0.04)
        # the last value if the road width, which should be equivalent to max_xte * 2
        self.initial_pos = (
            self.client.msg_handler.sim_data["pos_x"],
            self.client.msg_handler.sim_data["pos_y"],
            self.client.msg_handler.sim_data["pos_z"],
            self.max_xte * 2,
        )

    def simulate_scanario(
        self, agent: Union[ADS,None], 
        scenario: Scenario, 
        perturbation_controller: Union[ImagePerturbation,None], 
        perturb=False, 
        model_drive=False, 
        weather=None, 
        intensity=None
    ) -> ScenarioOutcome:
        try:
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
            pid_list = []
            speed_list = []
            original_image_list=[]
            perturbed_image_list=[]
            isSuccess = False
            done = False
            timeout = False

            prev_road_error = 0.0
            total_road_error = 0.0
            prev_angle_error = 0.0
            total_angle_error = 0.0
            prev_speed_error = 0.0
            total_speed_error = 0.0

            # reset the scene to match the scenario
            self.client.msg_handler.reset_scenario(waypoints)
            self.logger.info(f"Reset the scenario")
            if weather:
                print("Donkeysim does not support weather yet")
            time.sleep(2.0)
            start_time = time.time()
            target_speed=2.0
            # target_speed=25.0
            prev_throttle =  0.0
            prev_steering = 0.0
            waypoint_controller = Waypoint_control_utils(WAYPOINT_THRESHOLD, ANGLE_THRESHOLD)

            current_waypoint_index=0
            waypoint_list=waypoint_controller.convert_waypoints(waypoints)
            waypoint_list=waypoint_list
            counter=0

            # run the scenario
            while self._client_connected(self.client):
                try:
                    # TODO: Play around with this value
                    # time.sleep(0.01)
                    # we provide the actions and perturbed image here
                    obs: Dict[str, Any] = self.client.msg_handler.update(
                        actions, perturbed_image, perturbation_function_string
                    )
                    # check if we are done
                    if obs["done"]:
                        isSuccess = True
                        self.logger.info("SDSandBox: Done")
                        break
                    elif abs(obs["xte"]) > self.max_xte:
                        break
                    elif time.time() - start_time > 100:
                        self.logger.info("SDSandBox: Timeout after 100s")
                        timeout = True
                        break
                    original_image_list.append(obs["image"])

                    if perturb:
                        # perturb the image
                        perturbed_image = perturbation_controller.perturbation(
                            obs["image"],
                            perturbation_name=perturbation_function_string,
                            intensity=perturbation_scale,
                        )
                        image=perturbed_image
                    else:
                        image=obs["image"]
                    
                    road_error=float(obs['xte'])
                    angle_error=float(0) #TODO
                    speed_error=target_speed-float(obs['speed'])
                    # print(obs)
                    
                    rotation=obs["orientation_euler"]
                        
                    if  current_waypoint_index < len(waypoint_list):
                        current_waypoint = waypoint_list[current_waypoint_index]
                    x, y = current_waypoint
                    pose = [obs["pos_x"], obs["pos_y"], obs["pos_z"]]
                    
                    if any(rotation):
                        steering, throttle, dist, angl_diff,angle = waypoint_controller.calculate_control(x, y, pose, rotation)  
                        # dist=1
                        if dist <= WAYPOINT_THRESHOLD:
                            current_waypoint_index += 1
                            if  current_waypoint_index < len(waypoint_list):
                                current_waypoint = waypoint_list[current_waypoint_index]
                            x, y = current_waypoint
                            pose = [obs["pos_x"], obs["pos_y"], obs["pos_z"]]
                            steering,throttle , dist, angl_diff,angle = waypoint_controller.calculate_control(x, y, pose, rotation)  
                            print(angl_diff)
                        throttle, _, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error  = pid_speed21(road_error, angle_error, speed_error, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error)
                    else:
                        steering, throttle = 0, 0                                              
                    pid_actions = tf.constant([[steering, throttle]], dtype=tf.float32)

                    if model_drive:
                        actions = agent.action(image)
                    else:
                        actions = pid_actions

                    # save data for output
                    pos_list.append([obs["pos_x"], obs["pos_y"], obs["pos_z"]])
                    xte_list.append(obs["xte"])
                    speed_list.append(obs["speed"])
                    actions_list.append(actions)
                    pid_list.append(pid_actions)
                    if perturb:
                        perturbed_image_list.append(image)

                except KeyboardInterrupt:
                    self.logger.info(
                        f"{5 * '+'} SDSandBox Simulator Got Interrupted {5 * '+'}"
                    )
                    self.client.stop()
                    raise KeyboardInterrupt
            print("SDSandBox: Finished scenario")
            # break
            self.client.msg_handler.update([[0.0, 0.0]], None)
            # send reset to sim client
            self.client.msg_handler.reset_car()

            # return the resul of this simulation
            res = ScenarioOutcome(
                frames=[x for x in range(len(pos_list))],
                pos=pos_list,
                xte=xte_list,
                speeds=speed_list,
                actions=actions_list,
                pid_actions=pid_list,
                scenario=scenario,
                original_images=original_image_list,
                perturbed_images=perturbed_image_list,
                isSuccess=isSuccess,
                timeout=timeout,
            )
            del (
                pos_list,
                xte_list,
                speed_list,
                actions_list,
                obs,
                perturbed_image,
                actions,
            )
            return res
        except Exception as e:
            # close the simulator
            self.tear_down()
            del (
                pos_list,
                xte_list,
                speed_list,
                actions_list,
                obs,
                perturbed_image,
                actions,
            )
            # throw the exception
            raise e

    def tear_down(self):
        self.client.msg_handler.on_disconnect()
        self.process.quit()

    def _client_connected(self, client: SimClient) -> bool:
        """
        Retruns true if the client is still connected
        """
        return client.is_connected()

    def name(self) -> str:
        """
        Returns the name of the simulator
        """
        return "SDSandBoxSimulatorAdapter"
