# used modules from perturbation drive
from numpy import ndarray, uint8
import matplotlib.patches as patches
import tensorflow as tf
import cvxpy as cp
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
import matplotlib.pyplot as plt
from typing import Union
import cv2
import gym
import numpy as np
import time
import math

WAYPOINT_THRESHOLD = 5
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
            #print(f"Orientation: {round(angle_cur, 3)}, {round(math.radians(angle_cur), 3)} rad")
            distance = math.sqrt((x_target - x_cur)**2 + (y_target - y_cur)**2)
            #print(f"dist {distance}")
            _, angle_y_axis_degrees, _=self.angle_extraction(x_cur, 0.0, y_cur, x_target, 0.0, y_target)
            #print(f"Angle to goal: {angle_y_axis_degrees}")
            angle_difference=self.angle_difference(angle_cur,angle_y_axis_degrees)
            # print(f"angle diff {angle_difference}")
            steering = (math.radians(-angle_difference) / math.pi)*4
            # print(f"angle diff: {angle_difference}, steering: {steering}")
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



class UdacitySimulator(PerturbationSimulator):
    def __init__(
        self,
        simulator_exe_path: str = "./examples/udacity/udacity_utils/sim/udacity_sim.app",
        host: str = "127.0.0.1",
        port: int = 9091,
        show_image_cb=True
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
        self.show_image_cb=show_image_cb
            

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
        self, agent: Union[ADS,None], scenario: Scenario, perturbation_controller: Union[ImagePerturbation,None], perturb=False, model_drive=False, weather="Sun", intensity=90
    ) -> ScenarioOutcome:
        try:
            waypoints = scenario.waypoints

            
            perturbation_function_string = scenario.perturbation_function
            perturbation_scale = scenario.perturbation_scale
            monitor = ImageCallBack()
            monitor.display_waiting_screen()
            self.logger.info(f"{5 * '-'} Starting udacity scenario {5 * '_'}")

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
            # Road generatior ir none because we currently do not build random roads
            self.client.weather(weather,intensity)
            

            obs: ndarray[uint8] = self.client.reset(
                skip_generation=False, track_string=waypoints
            )
            
            
            obs, done, info = self.client.observe()
            start_time = time.time()
            target_speed=30.0
            # target_speed=25.0
            prev_throttle =  0.0
            prev_steering = 0.0
            waypoint_controller = Waypoint_control_utils(WAYPOINT_THRESHOLD, ANGLE_THRESHOLD)

            current_waypoint_index=0
            waypoint_list=waypoint_controller.convert_waypoints(waypoints)
            waypoint_list=waypoint_list
            # print(len(waypoint_list))
            
            


            counter=0
            # action loop
            once=True
            while not done:
                
                counter+=1
                if time.time() - start_time > 100:
                    self.logger.info("Udacity: Timeout after 100s")
                    timeout = True
                    break

                obs = cv2.resize(obs, (320, 240), cv2.INTER_NEAREST)
                original_image_list.append(obs)

                if perturb:
                    # perturb the image
                    perturbed_image = perturbation_controller.perturbation(
                        obs, perturbation_function_string, perturbation_scale
                    )
                    image=perturbed_image
                else:
                    image=obs
                
                road_error=float(info['cte_pid'])
                angle_error=float(info['angle'])
                speed_error=target_speed-float(info['speed'])

                

                # print(f"s:{info['speed']}")
                if PID:
                        throttle, steering, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error  = pid_speed20(road_error, angle_error, speed_error, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error)
                        if road_error==prev_road_error and prev_angle_error==angle_error and speed_error==prev_speed_error:
                            steering=prev_steering
                            throttle=prev_throttle
                else:
                        
                        
                        
                        rotation=info["orientation_euler"]
                        
                        if  current_waypoint_index < len(waypoint_list):
                            current_waypoint = waypoint_list[current_waypoint_index]
                        x, y = current_waypoint
                        steering, throttle, dist, angl_diff,angle = waypoint_controller.calculate_control(x, y, info["pos"], rotation)  
                        
                        if dist <= WAYPOINT_THRESHOLD:
                            current_waypoint_index += 4
                            if  current_waypoint_index < len(waypoint_list):
                                current_waypoint = waypoint_list[current_waypoint_index]
                            x, y = current_waypoint
                            steering,throttle , dist, angl_diff,angle = waypoint_controller.calculate_control(x, y, info["pos"], rotation)  
                       
                        throttle, _, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error  = pid_speed21(road_error, angle_error, speed_error, prev_road_error, prev_angle_error, prev_speed_error, total_road_error, total_angle_error, total_speed_error)
                
                        
                        # steering, _, dist, angl_diff = waypoint_controller.calculate_control(x, y, info["pos"], info["orientation"])  
                        # print(f'Going to: {current_waypoint[0]},{current_waypoint[1]} currently at {info["pos"][0]},{info["pos"][1]} distance: {dist} angle: {angl_diff}')
                        
                            
                        
                pid_actions = tf.constant([[steering, throttle]], dtype=tf.float32)

                
                prev_throttle=throttle
                prev_steering=steering
                
                


                # agent makes a move, the agent also selects the dtype and adds a batch dimension
                if model_drive:
                    actions = agent.action(image)
                else:
                    actions = pid_actions
                # print(actions)
                
                

                # clip action to avoid out of bound errors
                if isinstance(self.client.action_space, gym.spaces.Box):
                    actions = np.clip(
                        actions,
                        self.client.action_space.low,
                        self.client.action_space.high,
                    )
                if isinstance(self.client.action_space, gym.spaces.Box):
                    pid_actions = np.clip(
                        pid_actions,
                        self.client.action_space.low,
                        self.client.action_space.high,
                    )
                if self.show_image_cb:
                    monitor.display_img(
                    image,
                    f"{actions[0][0]}",
                    f"{actions[0][1]}",
                    perturbation_function_string,
                    )
                # obs is the image, info contains the road and the position of the car
                obs, done, info = self.client.step(actions)
                time.sleep(0.015)
                # print(actions)

                # log new info
                pos_list.append(info["pos"])
                xte_list.append(info["cte"])
                speed_list.append(info["speed"])
                actions_list.append(actions)
                pid_list.append(pid_actions)
                if perturb:
                    perturbed_image_list.append(image)

            # determine if we were successful
            # plt.close()
            isSuccess = max([abs(xte) for xte in xte_list]) < self.max_xte
            if timeout:
                isSuccess=False
            self.logger.info(
                f"{5 * '-'} Finished udacity scenario: {isSuccess} {5 * '_'}"
            )
            monitor.display_disconnect_screen()
            monitor.destroy()

            # reset for the new track
            _ = self.client.reset(skip_generation=False, track_string=waypoints)
            # return the scenario output
            return ScenarioOutcome(
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
