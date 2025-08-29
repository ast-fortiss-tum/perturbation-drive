# examples/carla/example_ads.py
import numpy as np
from perturbationdrive.AutomatedDrivingSystem.ADS import ADS

class ExampleADS(ADS):
    def reset(self):
        pass

    def predict(self, image: np.ndarray):
        # naive center-line keeper: keep straight, gentle throttle
        steer = 0.0
        throttle = 0.25
        brake = 0.0
        return steer, throttle, brake