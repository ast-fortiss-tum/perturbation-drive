from .Simulator.Simulator import PerturbationSimulator
from .AutomatedDrivingSystem.ADS import ADS
from .imageperturbations import ImagePerturbation, get_functions_from_module
from .Simulator.Scenario import Scenario, ScenarioOutcome, OfflineScenarioOutcome
from .RoadGenerator.RoadGenerator import RoadGenerator
from .utils.logger import ScenarioOutcomeWriter, OfflineScenarioOutcomeWriter

from typing import List, Union, Dict, Tuple
import copy
import os
import json
import cv2
import time


class PerturbationDrive:
    """
    Simulator independent ADS robustness benchmarking
    """

    def __init__(
        self,
        simulator: PerturbationSimulator,
        ads: Union[ADS, None],
    ):
        assert isinstance(
            simulator, PerturbationSimulator
        ), "Simulator must be a subclass of PerturbationSimulator"
        if ads is not None:
            assert isinstance(ads, ADS), "ADS must be a subclass of ADS"
        self.simulator = simulator
        self.ads = ads

    def setADS(self, ads: ADS):
        assert isinstance(ads, ADS), "ADS must be a subclass of ADS"
        self.ads = ads

    def setSimulator(self, simulator: PerturbationSimulator):
        assert isinstance(
            simulator, PerturbationSimulator
        ), "Simulator must be a subclass of PerturbationSimulator"
        self.simulator = simulator

    def offline_perturbation(
        self,
        dataset_path: str,
        perturbation_functions: List[str],
        attention_map: Dict = {},
        log_dir: Union[str, None] = "logs.json",
        overwrite_logs: bool = True,
        image_size: Tuple[float, float] = (240, 320),
    ) -> Union[None, List[OfflineScenarioOutcome]]:
        """
        Take a data set, and for each image perturb it based on the given perturbation function and calculate
        an error rate.

        The dataset needs to contain the frames and a json file for each frame:
        - The images name needs to be in the format of "`frame number`_`free_string`.{jpg | jpeg | png}" where `free_string` can be any string with length of more than 1
        - The json file name needs to be in the format of "record_`frame number`.json" where the `frame number` needs to correlate to the image
        - The json file needs to contain the ground truth values for steering angle and throttle in the files `user/angle` and `user/throttle`

        If log_dir is none, we return the scenario outcomes
        """
        assert os.path.isdir(dataset_path), f"{dataset_path} is not a directory"
        # we need files to iterate over all images
        files = os.listdir(dataset_path)

        has_json = any(file.endswith(".json") for file in files)
        assert has_json, "No .json file found in the directory"
        has_image = any(file.endswith((".jpg", ".jpeg", ".png")) for file in files)
        assert has_image, "No image file (.jpg, .jpeg, .png) found in the directory"
        image_files = [
            file for file in files if file.endswith((".jpg", ".jpeg", ".png"))
        ]
        perturbations: List[str] = copy.deepcopy(perturbation_functions)
        # populate all perturbations
        if len(perturbations) == 0:
            perturbation_fns = get_functions_from_module(
                "perturbationdrive.perturbationfuncs"
            )
            perturbations = list(map(lambda f: f.__name__, perturbation_fns))
        iamge_perturbation = ImagePerturbation(perturbations, attention_map, image_size)

        results: List[OfflineScenarioOutcome] = []

        print(f"{5 * '-'} Starting offline benchmarking {5 * '-'}")
        for image_path in image_files:
            # find the json output
            frame_number = os.path.basename(image_path).split("_")[0]
            # check if this file exists and only continues if it does
            json_filename = os.path.join(
                dataset_path, "record_" + frame_number + ".json"
            )
            if not os.path.exists(json_filename):
                print(
                    f"{5 * '+'} Warning: Offline Perturbation: The json {json_filename} for framn {image_path} does not exist {5 * '+'}"
                )
                continue
            with open(json_filename, "rt") as fp:
                data = json.load(fp)
            steering = float(data["user/angle"])
            throttle = float(data["user/throttle"])

            try:
                image_full_path = os.path.join(dataset_path, image_path)
                image = cv2.imread(image_full_path)
            except:
                print(
                    f"{5 * '+'} Warning: Offline Perturbation: Could not read {image_full_path} {5 * '+'}"
                )
                continue

            # iterate over all perturbations
            for function_str in perturbations:
                # iterate over all scales
                for intensity in range(5):
                    normal_image_actions = self.ads.action(image)

                    perturbed_image = iamge_perturbation.perturbation(
                        image, function_str, intensity
                    )
                    perturbed_image_actions = self.ads.action(perturbed_image)
                    steering = f"{perturbed_image_actions[0][0]}"
                    throttle = f"{perturbed_image_actions[0][1]}"

                    # store the result
                    results.append(
                        OfflineScenarioOutcome(
                            image_file_name=image_path,
                            json_file_name=json_filename,
                            perturbation_function=function_str,
                            perturbation_scale=intensity,
                            ground_truth_actions=[steering, throttle],
                            perturbed_image_actions=[
                                steering,
                                throttle,
                            ],
                            normal_image_actions=[
                                f"{normal_image_actions[0][0]}",
                                f"{normal_image_actions[0][1]}",
                            ],
                        )
                    )

        print(f"{5 * '-'} Finished offline benchmarking {5 * '-'}")

        # TODO: Show command line summary here
        del iamge_perturbation
        del image
        del image_files
        del files
        del perturbations

        if log_dir is None:
            return results
        else:
            scenario_writer = OfflineScenarioOutcomeWriter(log_dir, overwrite_logs)
            scenario_writer.write(results)

    def grid_seach(
        self,
        perturbation_functions: List[str],
        attention_map: Dict = {},
        road_generator: Union[RoadGenerator, None] = None,
        road_angles: List[int] = None,
        road_segments: List[int] = None,
        log_dir: Union[str, None] = "logs.json",
        overwrite_logs: bool = True,
        image_size: Tuple[float, float] = (240, 320),
        test_model: bool=False,
        perturb: bool=False,
        weather: Union[str, None] = "Sun",
        weather_intensity: Union[int, None] = 90,
        collect_train_data=False,
    ) -> Union[None, List[ScenarioOutcome]]:
        """
        Basically, what we hace done in image perturbations up until now but in a single nice function wrapped

        If log_dir is none, we return the scenario outcomes
        """
        if perturb:
            image_perturbation = ImagePerturbation(
                funcs=perturbation_functions,
                attention_map=attention_map,
                image_size=image_size,
            )
            
        else:
            image_perturbation=None

        scale = 0
        index = 0
        # outcomes: List[ScenarioOutcome] = []
        if log_dir is None:
            print("No log directory")
        else:
            scenario_writer = ScenarioOutcomeWriter(log_dir, overwrite_logs)
        
        perturbations: List[str] = []
        
        if perturb:
            perturbations: List[str] = copy.deepcopy(perturbation_functions)
            # populate all perturbations
            if len(perturbations) == 0:
                perturbation_fns = get_functions_from_module(
                    "perturbationdrive.perturbationfuncs"
                )
                perturbations = list(map(lambda f: f.__name__, perturbation_fns))
        
        # we append the empty perturbation here
        perturbations.append("")
        

        # set up simulator
        self.simulator.connect()
        # wait 1 second for connection to build up
        time.sleep(1)

        # set up initial road
        waypoints = None
        if not road_generator is None:
            # TODO: Insert here all kwargs needed for specific generator
            waypoints = road_generator.generate(starting_pos=self.simulator.initial_pos,angles=road_angles,seg_lengths=road_segments)

        # grid search loop
        while True:
            print(perturbations)
            perturbation = perturbations[index]
            print(
                f"{5 * '-'} Running Scenario: Perturbation {perturbation} on {scale} {5 * '-'}"
            )
            
            scenario = Scenario(
                waypoints=waypoints,
                perturbation_function=perturbation,
                perturbation_scale=scale,
            )

            # simulate the scenario
            outcome = self.simulator.simulate_scanario(
                self.ads, scenario=scenario, perturbation_controller=image_perturbation, perturb=perturb, model_drive=test_model, weather=weather, intensity=weather_intensity
            )
            

            # check if we drop the scenario, we never remove the empty perturbation
            # for comparison reasons
            if not outcome.isSuccess or perturb==False:
                perturbations.remove(perturbation)
            else:
                index += 1
            scenario_writer.write([outcome],images=collect_train_data)
            
            if len(perturbations) == 0:
                # all perturbations resulted in failures
                # we will still have one perturbation here because we never
                # drop the empty perturbation
                break
            if index == len(perturbations):
                # we increment the scale, so start with the first perturbation again
                index = 0
                scale += 1

            if scale > 4:
                # we went through all scales
                break
            

        # TODO: print command line summary of benchmarking process
        del image_perturbation
        del scenario
        del road_generator

        # tear down the simulator
        self.simulator.tear_down()

    def simulate_scenarios(
        self,
        scenarios: List[Scenario],
        attention_map: Dict = {},
        log_dir: Union[str, None] = "logs.json",
        overwrite_logs: bool = True,
        image_size: Tuple[float, float] = (240, 320),
    ) -> Union[None, List[ScenarioOutcome]]:
        """
        Basically, what we have done with open sbt

        If log_dir is none, we return the scenario outcomes
        """
        print(attention_map)

        # get all perturbations to set up this object
        perturbations: List[str] = []
        for scenario in scenarios:
            perturbations.append(scenario.perturbation_function)

        image_perturbation = ImagePerturbation(
            funcs=perturbations, attention_map=attention_map, image_size=image_size
        )
        # sim is setup in main to get starting pos

        outcomes: List[ScenarioOutcome] = []
        # iterate over all scenarios
        for scenario in scenarios:
            outcome = self.simulator.simulate_scanario(
                self.ads, scenario=scenario, perturbation_controller=image_perturbation, perturb=True, model_drive=True
            )
            outcomes.append(outcome)
            time.sleep(2.0)

        del image_perturbation
        del perturbations
        # tear sim down
        if log_dir is not None:
            scenario_writer = ScenarioOutcomeWriter(log_dir, overwrite_logs)
            scenario_writer.write(outcomes)
            del scenario_writer
        return outcomes
