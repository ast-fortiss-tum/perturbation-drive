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
        ads: ADS,
    ):
        assert isinstance(
            simulator, PerturbationSimulator
        ), "Simulator must be a subclass of PerturbationSimulator"
        assert isinstance(ads, ADS), "ADS must be a subclass of ADS"
        self.simulator = simulator
        self.ads = ads
        self.iamge_perturbation = ImagePerturbation()

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

        iamge_perturbation = ImagePerturbation(
            perturbation_functions, attention_map, image_size
        )

        results: List[OfflineScenarioOutcome] = []

        print(f"{5 * '-'} Starting offline benchmarking {5 * '-'}")

        for image_path in image_files:
            # find the json output
            frame_number = os.path.basename(image_path).split("_")[0]
            # check if this file exists and only continues if it does
            json_filename = os.path.join(
                os.path.dirname(image_path), "record_" + frame_number + ".json"
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
                image = cv2.imread(image_path)
            except:
                print(
                    f"{5 * '+'} Warning: Offline Perturbation: Could not read {image_path} {5 * '+'}"
                )
                continue

            # iterate over all perturbations
            for function_str in perturbation_functions:
                # iterate over all scales
                for intensity in range(5):
                    normal_image_actions = self.ads.action(image)

                    perturbed_image = iamge_perturbation.perturbation(
                        image, function_str, intensity
                    )
                    perturbed_image_actions = self.ads.action(perturbed_image)

                    # store the result
                    results.append(
                        OfflineScenarioOutcome(
                            image_file_name=image_path,
                            json_file_name=json_filename,
                            perturbation_function=function_str,
                            perturbation_scale=intensity,
                            ground_truth_actions=[steering, throttle],
                            perturbed_image_actions=perturbed_image_actions,
                            normal_image_actions=normal_image_actions,
                        )
                    )

        print(f"{5 * '-'} Finished offline benchmarking {5 * '-'}")

        # TODO: Show command line summary here

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
        log_dir: Union[str, None] = "logs.json",
        overwrite_logs: bool = True,
        image_size: Tuple[float, float] = (240, 320),
    ) -> Union[None, List[ScenarioOutcome]]:
        """
        Basically, what we hace done in image perturbations up until now but in a single nice function wrapped

        If log_dir is none, we return the scenario outcomes
        """
        image_perturbation = ImagePerturbation(
            funcs=perturbation_functions,
            attention_map=attention_map,
            image_size=image_size,
        )
        scale = 0
        index = 0
        outcomes: List[ScenarioOutcome] = []
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
        # wait 1 seconds for connection to build up
        time.sleep(1)

        # set up initial road
        waypoints = None
        if not road_generator is None:
            waypoints = road_generator.generate()

        # grid search loop
        while True:
            # check if we leave the loop, increment the index and scale
            index += 1
            if len(perturbations) == 1:
                # all perturbations resulted in failures
                # we will still have one perturbation here because we never
                # drop the empty perturbation
                break
            if index == len(perturbations):
                # we increment the scale, so start with the first perturbation again
                index = 0
                scale += 1
                # we also generate a new track here
                if not road_generator is None:
                    waypoints = road_generator.generate()

            if scale > 4:
                # we went through all scales
                break

            # get the road for the scenario

            # get the perturbation function for the scenario
            perturbation = perturbations[index]

            scenario = Scenario(
                waypoints=waypoints,
                perturbation_function=perturbation,
                perturbation_scale=scale,
            )

            # simulate the scenario
            outcome = self.simulator.simulate_scanario(
                self.ads, scenario=scenario, perturbation_controller=image_perturbation
            )
            outcomes.append(outcome)

            # check if we drop the scenario, we never remove the empty perturbation
            # for comparison reasons
            if not outcome.isSuccess and not perturbation == "":
                perturbations.remove(perturbation)

        # TODO: print command line summary of benchmarking process

        # tear down the simulator
        self.simulator.tear_down()
        if log_dir is None:
            return outcomes
        else:
            scenario_writer = ScenarioOutcomeWriter(log_dir, overwrite_logs)
            scenario_writer.write(outcomes)

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
        iamge_perturbation = ImagePerturbation()
