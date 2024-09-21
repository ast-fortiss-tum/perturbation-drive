from enum import Enum
from .Simulator.Simulator import PerturbationSimulator
from .AutomatedDrivingSystem.ADS import ADS
from .imageperturbations import ImagePerturbation, get_functions_from_module
from .Simulator.Scenario import Scenario, ScenarioOutcome, OfflineScenarioOutcome
from .RoadGenerator.RoadGenerator import RoadGenerator
from .utils.logger import ScenarioOutcomeWriter, OfflineScenarioOutcomeWriter

from typing import List, Union, Dict, Tuple, Type, Callable
import copy
import os
import json
import cv2
import time
from collections import defaultdict
from dataclasses import dataclass, field


class RoadGenerationFrequency(Enum):
    """
    Enum to control the frequency of road generation within the grid search

    - NEVER: Never generate a road
    - ONCE: Generate a road once before the grid search
    - AFTER_EARCH_INTENSITY_ITERATION: Generate a road after each intensity iteration
    - ALWAYS: Generate a road after each perturbation function
    """

    NEVER = 0
    ONCE = 1
    AFTER_EARCH_INTENSITY_ITERATION = 2
    ALWAYS = 3


@dataclass
class GridSearchConfig:
    """
    Configuration to control the grid search method

    :param perturbation_functions: list of perturbation functions to be used
    :type perturbation_functions: List[str]
    :param attention_map: attention map to be used. If None, no attention map is used
    :type attention_map: Dict = {}
    :param road_generator: road generator to be used. If None, no road is generated
    :type road_generator: Union[RoadGenerator, None] = None
    :param road_angles: angles for the road generator
    :type road_angles: List[int] = None
    :param road_segments: segments for the road generator
    :type road_segments: List[int] = None
    :param road_generation_frequency: frequency of road generation
    :type road_generation_frequency: Type[RoadGenerationFrequency] = RoadGenerationFrequency.ONCE
    :param log_dir: directory to save the logs. If None, no logs are saved
    :type log_dir: Union[str, None] = "logs.json"
    :param overwrite_logs: whether to overwrite existing logs
    :type overwrite_logs: bool = True
    :param image_size: size of the image returned by the simulator
    :type image_size: Tuple[float, float] = (240, 320)
    :param perturbation_class: class to be used for perturbation. Default is ImagePerturbation but can be any subclass of ImagePerturbation
    :type perturbation_class: Type[ImagePerturbation] = ImagePerturbation
    :param drop_perturbation: lambda function to evaluate whether to drop the perturbation after simulating a scenario. Default is to drop the perturbation if the scenario is not successful or if it times out. Receives the ScenarioOutcome of the simulation and returns a boolean
    :type drop_perturbation: Callable[[ScenarioOutcome], bool] = lambda outcome: (not outcome.isSuccess) or outcome.timeout
    :param evaluate_scenario_outcomes: lambda function to evaluate whether to increment the perturbation scale after all perturbations have been applied on a scale level. Default is to always increment the scale. Receives the list of all prior ScenarioOutcome and returns a boolean
    :type evaluate_scenario_outcomes: Callable[[List[ScenarioOutcome]], bool] = lambda outcomes: True
    """

    perturbation_functions: List[str]
    attention_map: Dict = field(default_factory=dict)
    road_generator: Union[RoadGenerator, None] = None
    road_angles: List[int] = field(default_factory=list)
    road_segments: List[int] = field(default_factory=list)
    road_generation_frequency: Type[RoadGenerationFrequency] = (
        RoadGenerationFrequency.ONCE
    )
    log_dir: Union[str, None] = "logs.json"
    overwrite_logs: bool = True
    image_size: Tuple[float, float] = (240, 320)
    perturbation_class: Type[ImagePerturbation] = ImagePerturbation
    drop_perturbation: Callable[[ScenarioOutcome], bool] = (
        lambda outcome: (not outcome.isSuccess) or outcome.timeout
    )
    increment_perturbation_scale: Callable[[List[ScenarioOutcome]], bool] = (
        lambda outcomes: True
    )

    def __post_init__(self):
        assert len(self.road_angles) == len(
            self.road_segments
        ), "Road angles and segments must have the same length"
        # assert that the perturbation class is a subclass of ImagePerturbation
        assert issubclass(
            self.perturbation_class, ImagePerturbation
        ), "Perturbation class must be a subclass of ImagePerturbation"
        # assert that perturbation functions is not empty
        assert len(self.perturbation_functions) > 0, "Perturbation functions is empty"
        # populate all perturbations
        if len(self.perturbation_functions) == 0:
            perturbation_fns = get_functions_from_module(
                "perturbationdrive.perturbationfuncs"
            )
            self.perturbation_functions = list(
                map(lambda f: f.__name__, perturbation_fns)
            )
        # add the empty perturbation if it is not in the list
        if "" not in self.perturbation_functions:
            self.perturbation_functions.append("")


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
        config: GridSearchConfig,
    ) -> Union[None, List[ScenarioOutcome]]:
        """
        Run a grid search over all perturbation functions given in the config and finds the maximum perturbation scale.


        The road generator receives the following keyword parameters:
        - starting_pos: Tuple[float, float, float, float] the starting position of the road
        - angles: List[int] the angles for the road
        - seg_lengths: List[int] the segment lengths for the road
        - prior_results: List[ScenarioOutcome] the prior results of the grid search


        :param config: configuration for the grid search
        :type config: GridSearchConfig
        :return: list of scenario outcomes if log_dir is None, else None
        :return type: Union[None, List[ScenarioOutcome]]
        """
        image_perturbation = config.perturbation_class(
            funcs=config.perturbation_functions,
            attention_map=config.attention_map,
            image_size=config.image_size,
        )
        scale = 0
        index = 0
        outcomes: List[ScenarioOutcome] = []
        perturbations: List[str] = copy.deepcopy(config.perturbation_functions)
        # set up simulator
        self.simulator.connect()
        # wait 1 seconds for connection to become stable
        time.sleep(1)

        # set up initial road
        waypoints = None
        if (
            not config.road_generator is None
            and config.road_generation_frequency != RoadGenerationFrequency.NEVER
        ):
            waypoints = config.road_generator.generate(
                starting_pos=self.simulator.initial_pos,
                angles=config.road_angles,
                seg_lengths=config.road_segments,
                prior_results=[],
            )

        # grid search loop
        while True:
            # get the perturbation function for the scenario
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
                self.ads, scenario=scenario, perturbation_controller=image_perturbation
            )
            print(
                f"{5 * '-'} Finished Scenario: Perturbation {perturbation} on {scale} {5 * '-'}"
            )
            print(
                f"{5 * '-'} Outcome: Success: {outcome.isSuccess}, Timeout: {outcome.timeout} {5 * '-'}"
            )
            outcomes.append(outcome)

            # let the callback decide if we drop the perturbation
            if config.drop_perturbation(outcome):
                perturbations.remove(perturbation)
                print(f"{5 * '-'} Removed Perturbation: {perturbation} {5 * '-'}")
            else:
                index += 1
            # check if we leave the loop, increment the index and scale
            if len(perturbations) == 0:
                # all perturbations resulted in failures
                break
            if index == len(perturbations):
                # we increment the scale, so start with the first perturbation again
                index = 0
                if config.increment_perturbation_scale(outcomes):
                    scale += 1
                    print(
                        f"{5 * '-'} Incremented Perturbation Scale to {scale} {5 * '-'}"
                    )
                if (
                    config.road_generation_frequency
                    == RoadGenerationFrequency.AFTER_EARCH_INTENSITY_ITERATION
                ):
                    waypoints = config.road_generator.generate(
                        starting_pos=self.simulator.initial_pos,
                        angles=config.road_angles,
                        seg_lengths=config.road_segments,
                        prior_results=outcomes,
                    )

            if scale > 4:
                # we went through all scales
                break
            if config.road_generation_frequency == RoadGenerationFrequency.ALWAYS:
                waypoints = config.road_generator.generate(
                    starting_pos=self.simulator.initial_pos,
                    angles=config.road_angles,
                    seg_lengths=config.road_segments,
                    prior_results=outcomes,
                )

        self._print_summary(outcomes)

        del image_perturbation
        del scenario
        del config.road_generator

        # tear down the simulator
        self.simulator.tear_down()
        if config.log_dir is None:
            return outcomes
        else:
            scenario_writer = ScenarioOutcomeWriter(
                config.log_dir, config.overwrite_logs
            )
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
                self.ads, scenario=scenario, perturbation_controller=image_perturbation
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

    def _print_summary(self, scenario_outcomes: List[ScenarioOutcome]):
        """
        Print a summary of the outcomes
        """
        # Dictionary to store data in the form {(perturbation_function, perturbation_scale): (success_count, timeout_count)}
        table = defaultdict(lambda: (0, 0))

        # Populate the table with success and timeout counts
        for outcome in scenario_outcomes:
            if outcome.scenario:  # Ensure scenario is not None
                key = (
                    outcome.scenario.perturbation_function,
                    outcome.scenario.perturbation_scale,
                )
                success, timeout = table[key]

                if outcome.isSuccess:
                    success += 1
                if outcome.timeout:
                    timeout += 1

                table[key] = (success, timeout)
        # Find all unique perturbations and scales for formatting the table
        perturbations = sorted(set(perturb for perturb, _ in table.keys()))
        scales = sorted(set(scale for _, scale in table.keys()))

        # Print the table
        print(f"{'Perturbation':<20}", end="")
        for scale in scales:
            print(f"{scale:>10}", end="")
        print()

        for perturbation in perturbations:
            print(f"{perturbation:<20}", end="")
            for scale in scales:
                success, timeout = table.get((perturbation, scale), (0, 0))
                print(f"{success}/{timeout:>10}", end="")
            print()
