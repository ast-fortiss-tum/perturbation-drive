import logging
import csv
from dataclasses import asdict
import json
import os
from typing import List
import sys
import numpy as np

from ..Simulator.Scenario import ScenarioOutcome, OfflineScenarioOutcome
from .custom_types import LOGGING_LEVEL
from PIL import Image
import gc


class CSVLogHandler(logging.FileHandler):
    """
    Util class to log perturbation output and metrics

    :param: filename="logs.csv": String name of log file
    :param: mode="w": Mode of the logger. Here we can use options such as "w", "a", ...
    :param: encoding=None: Encoding of the file
    """

    def __init__(self, filename="logs.csv", mode="w", encoding=None):
        super().__init__(filename, mode, encoding, delay=False)
        self.writer = csv.writer(
            self.stream, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        self.current_row = []

    def emit(self, record):
        if isinstance(record.msg, (list, tuple)):
            self.current_row.extend(record.msg)
        else:
            self.current_row.append(record.msg)

    def flush_row(self):
        if self.current_row:
            self.writer.writerow(self.current_row)
            self.flush()
            self.current_row = []


class ScenarioOutcomeWriter:
    def __init__(self, file_path: str, overwrite_logs: bool = True):
        """
        Write scenario outcomes to a json file

        :param file_path: path to the json file
        :param overwrite_logs: whether to overwrite the existing logs
        """
        self._write = True
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(
                f"{ 5* '+'} Warning Scenario Writer: The log file path already exists {5 * '+'}"
            )
            if not overwrite_logs:
                print(
                    f"{5 * '+'} Warning Scenario Writer: No logs will be saved, due to overwrite_logs=false {5 * '+'}"
                )
                self._write = False
        self.file_path = file_path

    def write(self, scenario_outcomes: List[ScenarioOutcome],images=False):
        """
        Write scenario outcomes to a json file

        :param scenario_outcomes: list of scenario outcomes
        """
        if len(scenario_outcomes) == 0:
            print(f"{ 5* '+'} Error Scenario Writer: The scenario is empty {5 * '+'}")
            return
        if self._write:
            # Read existing data
            if os.path.exists(self.file_path):
                with open(self.file_path, "r") as file:
                    try:
                        data = json.load(file)
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []
            # Append new data
            for scenario_outcome in scenario_outcomes:
                image_folder_name=self.file_path.split("logs_")[0]+"_"+str(scenario_outcome.scenario.perturbation_function)+"_"+str(scenario_outcome.scenario.perturbation_scale)
                scenario_data=asdict(scenario_outcome)
                perturbed_images = scenario_data.pop('perturbed_images', None)
                original_images = scenario_data.pop('original_images', None)
                if images:
                    image_frames=scenario_outcome.frames
                    if len(perturbed_images)>0:
                        if not os.path.exists(image_folder_name+"_perturbed"):
                            os.makedirs(image_folder_name+"_perturbed")
                        for i, img_array in enumerate(perturbed_images):
                            image_data_int = np.rint(img_array).astype(np.uint8)
                            # print(image_data_int)
                            img = Image.fromarray(image_data_int)
                            frame=image_frames[i]
                            img.save(os.path.join(image_folder_name+f"_perturbed/{frame}.jpg"))
                    else:
                        if not os.path.exists(image_folder_name+"_original"):
                            os.makedirs(image_folder_name+"_original")
                        for i, img_array in enumerate(original_images):
                            img = Image.fromarray(img_array)
                            frame=image_frames[i]
                            img.save(os.path.join(image_folder_name+f"_original/{frame}.jpg"))
                # print(perturbed_images)
                data.append(scenario_data)
                gc.collect()
            # Write updated data back to file
            # print(data)
            with open(self.file_path, "w") as file:
                json.dump(data, file, cls=NumpyEncoder, indent=4)


class OfflineScenarioOutcomeWriter:
    def __init__(self, file_path: str, overwrite_logs: bool = True):
        """
        Write offline scenario outcomes to a json file

        :param file_path: path to the json file
        :param overwrite_logs: whether to overwrite the existing logs
        """
        self._write = True
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(
                f"{ 5* '+'} Warning Offline Scenario Writer: The log file path already exists {5 * '+'}"
            )
            if not overwrite_logs:
                print(
                    f"{5 * '+'} Warning Offline Scenario Writer: No logs will be saved, due to overwrite_logs=false {5 * '+'}"
                )
                self._write = False
        self.file_path = file_path

    def write(self, scenario_outcomes: List[OfflineScenarioOutcome]):
        """
        Write offline scenario outcomes to a json file

        :param scenario_outcomes: list of offline scenario outcomes
        """
        if len(scenario_outcomes) == 0:
            print(
                f"{ 5* '+'} Error Offline Scenario Writer: The scenario is empty {5 * '+'}"
            )
            return
        if self._write:
            # Read existing data
            if os.path.exists(self.file_path):
                with open(self.file_path, "r") as file:
                    try:
                        data = json.load(file)
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []

            # Append new data
            for scenario_outcome in scenario_outcomes:
                data.append(asdict(scenario_outcome))

            # Write updated data back to file
            with open(self.file_path, "w") as file:
                json.dump(data, file, indent=4)


class GlobalLog:
    """This class is used to log acress different modeles in the project"""

    def __init__(self, logger_prefix: str):
        """
        We use the logger_prefix to distinguish between different loggers

        :param logger_prefix: prefix of the logger
        """
        self.logger = logging.getLogger(logger_prefix)
        # avoid creating another logger if it already exists
        if len(self.logger.handlers) == 0:
            self.logger = logging.getLogger(logger_prefix)
            self.logger.setLevel(level=LOGGING_LEVEL)

            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(level=logging.DEBUG)
            self.logger.addHandler(ch)

    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)

    def info(self, message):
        """Log info message"""
        self.logger.info(message)

    def warn(self, message):
        """Log warning message"""
        self.logger.warn(message)

    def error(self, message):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
