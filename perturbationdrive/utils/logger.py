import logging
import csv
from dataclasses import asdict
import json
import os
from typing import List

from ..Simulator.Scenario import ScenarioOutcome, OfflineScenarioOutcome


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

    def write(self, scenario_outcomes: List[ScenarioOutcome]):
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
                data.append(asdict(scenario_outcome))

            # Write updated data back to file
            with open(self.file_path, "w") as file:
                json.dump(data, file, indent=4)


class OfflineScenarioOutcomeWriter:
    def __init__(self, file_path: str, overwrite_logs: bool = True):
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
