# related to open_sbt
from evaluation.fitness import *
from evaluation.critical import *
from simulation.simulator import Simulator, SimulationOutput
from model_ga.individual import Individual

# imports
from typing import List

# other example modules
from examples.models.example_agent import ExampleAgent

# Related to example
from examples.udacity.udacity_simulator import UdacitySimulator

from examples.open_sbt.utils_open_sbt import individualToScenario, calculate_velocities

# related to perturbation drive
from perturbationdrive import (
    PerturbationDrive,
    ScenarioOutcome,
    Scenario,
    CustomRoadGenerator,
)


class Udacity_OpenSBTWrapper(Simulator):
    @staticmethod
    def simulate(
        list_individuals: List[Individual],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float,
        do_visualize: bool = False,
    ) -> List[SimulationOutput]:
        """
        Runs all indicidual simulations and returns simulation outputs
        """
        # set up all perturbation drive objects
        simulator = UdacitySimulator(
            simulator_exe_path="./examples/udacity/udacity_utils/sim/udacity_sim.app",
            host="127.0.0.1",
            port=9091,
        )
        ads = ExampleAgent()
        benchmarking_obj = PerturbationDrive(simulator, ads)
        road_generator = CustomRoadGenerator()

        # we need to set the sim here up to get the starting position
        benchmarking_obj.simulator.connect()
        starting_pos = benchmarking_obj.simulator.initial_pos

        # create all scenarios
        scenarios: List[Scenario] = [
            individualToScenario(
                individual=ind,
                variable_names=variable_names,
                road_generator=road_generator,
                starting_pos=starting_pos,
            )
            for ind in list_individuals
        ]

        # run the individualts
        outcomes: List[ScenarioOutcome] = benchmarking_obj.simulate_scenarios(
            scenarios=scenarios,
            attention_map={},
            log_dir=None,
            overwrite_logs=False,
            image_size=(240, 320),
        )

        # convert the outcomes to sbt format
        return [
            SimulationOutput(
                simTime=float(len(outcome.frames)),
                times=outcome.frames,
                location={"ego": [(x[0], x[1]) for x in outcome.pos]},
                velocity={"ego": calculate_velocities(outcome.pos, outcome.speeds)},
                speed={"ego": outcome.speeds},
                acceleration={"ego": []},
                yaw={
                    "ego": [],
                },
                collisions=[],
                actors={
                    1: "ego",
                },
                otherParams={"xte": outcome.xte},
            )
            for outcome in outcomes
        ]
