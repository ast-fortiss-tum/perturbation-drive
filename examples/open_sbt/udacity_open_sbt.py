# related to open_sbt
from evaluation.fitness import *
from evaluation.critical import *
from simulation.simulator import Simulator, SimulationOutput
from model_ga.individual import Individual

# imports
from typing import List
import traceback
import gc

# other example modules
from examples.models.dave2_agent import Dave2Agent

# Related to example
from examples.udacity.udacity_simulator import UdacitySimulator
from examples.open_sbt.utils_open_sbt import (
    mapOutComeToSimout,
    shortIndividualToScenario,
    individualsToName,
)

# related to perturbation drive
from perturbationdrive import (
    PerturbationDrive,
    ScenarioOutcome,
    Scenario,
    InformedRoadGenerator,
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
        ads = Dave2Agent()
        benchmarking_obj = PerturbationDrive(simulator, ads)
        # extract the amount of angles from the variable names
        road_generator = InformedRoadGenerator(num_control_nodes=8, max_angle=35)
        # we need to set the sim here up to get the starting position
        try:
            benchmarking_obj.simulator.connect()
            starting_pos = benchmarking_obj.simulator.initial_pos

            # create all scenarios
            scenarios: List[Scenario] = [
                shortIndividualToScenario(
                    individual=ind,
                    variable_names=variable_names,
                    road_generator=road_generator,
                    starting_pos=starting_pos,
                )
                for ind in list_individuals
            ]
            hased_name = individualsToName(
                individuals=list_individuals,
                variable_names=variable_names,
                sim_folder="udacity",
                prefix="dave2",
            )
            # run the individualts
            outcomes: List[ScenarioOutcome] = benchmarking_obj.simulate_scenarios(
                scenarios=scenarios,
                attention_map={},
                log_dir=f"./logs/open_sbt/udacity/dave2_res_{hased_name}.json",
                overwrite_logs=False,
                image_size=(240, 320),
            )
        except Exception as e:
            print(f"Exception {e} in OpenSBT Udacity Wrapper")
            traceback.print_exc()
        finally:
            benchmarking_obj.simulator.tear_down()
            del benchmarking_obj
            del simulator
            del ads
            gc.collect()

        # convert the outcomes to sbt format
        return [mapOutComeToSimout(outcome) for outcome in outcomes]
