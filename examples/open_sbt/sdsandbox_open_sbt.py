# related to open_sbt
from simulation.simulator import Simulator, SimulationOutput
from model_ga.individual import Individual

from typing import List, Dict, Any, Union

from examples.models.dave2_agent import Dave2Agent

from examples.open_sbt.utils_open_sbt import (
    calculate_velocities,
    shortIndividualToScenario,
    individualsToName,
)
from examples.self_driving_sandbox_donkey.sdsandbox_simulator import SDSandboxSimulator

# related to perturbation drive
from perturbationdrive import (
    PerturbationDrive,
    ScenarioOutcome,
    Scenario,
    InformedRoadGenerator,
)


class SDSandBox_OpenSBTWrapper(Simulator):
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
        Runs all indicidual simulations and returns simulation outputs for each individual
        """
        # set up all perturbation drive objects
        simulator = SDSandboxSimulator(
            simulator_exe_path="./examples/self_driving_sandbox_donkey/sim/donkey-sim.app",
            host="127.0.0.1",
            port=9091,
        )
        ads = Dave2Agent()
        benchmarking_obj = PerturbationDrive(simulator, ads)

        # extract the amount of angles from the variable names
        road_generator = InformedRoadGenerator(num_control_nodes=8, max_angle=35)

        # we need to set the sim here up to get the starting position
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
            individuals=list_individuals, variable_names=variable_names, sim_folder="sdsandbox", prefix="dave2"
        )

        # run the individuals
        outcomes: List[ScenarioOutcome] = benchmarking_obj.simulate_scenarios(
            scenarios=scenarios,
            attention_map={},
            log_dir=f"./examples/open_sbt/sdsandbox/dave2_res_{hased_name}.json",
            overwrite_logs=True,
            image_size=(240, 320),
        )
        benchmarking_obj.simulator.tear_down()

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
