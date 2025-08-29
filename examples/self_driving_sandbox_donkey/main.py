# related to open_sbt
from problem.adas_problem import ADASProblem
from evaluation.fitness import *
from evaluation.critical import *
from algorithm.nsga2_optimizer import NsgaIIOptimizer
from experiment.search_configuration import DefaultSearchConfiguration

from utils import log_utils
import argparse
from typing import List, Dict, Any, Union
import traceback

from sdsandbox_simulator import SDSandboxSimulator
from examples.models.example_agent import ExampleAgent
from examples.open_sbt.criticality import FitnessFunction, Criticality
from examples.open_sbt.sdsandbox_open_sbt import SDSandBox_OpenSBTWrapper

# related to perturbation drive
from perturbationdrive import (
    PerturbationDrive,
    RandomRoadGenerator,
)


def open_sbt():
    # Define search problem
    problem = ADASProblem(
        problem_name="UdacityRoadGenerationProblem",
        scenario_path="",
        xl=[-20, -20, -20, -20, -20, -20, -20, -20, 0, 1],
        xu=[20, 20, 20, 20, 20, 20, 20, 20, 4, 6],
        simulation_variables=[
            "angle1",
            "angle2",
            "angle3",
            "angle4",
            "angle5",
            "angle6",
            "angle7",
            "angle8",
            "perturbation_scale",
            "perturbation_function",
        ],
        fitness_function=FitnessFunction(max_xte=4.0),
        critical_function=Criticality(),
        simulate_function=SDSandBox_OpenSBTWrapper.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )

    log_utils.setup_logging("./log.txt")

    # Set search configuration
    config = DefaultSearchConfiguration()
    config.n_generations = 10
    config.population_size = 2

    # Instantiate search algorithm
    optimizer = NsgaIIOptimizer(problem=problem, config=config)

    # Run search
    res = optimizer.run()

    # Write results. This currently does not work
    # res.write_results()


def go(
    host: str,
    port: int,
    pert_funcs: List[str] = [],
    attention: Dict[str, Any] = {},
    simulator_exe_path: str = "./sim/donkey-sim.app",
):
    try:
        simulator = SDSandboxSimulator(
            simulator_exe_path=simulator_exe_path, host=host, port=port
        )
        ads = ExampleAgent()
        road_generator = RandomRoadGenerator(num_control_nodes=8)
        benchmarking_obj = PerturbationDrive(simulator, ads)

        # start the benchmarking
        benchmarking_obj.grid_seach(
            perturbation_functions=pert_funcs,
            attention_map=attention,
            road_generator=road_generator,
            log_dir="./examples/self_driving_sandbox_donkey/logs.json",
            overwrite_logs=True,
            image_size=(240, 320),  # images are resized to these values
        )
        print(f"{5 * '#'} Finished Running SDSandBox Sim {5 * '#'}")
    except Exception as e:
        print(
            f"{5 * '#'} SDSandBox Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDSandBox Example")

    parser.add_argument("--host", type=str, default="127.0.0.1", help="server sim host")
    parser.add_argument("--port", type=int, default=9091, help="bind to port")
    parser.add_argument(
        "--perturbation",
        dest="perturbation",
        action="append",
        type=str,
        default=[],
        help="perturbations to use on the model. by default all are used",
    )
    parser.add_argument(
        "--sim_exe",
        type=str,
        default="./examples/self_driving_sandbox_donkey/sim/donkey-sim.app",
        help="sim executable path",
    )    
    parser.add_argument(
        "--attention_map", type=str, default="", help="which attention map to use"
    )
    parser.add_argument(
        "--attention_threshold",
        type=float,
        default=0.5,
        help="threshold for attention map perturbation",
    )
    parser.add_argument(
        "--attention_layer",
        type=str,
        default="conv2d_5",
        help="layer for attention map perturbation",
    )

    args = parser.parse_args()
    attention = (
        {}
        if args.attention_map == ""
        else {
            "map": args.attention_map,
            "threshold": args.attention_threshold,
            "layer": args.attention_layer,
        }
    )

    print(f"{5 * '#'} Started Running SDSandBox Sim {5 * '#'}")
    #go(
    #   host=args.host,
    #   port=args.port,
    #   pert_funcs=args.perturbation,
    #   attention=attention,
    #  simulator_exe_path=args.sim_exe,
    #)
    open_sbt()
