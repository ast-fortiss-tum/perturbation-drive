# Open-SBT Integration

This folder provides all python files to integrate [Open SBT](https://git.fortiss.org/opensbt) with this project.

## SDSandBox Wrapper

The file `sdsandbox_open_sbt` provides the implementation of the `Simulator` interface by implementing the following method. The experiment searches for scenarios consisting of roads and perturbation functions.
The road is defined by the angles between adjacent waypoints. See the `CustomRoadGenerator` in `perturbationdrive` for guidance on this road generation and definition.

- `simulate`: Simulates a list of individuals by converting the individuals to scenarios, running the scenarios in perturbation drive and then returning a list of SimulationOutputs.

## Udacity Wrapper

The file `udacity_open_sbt` provides the implementation of the `Simulator` interface by implementing the following mehotd. The experiment searches for scenarios consisting of roads and perturbation functions.
The road is defined by the angles between adjacent waypoints. See the `CustomRoadGenerator` in `perturbationdrive` for guidance on this road generation and definition.

- `simulate`: Simulates a list of individuals by converting the individuals to scenarios, running the scenarios in perturbation drive and then returning a list of SimulationOutputs.

```Python
def simulate(
    list_individuals: List[Individual],
    variable_names: List[str],
    scenario_path: str,
    sim_time: float,
    time_step: float,
    do_visualize: bool = False,
) -> List[SimulationOutput]:
    """
    Runs all individual simulations and returns simulation outputs for each individual.

    Args:
        list_individuals (List[Individual]): The list of individuals to simulate.
        variable_names (List[str]): The list of variable names.
        scenario_path (str): The path to the scenario.
        sim_time (float): The simulation time.
        time_step (float): The time step.
        do_visualize (bool, optional): Whether to visualize the simulation. Defaults to False.

    Returns:
        List[SimulationOutput]: The list of simulation outputs for each individual.
    """
```

## Criticality and Fitness

Provides criticallity and fitness functions for the Open-SBT integration. These functions are used to select new individuals which result in failures or near failure for the ADS.

The criticality function returns a boolean values indicating if the ADS failed the scenario. Here, a individual is considered as failed, if the vehicle crosses the border of the road.

The fitness function returns a tuple of values which should either be maximized or minimized to find new scenarios. Here, we maximize both the average Cross Track Error (XTE) and the maximum XTE. The XTE measures the distance of the center of the vehicle from the median stripe of the road.

## Utils

The file `utils_open_sbt` provides util functions to interface with Open-SBT.

- `individualToScenario`: Converts a individual Open-SBT experiment to a scenario.<br/> :param individual: The individual.<br/> :param variable_names: The List of variable names.<br/> :param road_generator: The generator for generating a new road string.<br/> :param starting_pos: Tuple of the starting position of the vehicel.<br/> :returns Scenario: Returns the Scenario
- `calculate_velocities`: Calculates the velocities at each time step based in the positions and vehicle speed.<br/> :param positions: List of the positions (as tuple of x, y, z values).<br/> :param speeds: List of the float speed values.<br/> :returns Tuple[float, float, float]: Returns the velocities

## Integration into an Experiment

This code sinpped shows how one can integrate the Simulation Wrapped into an experiment. In order to run new experiements, one simply needs to create a new simulation adapter or alter the fitness and criticality function. For more information on Open-SBT visit the [docs](https://git.fortiss.org/opensbt/opensbt-core).

```Python
from problem.adas_problem import ADASProblem
from evaluation.fitness import *
from evaluation.critical import *
from algorithm.nsga2_optimizer import NsgaIIOptimizer
from experiment.search_configuration import DefaultSearchConfiguration
from examples.open_sbt.sdsandbox_open_sbt import SDSandBox_OpenSBTWrapper
from examples.open_sbt.criticality import FitnessFunction, Criticality


problem = ADASProblem(
    problem_name="UdacityRoadGenerationProblem",
    scenario_path="",
    xl=[-10, -10, -10, -10, -10, -10, -10, -10, 0, 1],
    xu=[10, 10, 10, 10, 10, 10, 10, 10, 4, 6],
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

```
