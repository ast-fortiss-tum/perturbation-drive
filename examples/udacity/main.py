# related to open_sbt
from problem.adas_problem import ADASProblem
from evaluation.fitness import *
from evaluation.critical import *
from algorithm.nsga2_optimizer import NsgaIIOptimizer
from experiment.search_configuration import DefaultSearchConfiguration
from utils import log_utils

# Related to example
from udacity_simulator import UdacitySimulator
from Criticality import UdacityFitnessFunction, UdacityCriticality

if __name__ == "__main__":
    # Define search problem
    problem = ADASProblem(
        problem_name="UdacityRoadGenerationProblem",
        scenario_path="",
        xl=[-10, -10, -10, -10, -10, -10, -10, -10, 0],
        xu=[10, 10, 10, 10, 10, 10, 10, 10, 4],
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
        ],
        fitness_function=UdacityFitnessFunction(),
        critical_function=UdacityCriticality(),
        simulate_function=UdacitySimulator.simulate,
        simulation_time=30,
        sampling_time=0.25,
    )

    log_utils.setup_logging("./log.txt")

    # Set search configuration
    config = DefaultSearchConfiguration()
    config.n_generations = 50
    config.population_size = 20

    # Instantiate search algorithm
    optimizer = NsgaIIOptimizer(problem=problem, config=config)

    # Run search
    res = optimizer.run()

    # Write results
    res.write_results(params=optimizer.parameters)
