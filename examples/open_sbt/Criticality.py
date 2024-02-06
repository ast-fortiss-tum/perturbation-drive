# open sbt imports
from evaluation.fitness import Fitness
from evaluation.critical import Critical
from simulation.simulator import SimulationOutput

# other imports
from typing import Tuple
import numpy as np


class FitnessFunction(Fitness):
    """
    Fitness function simply returns average xte

    We aim at finding scenarios with a high xte since these exhibit bad driving scenarios
    """

    def __init__(self, max_xte: float = 4.0):
        self.max_xte = max_xte

    @property
    def min_or_max(self):
        return "max", "max", "min", "max"

    @property
    def name(self):
        return "Average XTE", "TTF", "Criticality", "Max XTE"

    def eval(self, simout: SimulationOutput) -> Tuple[float]:
        try:
            traceXTE = [abs(x) for x in simout.otherParams["xte"]]
            criticality = int(simout.otherParams["timeout"]) + (
                1 - int(simout.otherParams["isSuccess"])
            )
            fitness = (
                np.average(traceXTE),
                simout.otherParams["ttf"],
                criticality,
                max(traceXTE),
            )
            print("Fitness is", fitness)
            return fitness
        except Exception as e:
            print(f"Fitness Function: Exception is {e}")
            return (0.0, 0.0, 0.0, 0.0)


class Criticality(Critical):
    def __init__(self, max_xte: float = 4.0):
        self.max_xte = max_xte

    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if we had either a timeout or a out of bounds
        if vector_fitness[2] < 0:
            print("Found critical scenario")
        else:
            print("Found non-critical scenario")
        return vector_fitness[2] < 0
