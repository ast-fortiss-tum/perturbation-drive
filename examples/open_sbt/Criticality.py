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
        return "max", "max"

    @property
    def name(self):
        return "Average xte", "Max xte"

    def eval(self, simout: SimulationOutput) -> Tuple[float]:
        try:
            traceXTE = [abs(x) for x in simout.otherParams["xte"]]
            return (np.average(traceXTE), max(traceXTE))
        except Exception as e:
            print(f"Fitness Function: Exception is {e}")
            return (0.0, 0.0)


class Criticality(Critical):
    def __init__(self, max_xte: float = 4.0):
        self.max_xte = max_xte

    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3
        return vector_fitness[1] > self.max_xte
