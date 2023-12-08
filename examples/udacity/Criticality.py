from typing import Tuple
from evaluation.fitness import Fitness
from evaluation.critical import Critical
from simulation.simulator import SimulationOutput
import numpy as np
from udacity_utils.envs.udacity.config import MAX_CTE_ERROR


class UdacityFitnessFunction(Fitness):
    """
    Fitness function simply returns average xte

    We aim at finding scenarios with a high xte since these exhibit bad driving scenarios
    """

    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Average xte", "Max xte"

    def eval(self, simout: SimulationOutput) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]

        return (np.average(traceXTE), max(traceXTE))


class UdacityCriticality(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # we fail the scenario, if max xte > 3
        return vector_fitness[1] > MAX_CTE_ERROR
