# Instructions

Instructions on running OpenSBT in the udacity simulator to search for failure scenarios. Each scenario considers a road
and a perturbation with a scale of the perturbation.

## Installation

To use this example, you need to install all pip packages, `open-sbt` and `perturbationdrive`

- Download `open-sbt` via git `https://git.fortiss.org/opensbt/opensbt-core.git` and install it locally via pip.
  - Clone the Core Open SBT repo from GitLab
  - Navigate into the root folder (opensbt-core)
  - You might need to add additionaly `__init__.py` files in the following folders: `experiment`, `algorithm/classification/decision_tree`, `quality_indicators`, `exception`, and `visualization`.
  This will gurantee a smooth import of all Open SBT packages.
  - Install all packages locally `pip install .`
- Download `perturbationdrive` via git `https://github.com/HannesLeonhard/PerturbationDrive/tree/feature/open_sbt` and install it locally via pip.
  - Clone the repository from GitHub.
  - Navigate into the root folder (Perturbationdrive)
  - Install all packages locally `pip install .`
- Download all required python packages via pip from via the requirements.txt. If you have a Mac with an M1/M2 chip, use the requirements_macM1.txt.
  - Navigate into `perturbationdrive/examples/udacity` and install all requirements `pip install -r requirements.txt`

Download the binary of udacity. You can find precompilled files for Windows/Linux/macOS [here](https://drive.google.com/drive/folders/1wljVnkjUlYF3ILLqxybKowj0M6cZatAg?usp=drive_link).
The binary needs to be placed in the following folder `examples/udacity/udacity_utils/sim/`
Additionally you will need download the models for the ADS and place them here `examples/sdsandbox_perturbations`

## Example

Run the example via

```Bash
python examples/udacity/main.py
```

This will create an ADAS Problem with the following configuration. The problem creates roads with 8 waypoints where two adjacent waypoints are
are connected via a curve having the angle specified in the simulation variables. Additionally, it selected one out of six perturbation functions
and a scale to perturbate the input.

```Python
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
    fitness_function=UdacityFitnessFunction(),
    critical_function=UdacityCriticality(),
    simulate_function=UdacitySimulator.simulate,
    simulation_time=30,
    sampling_time=0.25,
)
```

The fitness function tries to maximize both the average and maximum cross track error to find failure scenarios.

```Python
class UdacityFitnessFunction(Fitness):
    @property
    def min_or_max(self):
        return "max", "max"

    @property
    def name(self):
        return "Average xte", "Max xte"

    def eval(self, simout: SimulationOutput) -> Tuple[float]:
        traceXTE = [abs(x) for x in simout.otherParams["xte"]]

        return (np.average(traceXTE), max(traceXTE))
```

We consider a scenario as ciritcal, if the maximum cross track error during the scenario is higher than half of the road width, e.g. the
car goes off the road during the scenario.

```Python
from udacity_utils.envs.udacity.config import MAX_CTE_ERROR

class UdacityCriticality(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        return vector_fitness[1] > MAX_CTE_ERROR
```

## ADS Selection

Per default we use a pretrained ADS based on the Dave2 architecture. If this model is not present when executing the experiment, the model is downloaded and
stored in the appropriate folder.
If you want to use your own model, you need to place it in this folder `./examples/models/` and call it `generatedRoadModel.h5`.

Please notice, that this model will receive an image with the dimensions (1, 240, 320, 3) and the dtype float32 and the model must return an array [[steering_angle, throttle]].

## Trouble Shooting

- If the udacity simulator fails to start, check your available ports. Per default, we use `BASE_PORT = 4567` to connect to the simulator.
  You can change the port by updating the port in `examples/udacity/udactiy_utils/envs_udacity`.

## Known Issues

- If no road can be generated for the given configuration, we generate a random road. This behavior should be reflected in the SimulationOutput.
- Currently the perturbation functions have an ordering, i.e. from 1 to 6. This does not make sense, as there is no natural ordering between perturbations.
- Setup a monitor to view the perturbated images.
