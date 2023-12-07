# Instructions

## Installation

To use this example, you need to install all pip packages, `open-sbt` and `perturbationdrive`

- Download all required python packages via pip from via the requirements.txt.
- Download `open-sbt` via git `https://git.fortiss.org/opensbt/opensbt-core.git` and install it locally via pip.
  - You might need to add additionaly `__init__.py` files in the following places  
- Download `perturbationdrive` via git `https://github.com/HannesLeonhard/PerturbationDrive/tree/feature/open_sbt` and install it locally via pip.

Download the binary of udacity. You can find precompilled files for Windows/Linux/macOS [here](https://drive.google.com/drive/folders/1wljVnkjUlYF3ILLqxybKowj0M6cZatAg?usp=drive_link).
The binary needs to be placed in the following folder `examples/udacity/udacity_utils/sim/`
Additionally you will need download the models for the ADS and place them here `examples/sdsandbox_perturbations`

## Example

Run the example via

```Bash
python examples/udacity/main.py
```

This will create an ADAS Problem with the following configuration. The problem creates roads with 8 waypoints where
each waypoint is connected by the angle simulation parameters.

```Python
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
```

The fitness function tries to maximize both the average and maximum cross track error to find failure scenarios.
A scenario is considered as failed, if the car has a cross track error higher than 3 at any time.

## Known Issues

- If no road can be generated for the given configuration, we generate a random road. This behavior should be reflected in the SearchConfiguration
