# Udacity Simulator Interface

This directory provides an example on integrating the [Self-Driving Sandbox Donkey (also referred to as SDSandbox)](https://github.com/udacity/self-driving-car-sim) Simulator with this project.
Please note, that this project uses a fork of the project provided by [fortiss automated software testing](https://www.fortiss.org/forschung/forschungsfelder/detail/automated-software-testing).

➜ The precompilled binaries can be found in [this google drive](https://drive.google.com/drive/folders/1wljVnkjUlYF3ILLqxybKowj0M6cZatAg?usp=drive_link) or in [this GitHub Repo](https://github.com/ast-fortiss-tum/udacity-test-generation?tab=readme-ov-file#udacity-driving-simulator).

Note, that the fortiss implementation and the fork are identical to the original Udacity Sim other than the default-tracks and the styling of the tracks.

Before running these examples

- Install all requirements for this example using `pip install -r requirements.txt`
- Install Open-SBT via pip
- Install perturbation drive via pip

## Table of Contents

- [Simulator Implementation](#simulator-implementation)
  - [UdacitySimulator](#udacitysimulator)
  - [UdacityGymEnv_RoadGen](#udacitygymenv_roadgen)
  - [UdacitySimController]()
  - [UnityProcess](#unityprocess)
- [Interface with PerturbationDrive](#interface-with-perturbationdrive)
- [Interface with SDSandBox](#interface-with-sdsandbox)

## Simulator Implementation

This section provides concrete details on the simulator interface implementation and tips on altering your implementation

### UdacitySimulator

This section details the implementation of the `PerturbationSimulator` in the `udacity_simulator.py` script via the `UdacitySimulator`-class.
The class implements the abstract class methods `connect`, `simulate_scenario` and `tear_down`.

Additionally, this class is used to display the current input of the ADS in a dedicated `pygame`-window via the `ImageCallBack` class.

#### UdacitySimulator.Class

Initializes the simualtor and sets all class variables.
Here one should set the path to the simualtor binary in the `simulator_exe_path` parameters to gurantee an automatic launch of the binary. Please note, that on mac the file should be an `.app` file, on linux an `.x64_84` or `.x64` and on windows an `.exe` file. The binary should be placed in the `./sim/`-directory.

The class initilaizes the `GlobalLog` logger util class.

#### UdacitySimulator.connect

Establishes a connection to the Udacity Sim via the following steps:

1. Launches the Udacity Sim Binary via the creation of a new UdacityGymEnv_RoadGen (line 41).
2. Waits for the first observation from the Udacity Sim (line 47-49).

#### UdacitySimulator.simulate_scenario

Simulates the given scenario on the given agent with a given perturbation controller via the following steps:

1. Uses the `UdacityGymEnv_RoadGen` env to build the given waypoints in the simulator (line 77).
2. Performs the control loop
   1. Perturbs the image based on the scenario (line 86)
   2. Generates the next actions based on the perturbed input (line 91)
   3. Displays the perturbed image via the `ImageCallBack` class
   4. Sends the next driving commands via the env to the simulator and receives the observation from the simulator (line 106).
   5. Logs simulator metrics (line 109-112)
3. Uses the env to reset the position of the vehicle to the starting position (line 120)
4. Returns the `ScenarioOutcome` (line 122)

#### UdacitySimulator.tear_down

Disconnects the simualtor interface from the simulator binary via the `UdacityGymEnv_RoadGen.close` method.

### UdacityGymEnv_RoadGen

The file `udacity_utils/envs/udacity_gym_env.py` provides a gmy interface for the udacity simulator

#### UdacityGymEnv_RoadGen.Class

Initializes the gym env

- `seed: int`: Random seed
- `exe_path: str`: Path to the udacity binary used to simulate the scenario

Additionally, the following steps are performed:

1. `GlobalLog`-class is initialized (line 34)
2. The `UnityProcess` is initialized and started (line 41-60)
3. The `UdacitySimController` is initialized (line 62)
4. The action and observation space are set (line 65-71)

#### UdacityGymEnv_RoadGen.step

Performs an action in the udacity env and returns the observation and info after taking the action

Parameters

- `action: np.ndarray`: Next driving commands. action[0] is the steering angle and action[1] is the throttle value

Returns

- `Tuple[np.ndarray, bool, Dict]`: Returns a tuple of next observation (equal to image of the cars camera), if the scenario is done and environment indo dict.

#### UdacityGymEnv_RoadGen.reset

This method is called to build a new road in the Simualtor Binary via a string of waypoints.

Parameters

- `skip_generation: bool=False`: Boolean value indicating if the track generation should be skipped
- `track_string: Union[str, None]=None`: String of the waypoints (x,y,z) of the new track sperated by `@`.

Returns

- `np.ndarray`: Observation after the track is reset.

#### DonkeySimMsgHandler.observe

Used to get the current observation from the env.

Returns:

- `Tuple[np.ndarray, bool, Dict]`: Returns a tuple of next observation (equal to image of the cars camera), if the scenario is done and environment indo dict.

#### DonkeySimMsgHandler.close

Closes the environment and quits the `UnityProcess`

### UnityProcess

The file `donkey_exec` contains the class `DonkeyProcess`. This class launches the SDSandbox Simulator Binary in a dedicated process. The launched simualtor binary can then interface with the `SDSandboxSimulator`-class.

#### UnityProcess.class

Util class to launch the simulator in a dedicated subprocess.

#### UnityProcess.start

Starts a process running the Udacity Simualtor.

Parameters:

- `sim_path: str`: Path to the simualtor binary. The binary should be placed in `/examples/udacity/sim/`. On windows, the binary is a `.exe` file, on mac a `.app` file and on linux a `.x86` or `.x86_64` file.
- `port: int=9091`: The port of the SDSandbox

#### UnityProcess.quit

Kills the process running the Udacity Simualtor.

### UdacitySimController

The `UdacitySimController` class in `examples/udacity/udacity_utils/envs/udacity/core/udacity_sim.py` provides a wrapper for communicating with unity simulation.

It utilizes `socketio` and `flask` to create a client for communication with the Udacity Sim Binary.

## Interface with PerturbationDrive

The `SDSandboxSimulator` can easily be integrated into the `PerturbationDrive` library. Note, that this example needs to be run from a file in the root directory in order to resolve all imports properly.
Also note, that the example agent behaves randomly.

```Python
from examples.udacity.udacity_simulator import UdacitySimulator
from perturbationdrive import PerturbationDrive, RandomRoadGenerator, GridSearchConfig
import traceback
from examples.models.example_agent import ExampleAgent

try:
    simulator = UdacitySimulator(
        simulator_exe_path="./examples/udacity/udacity_utils/sim/udacity_sim.app",
        host="127.0.0.1",
        port=9091,
    )    
    ads = ExampleAgent()
    road_generator = RandomRoadGenerator(num_control_nodes=8)
    benchmarking_obj = PerturbationDrive(simulator, ads)
    # start the benchmarking
    benchmarking_obj.grid_seach(
        config=GridSearchConfig(
            perturbation_functions=["gaussian_noise", "poisson_noise"],
        )
    )
    print(f"{5 * '#'} Finished Running Udacity Sim {5 * '#'}")
except Exception as e:
    print(
        f"{5 * '#'} Udacity Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
    )
```

The method `go` in the file `main.py` provides an example on running the SDSandbox Simulator with the `PerturbationDrive.grid_search method`.

## Interface with SDSandBox

The `UdacitySimulator` can easily be intergrated into `OpenSBT` via the wrapper class provieded in `examples/open_sbt`. Please refer to the README.md in `examples/open_sbt` for details on the wrapper class.

```Python
# Define search problem
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
    simulate_function=Udacity_OpenSBTWrapper.simulate,
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

The method `open_sbt` in the file `main.py` provides an example on running the SDSandbox Simulator with Open SBT.
