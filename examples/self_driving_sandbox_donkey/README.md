# Self Driving Sandbox Donkey Simulator Interface

This directory provides an example on integrating the [Self-Driving Sandbox Donkey (also referred to as SDSandbox)](https://docs.donkeycar.com/guide/deep_learning/simulator/) Simulator with this project.
Please note, that this project uses a fork of the [fortiss automated software testing](https://www.fortiss.org/forschung/forschungsfelder/detail/automated-software-testing) implementation of the fortis [SDSandbox](https://github.com/ast-fortiss-tum/sdsandbox).

➜ The fork is available via this [GitHub Repository](https://github.com/HannesLeonhard/sdsandbox_perturbations).

Note, that both the fortiss implementation and the fork are identical to the original SDSandbox Sim other than the default-tracks and the styling of the tracks.
The [fork](https://github.com/HannesLeonhard/sdsandbox_perturbations) differs from the [fortiss SDSandbox](https://github.com/ast-fortiss-tum/sdsandbox) via the following points:

- Road Generation API: The fork offers the API interface to build custom roads via waypoints send from the client.
- Automatic Setup: The fork offers the possibility to automatically select the `GeneratedTrack`-Scene on initial connection of a client.

Before running these examples

- Install all requirements for this example using `pip install -r requirements.txt`
- Install Open-SBT via pip
- Install perturbation drive via pip

## Table of Contents

- [Simulator Implementation](#simulator-implementation)
  - [SDSandbox Simulator](#sdsandboxsimulator)
  - [Donkey Sim Messanger Handler](#donkey-sim-messanger-handler)
  - [DonkeyProcess](#donkeyprocess)
- [Interface with PerturbationDrive](#interface-with-perturbationdrive)
- [Interface with SDSandBox](#interface-with-sdsandbox)
- [Create a Binary](#create-a-binary)

## Simulator Implementation

This section provides concrete details on the simulator interface implementation and tips on altering your implementation

### SDSandboxSimulator

This section details the implementation of the `PerturbationSimulator` in the `sdsandbox_simulator.py` script via the `SDSandboxSimulator`-class.
The class implements the abstract class methods `connect`, `simulate_scenario` and `tear_down`.

#### SDSandboxSimulator.Class

Initializes the simualtor and sets all class variables.
Here one should set the path to the simualtor binary in the `simulator_exe_path` parameters to gurantee an automatic launch of the binary. Please note, that on mac the file should be an `.app` file, on linux an `.x64_84` or `.x64` and on windows an `.exe` file. The binary should be placed in the `./sim/`-directory.

The class initilaizes the `DonkeyProces` and the `GlobalLog` logger util class.

#### SDSandboxSimulator.connect

Establishes a connection to the SDSandbox Sim via the following steps:

1. Launches the SDSandBox Sim Binary in the `DonkeyProcess` via the start-method in a new process (line 40).
2. Creates a `handler` to hanld the message stream from and to the SDSandbox Sim Binary (line 44).
3. Wraps the `hanlder` in a `SimClient`-object (line 45).
4. Waits for the first observation from the SDSandbox Sim (line 47-57).

#### SDSandboxSimulator.simulate_scenario

Simulates the given scenario on the given agent with a given perturbation controller via the following steps:

1. Uses the `SimClients` message handler to build the given waypoints in the simulator (line 78).
2. Performs the control loop
   1. Sends the next driving commands via the message handler to the simulator and receives the observation from the simulator (line 88).
   2. Checks if the scenario is done or failed (line 92-97)
   3. Perturbs the image based on the scenario (line 100)
   4. Generates the next actions based on the perturbed input (line 106)
   5. Logs simulator metrics (line 109-112)
3. Uses the `SimClients` message handler to reset the position of the vehicle to the starting position (line 122)
4. Returns the `ScenarioOutcome` (line 125)

#### SDSandboxSimulator.tear_down

Disconnects the simualtor interface from the simulator binary

1. Sends a disconnect message to the simualtor via the message handler
2. Quits the process running the SDSandbox Simulator.

### Donkey Sim Messanger Handler

The file `donkey_sim_msg_handler.py` provides the `DonkeySimMsgHandler` handler used to send messages to the SDSandbox Sim and receive messages from the SDSandbox Sim.

Additionally, this class is used to display the current input of the ADS in a dedicated `pygame`-window via the `ImageCallBack` class.

#### DonkeySimMsgHandler.Class

Initializes the message handler class by creating class variables

- `sim_data: Dict`: Data received from the simualtor as dict
- `image_cb: ImageCallBack`: ImageCallback object to view the perturbed images
- `steering_angle: float`: Next steering angle action
- `throttle: float`: Next throttle value
- `fns: Dict`: Dict of function handler names and functions to be called when receiving new data from the Simulator Binary

#### DonkeySimMsgHandler.on_recv_message

Called by the Simulator Binary when a new message is send. The Simualtor Binary either send new telemetry data, or information about the status in from of the stati `car_loaded`, `on_disconnect` or `aborted`.

This message forwards the data from the simulator to the correct message in this class.

Parameters

- `message: Dict`: Simualtor Binary Message containing the key `msg_type` which states the intention of the message and other keys for the telemetry data

#### DonkeySimMsgHandler.on_telemetry

When the Simulator Binary send new telemetry data, the data is forwarded to this method where the data is stored in the `sim_data` class variable. The image is decoded from a base-64 encoded string to a cv2 image.

```Python
self.sim_data = {
    "xte": data["cte"],
    "pos_x": data["pos_x"],
    "pos_y": data["pos_z"],
    "pos_z": data["pos_y"],
    "speed": data["speed"],
    "done": False, 
    "image": image,
}
```

Parameters

- `data: Dict`: Telemetry Data from the Simualtor Binary

#### DonkeySimMsgHandler.update

This method is called by the `SDSandboxSimulator` to send driving commands to the Simulator Binary and receive the latest telemetry data from the Simualtor Binary.

1. Steering commands are send to the simualtor binary (line 153-162)
2. Display the lastest perturbed image and the steering commands (line 165)
3. Return the `sim_data` class variable

Parameters:

- `actions: List[List[float, flaot]]`: Next steering angle and throttle value
- `perturbed_iamge: Union[any, None]`: Image to display in the image callback
_ `perturbation: str=""`: Name of the perturbation applied on the image

#### DonkeySimMsgHandler.reset_scenario

This method is called by the `SDSandboxSimulator` to build a new road in the Simualtor Binary via a string of waypoints.

Parameters:

- `waypoints: Union[str, None]`: String of waypoints seperated by `@`

#### DonkeySimMsgHandler.reset_car

Resets the vehicle to the starting position

### DonkeyProcess

The file `donkey_exec` contains the class `DonkeyProcess`. This class launches the SDSandbox Simulator Binary in a dedicated process. The launched simualtor binary can then interface with the `SDSandboxSimulator`-class.

### DonkeyProcess.class

Util class to launch the simulator in a dedicated subprocess.

### DonkeyProcess.start

Starts a process running the SDSandbox Simualtor.

Parameters:

- `sim_path: str`: Path to the simualtor binary. The binary should be placed in `/examples/seld_driving_sandbox_donkey/sim/`. On windows, the binary is a `.exe` file, on mac a `.app` file and on linux a `.x86` or `.x86_64` file.
- `port: int=9091`: The port of the SDSandbox

### DonkeyProcess.quit

Kills the process running the SDSandbox Simualtor.

## Interface with PerturbationDrive

The `SDSandboxSimulator` can easily be integrated into the `PerturbationDrive` library. Note, that this example needs to be run from a file in the root directory in order to resolve all imports properly.
Also note, that the example agent behaves randomly.

```Python
from examples.self_driving_sandbox_donkey.sdsandbox_simulator import SDSandboxSimulator
from perturbationdrive import PerturbationDrive, RandomRoadGenerator
import traceback
from examples.models.example_agent import ExampleAgent

try:
    simulator = SDSandboxSimulator(
        simulator_exe_path="./examples/self_driving_sandbox_donkey/sim/donkey-sim.app",
        host="127.0.0.1", 
        port=9091
    )
    ads = ExampleAgent()
    road_generator = RandomRoadGenerator(num_control_nodes=8)
    benchmarking_obj = PerturbationDrive(simulator, ads)
    # start the benchmarking
    benchmarking_obj.grid_seach(
        perturbation_functions=["gaussian_noise"],
        attention_map={},
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
```

The method `go` in the file `main.py` provides an example on running the SDSandbox Simulator with the `PerturbationDrive.grid_search method`.

## Interface with SDSandBox

The `SDSandboxSimualator` can easily be intergrated into `OpenSBT` via the wrapper class provieded in `examples/open_sbt`. Please refer to the README.md in `examples/open_sbt` for details on the wrapper class.

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

The method `open_sbt` in the file `main.py` provides an example on running the SDSandbox Simulator with Open SBT.

## Create a binary

Note, the Unity Version used during development is `2022.3.10f1`

1. Install Unity on your device
2. Clone the [fork of the SDSandbox repo](https://github.com/HannesLeonhard/sdsandbox_perturbations) and install all requirements.

    ```bash
    git clone https://github.com/HannesLeonhard/sdsandbox_perturbations
    ```

3. Open Unity Hub and load the project `sdsandbox_perturbations/sdsim`.
4. Install all missing packages, such as the `Unity UI`-package.
5. Click `File` ➜ `Build and Run` to build the test app. Select the interpreter which you need. Make sure to name the binary `donkey-sim.*` and save the binary in the `examples/self_driving_sandbox_donkey/sim/` directory.
