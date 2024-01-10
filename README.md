# Perturbation Drive

A library to test the robstuness of Self-Driving-Cars via image perturbations.

This library is split up into three sections:

1) A collection of the most common perturbation functions which can be used by itself.
2) A benchmarking controller to benchmark the robustness of models to common image perturbations. This controller can be used for end-to-end tests and offline testing.
3) Modular simulator integration. New simulators can be integrated by implementing a predefined interface and this project shows the example integration of the [Self Driving Sandbox Donkey](https://docs.donkeycar.com/guide/deep_learning/simulator/) (here often referred as SDSandbox) and [Udacity](https://github.com/udacity/self-driving-car-sim) Simulator. Note, that the simualtors used in this project have been adapted minimally to suits the scope of this project.

![PerturbationDrive Overview Library Architecture](./docAssets/perturbationDriveOverview.png "PerturbationDrive Overview")

## Installation

You can install the library using pip

```Shell
pip install perturbationdrive
```

After installing this library via `pip` you can use all perturbationdrive classes and functions via top level imports. See this examples for the usage.

```Python
from perturbationdrive import (
    PerturbationDrive,
    RandomRoadGenerator,
    PerturbationSimulator,
    ADS,
    Scenario,
    ScenarioOutcome,
    ImageCallBack,
    ImagePerturbation,
    GlobalLog as Gl,
)
```

## Project Structure

The projct has the following structure. Please note that this only provides a high level overview and does not detail all files and scripts.

```bash
perturbationdruve/
│
├── perturbationdrive/                  # All scripts related to running perturbations
│   ├── AutomatedDrivingSystem/         # Contains all script regarding the ADS interface
│   │   └── ADS.py                      # Abstract Class of the driving system
│   │
│   ├── Generative/                     # Contains all script regarding cycle gan perturbations
│   │   ├── Sim2RealGan.py              # Implementation of a CycleGAN from Donkey Sim to Real World
│   │   └── TrainCycleGan.py            # Script to train a CycleGAN given a dataset
│   │
│   ├── kernels/                        # Util scripts regarding image kernels
│   │
│   ├── NeuralStyleTransfer/            # Implementation of the fast style transfer models
│   │
│   ├── OverlayImages/                  # Util Folder containing all images used for perturbations
│   │
│   ├── OverlayMasks/                   # Util Folder containing all videos for dynamic masks
│   │
│   ├── RoadGenerator/                  # Contains all script regarding road generation
│   │   ├── RoadGenerator.py            # Abstract base class of the road generator
│   │   ├── RandomRoadGenerator.py      # Generates a random road
│   │   └── CustomRoadGenerator.py      # Generates a raod given angles and segment lengths between waypoints
│   │
│   ├── SaliencyMap/                    # Contains all scripts regarding attention map generation
│   │
│   ├── Simulator/                      # Contains all script regarding simualtors and scenarios
│   │   ├── Simualtor.py                # Abstract base class of the simulator
│   │   ├── Scenario.py                 # Data-Classes for Scenarios and Scenario Outcomes
│   │   └── image_callback.py           # Provides functionality to view images in a second monitor
│   │
│   ├── utils/                          # Contains all util scripts of the library
│   │
│   ├── README.md                       # Further Documentation on the perturbations library
│   ├── imageperturbations.py           # Perturbation Controller
│   ├── perturbationdrive.py            # Benchmarking Controller
│   └── perturbationsfuncs.py           # Collection of image perturbations
│
├── examples /                          # Provides examples on simualtor integrations
│   ├── models/                         # Example implenetation of the ADS class
│   │   ├── README.md                   # Documentation and Explanation on the example
│   │   └── example_agent.py            # Example subclass of the ADS
│   │
│   ├── open_sbt/                       # Documentation and Examples of OpenSBT integration
│   │   └── README.md                   # Documentation and Explanation on the example
│   │
│   ├── sdsandbox_perturbations/        # Example integration of Self Driving Sandbox Donkey Sim
│   │   ├── README.md                   # Documentation and Explanation on the SDSandbox Simualtor
│   │   ├── sdsandbox_simulator.py      # Simulator class implementation for the SDSandbox Donkey Sim 
│   │   └── main.py                     # Entry point to run the example
│   │
│   └── udacity/                        # Example integration of Udacity Sim
│       ├── README.md                   # Documentation and Explanation on the Udacity Simualtor
│       ├── udacity_simulator.py        # Simulator class implementation for the Udacity Simulator
│       └── main.py                     # Entry point to run the example
│
├── work.pdf                            # TODO: Insert final thesis pdf here 
├── README.md                           # The top level ReadME of the project (this file)
└── requirements.txt                    # Requirements for running this project
```

## Performing Image Perturbations

Apply common image perturbations and corruptions to images.
Each perturbation needs an input image and the scale of the perturbation as input.
The scale is in the range from 0 to 4.

```Python
from perturbationdrive import poisson_noise

perturbed_image = poisson_noise(image, 0)

```

## Benchmarking Self-Driving Cars

View the examples folder for examples on how to run this benchmarking library with different simulators. There are examples for the SDSandbox simulator and the udacity simulator.

The benchmarking is performed by the `PerturbationDrive` class. This class can perform either offline evaluation of a dataset, perform gird search over the entire search space or simulate a list of specific scenarions.

A benchmarking object can be created by instanciating a new `PerturbationDrive` object. Each object needs to be constructed with the simulator under test and the system under test. The system under test is an `Autonomous Driving System (ADS)`.

```Python
benchmarking_object = PerturbationDrive(simulator=simulator, ads=ads)
```

### Simulator

`PerturbationSimulator` serves as an abstract base class for creating simulator adapters. It is primarily designed for use in automated driving system (ADS) simulations, where various scenarios and image perturbations are applied to evaluate and test ADS responses.

This repo contains two example simulator adapters.

- Udacity Simulator Adapter:<br/>This example along with instructions on running it can be found in the `examples/udacity/` folder.
- SDSandBox Simulator Adapter:<br/>This example along with instruction on running it can be found in the `examples/sdsandbox_perturbations` folder.

#### Simulator-Constructor

`__init__(self, max_xte: float, simulator_exe_path: str, host: str, port: int, initial_pos: Union[Tuple[float, float, float, float], None])`
Initializes the simulator with the specified parameters.

`max_xte: float` (default = 2.0):
Maximum cross-track error allowed.
`simulator_exe_path: str`(default = ""):
Path to the simulator executable.
`host: str` (default = "127.0.0.1"):
Host address for connecting to the simulator.
`port: int` (default = 9091):
Port number for the connection.
`initial_pos: Union[Tuple[float, float, float, float], None]` (default = None):
Initial position in the format (x, y, z, angle).

#### Simulator-Methods

`connect(self)`
Connects to the simulator. This method must be overridden in the subclass.

`simulate_scanario(self, agent: ADS, scenario: Scenario, perturbation_controller: ImagePerturbation) -> ScenarioOutcome`
Simulates a given scenario.

Parameters:

- `agent: ADS`: The automated driving system agent.
- `scenario: Scenario`: The scenario to be simulated.
- `perturbation_controller`: ImagePerturbation: Controller for image perturbations.

Returns:

- `ScenarioOutcome`: The outcome of the scenario simulation.

Description:
This method simulates the specified scenario, applying perturbations to the observations and evaluating the ADS's actions. The process includes resetting the simulator, building the scenario waypoints, and running an action loop where the ADS responds to perturbed observations.

`tear_down(self)`
Tears down the connection to the simulator. This method must be overridden in the subclass.

### ADS

`ADS` (Automated Driving System) is an abstract base class designed to simulate the behavior of automated driving systems. It serves as a blueprint for creating concrete implementations that model the decision-making process of an automated vehicle based on visual input.

This repo contains an example agent implementation in the folder `/examples/ads/`.

#### ADS-Methods

`action(self, input: ndarray[Any, dtype[uint8]]) -> List`
Description:
This method represents a single action step for the automated driving system. It takes as input a computer vision image (in the form of a NumPy array) and outputs a decision or a series of actions that the automated driving system should take in response.

Parameters:

- `input: ndarray[Any, dtype[uint8]]`: The input image to the system, typically in the form of an array with uint8 data type. This image is processed and used to make driving decisions.

Returns:

- `Tuple[Tuple[float, float]]`: A tuple of tuple pf actions or decisions made by the automated driving system based on the input image. The first tuple value is the steering angle and the second tuple value is the throttle value.

Implementation Requirements:
Implementers of this method should include all the necessary preprocessing of the input image and the logic for making driving decisions. This may involve using machine learning models, heuristic algorithms, or a combination of different techniques.

### Scenario

The `Scenario` data class is designed to model a scenario in the context of automated driving system (ADS) simulations. It encapsulates the key elements of a scenario, including the waypoints, perturbation function, and perturbation scale. This class has the following attributes

`waypoints: Union[str, None]`

- Description:<br/>This attribute represents the waypoints for the scenario. Waypoints are critical in defining the path or route that the ADS should follow or react to during the simulation.
- Type:<br/>`Union[str, None]` – It can be a string representing the waypoints' data or None if waypoints are not applicable or not defined. Waypoints should be seperated by the `@`-char, e.g. `1.0,1.0,1.0@2.0,2.0,2.0@3.0,3.0,2.0`.

`perturbation_function: str`

- Description:<br/>Defines the specific function or method used to introduce perturbations in the scenario. Perturbations are variations or disturbances introduced in the simulation environment, which can affect the behavior or performance of the ADS.
- Type:<br/>`str` – A string that identifies or names the perturbation function to be applied in the scenario.

`perturbation_scale: int`

- Description:<br/>Specifies the scale or intensity of the perturbation. This determines how significantly the perturbation will impact the scenario, allowing for control over the difficulty or complexity of the simulation for the ADS.
- Type:<br/>`int` – An integer value representing the magnitude or extent of the perturbation. This should be in the range of [0;4].

### Road Generator

`RoadGenerator` is an abstract base class (ABC) designed for generating new roads in simulations involving automated driving systems (ADS). It provides a standardized approach for creating diverse road scenarios.

There are already two concrete `RoadGenerator` subclasses implemented in this project

- `RandomRoadGenerator`:<br/>Generates a random road for a given map size. The map size needs to be spcified in the constructor of this road.
- `CustomRoadGenerator`:<br/> Generates a road based on the specified angles and segment lengths. The constructor also requires a map size for this generator.<br/>The `generate` method expects the parameter `angles` containing a list of integer angles and optionaly the parameter `seg_lengths`, containing a list of integer segment lengths.

#### Road Generator Method

`generate(self, *args, **kwargs) -> Union[str, None]`

- Description:<br/>This method is responsible for generating a new road and returning its string representation. The method allows for flexibility in terms of input parameters using *args and **kwargs. A key requirement is that kwargs must contain the initial starting position as an argument named starting_pos.
- Return Type: <br/>`Union[str, None]` - The method returns a string representation of the generated road. An example format could be 1.0,1.0,1.0@2.0,2.0,2.0@3.0,3.0,2.0, representing a sequence of waypoints or coordinates. The method can also return None if a road cannot be generated under given conditions or parameters.
- Implementation Requirements:<br/>Subclasses must provide a concrete implementation of this method, defining how roads are generated. This could involve algorithms for random road generation, procedures for creating roads based on specific criteria, or methods to produce roads that challenge the ADS in unique ways.

## Grid-Search

Performs Grid Search over the entire input space. The input space is defined by all perturbations, and their scale.
Optionally, the user can alter the roads between different perturbation difficulties by specifying a road generator. If the road generator supplied is `None` or allways returns the same raod, grid search is only performed over all specified perturbations.

```Python
from perturbationdrive import PerturbationDrive, RandomRoadGenerator

# Setup the simulator
simulator = ExampleSimulator(
    simulator_exe_path=simulator_exe_path,
    host=host,
    port=port,
)
# Setup the ADS
ads = ExampleAgent()
road_generator = RandomRoadGenerator(map_size=250)
benchmarking_obj = PerturbationDrive(simulator, ads)
# start the benchmarking
benchmarking_obj.grid_seach(
    perturbation_functions=pert_funcs,
    attention_map=attention,
    road_generator=road_generator,
    log_dir="./examples/example/logs.json",
    overwrite_logs=True,
    image_size=(240, 320),  # images are resized to these values
)
```

If the list of perturbation functions supplied is empty, we perform all non-generative perturbations.

### Grid-Search Method

Parameters

- perturbation_functions: `List[str]`:<br/>A list of strings representing the perturbation functions to be tested.
- attention_map: `Dict (default = {})`:<br/>A dictionary representing the attention map for image perturbations.
- road_generator: `Union[RoadGenerator, None]` (default = None):<br/>An optional RoadGenerator instance to generate roads for the scenarios. If None, no road generation is performed.
- log_dir: `Union[str, None]` (default = "logs.json"):<br/>The directory for storing log files. If None, scenario outcomes are returned instead of being logged.
- overwrite_logs: `bool` (default = True):<br/>Indicates whether to overwrite existing logs.
- image_size: `Tuple[float, float]` (default = (240, 320)):<br/>The size of the images to be used for perturbations.

Return Value

- `Union[None, List[ScenarioOutcome]]`:<br/>The method returns a list of ScenarioOutcome objects if log_dir is None. Otherwise, the outcomes are logged, and None is returned.

Method Description

1. The grid_search method executes a grid search over different perturbations and scales to assess the performance of an ADS in various scenarios. The method follows these steps:
2. Initialize Image Perturbation: Based on the provided perturbation functions, attention map, and image size.
3. Perturbation and Scenario Setup: Iterates over the perturbation functions and scales, generating new scenarios each time. An empty perturbation is always included for comparison.
4. Simulator Setup and Connection: Connects to the simulator and waits for the connection to establish.
5. Road Generation: If a road_generator is provided, generates new roads using the starting position from the simulator.
6. Grid Search Loop: Iterates over the perturbations and scales. For each iteration, a new Scenario is created and simulated. The outcome is then evaluated, and unsuccessful perturbations (except the empty one) are removed from the list.
7. Logging and Returning Outcomes: If a log directory is specified, the outcomes are written to the logs. Otherwise, the list of outcomes is returned.
8. Simulator Teardown: Disconnects and tears down the simulator setup.

### Attention Based Perturbation

Optionally, you can perturbate the images based on the attention map of the model on the input image. To enable this feature, you will need to specify the attention parameters when creating the `ImagePerturbation` object. The `ImagePerturbation` constructor requires an dictionary contining the following information:

- `map`: A string value if the object should calculate the vanilla saliency map or calculate the Grad Cam map. This parameter is required. Possible values are either `vanilla` or `grad_cam`.
- `model`: The underlying model. This value if required.
- `threshold=0.5`: A float value in the range of [0, 1]. If a input image pixel achieves a value higher or equal to this threshold we apply the perturbation in this region. This param is not required and the default is 0.5.
- `layer="conv2d_5"`: The string name of the model layer which shalll be used to calculate the Grad Cam map. This param is not required and the default is "conv2d_5".

 ```Python
# Instantiate a perturbation object
funcs = [
    "elastic"
]
attention = {
    "map": "grad_cam",
    "model": model,
    "threshold": 0.4,
    "layer": "conv2d_3"
}
# start the benchmarking
benchmarking_obj.grid_seach(
    perturbation_functions=funcs,
    attention_map=attention,
)
```

## Scenario Simulation

Simulates a list of scenarios and returns their outcome. Each scenario needs to contain a perturbation function, perturbation scale and optionally a road string.

```Python
outcomes: List[ScenarioOutcome] = benchmarking_obj.simulate_scenarios(
    scenarios=scenarios,
    attention_map={},
    log_dir=None,
    overwrite_logs=False,
    image_size=(240, 320),
)
```

### Scenario Simulation Method

Parameters

- scenarios: `List[Scenario]``:<br/>A list of Scenario objects to be simulated.
- attention_map: `Dict` (default = {}):<br/>A dictionary representing the attention map for image perturbations.
- log_dir: `Union[str, None]` (default = "logs.json"):<br/>The directory for storing log files. If None, scenario outcomes are returned directly.
- overwrite_logs: `bool` (default = True):<br/>Indicates whether to overwrite existing log files.
- image_size: `Tuple[float, float]` (default = (240, 320)):<br/>The size of the images to be used for perturbations.

Return Value

- `Union[None, List[ScenarioOutcome]]`:<br/>Returns a list of ScenarioOutcome objects if log_dir is None. If log_dir is specified, the outcomes are written to the logs, and None is returned.

Method Description

1. The simulate_scenarios method performs the following steps:
2. Perturbation Setup: Collects all the perturbation functions from the provided scenarios to set up the ImagePerturbation object.
3. Image Perturbation Initialization: Creates an ImagePerturbation instance using the collected perturbation functions, attention map, and image size.
4. Simulating Each Scenario: Iterates over each scenario in the provided list, simulating them using the simulate_scanario method of the simulator instance. Each scenario is executed with the configured image perturbation controller.
5. Collecting Outcomes: For each simulated scenario, the outcome is appended to a list of ScenarioOutcome objects.
6. Simulator Teardown: After simulating all scenarios, the simulator is torn down.
7. Logging or Returning Outcomes: If a log directory is provided, the outcomes are written to the specified log file. Otherwise, the outcomes are returned directly.

## Offline Perturbation

The `offline_perturbation` method is designed to apply perturbations to a dataset of images and assess the impact on an automated driving system (ADS). It calculates an error rate by comparing the ADS's responses to both perturbed and original images.

### Offline Perturbation Method

Parameters

- dataset_path: str:<br/>Path to the dataset directory containing images and JSON files.
- perturbation_functions: List[str]:<br/>A list of perturbation function names to be applied to the images.
- attention_map: Dict (default = {}):<br/>A dictionary representing the attention map for image perturbations.
- log_dir: Union[str, None] (default = "logs.json"):<br/>Directory for storing log files. If None, the scenario outcomes are returned.
- overwrite_logs: bool (default = True):<br/>Indicates whether to overwrite existing log files.
- image_size: Tuple[float, float] (default = (240, 320)):<br/>The size of the images for perturbations.

Return Value

- Union[None, List[OfflineScenarioOutcome]]:<br/>Returns a list of OfflineScenarioOutcome objects if log_dir is None. Otherwise, the outcomes are logged, and None is returned.
Method Description

The offline_perturbation method performs the following steps:

1. Dataset Validation: Checks if the dataset directory exists and contains the required JSON and image files.
2. Perturbation Setup: Initializes the ImagePerturbation object with the specified perturbation functions, attention map, and image size.
3. Processing Each Image: Iterates through each image in the dataset, performing the following actions:
    - Retrieves the corresponding JSON file for ground truth data (steering angle and throttle).
    - Reads the image and applies each perturbation function at different intensities.
    - For each perturbed image, calculates the ADS's response and compares it with the response to the original image.
    - Storing Results: Collects the outcomes in a list of OfflineScenarioOutcome objects, including details like file names, perturbation functions, scales, and actions.
4. Logging or Returning Outcomes: If a log directory is provided, the results are written to the specified file. Otherwise, the results are returned directly.

### Offline Perturbation DataSet Spefifications

All files in the dataset must adhere to this naming specification for the offline_perturbation method to function correctly. The method assumes this specific format to correctly pair each image with its corresponding JSON file.

Image Files

- Format: `{frame_number}_{free_string}.{extension}`
- Components:
  - {frame_number}: A unique identifier for each frame, typically a number. This should correlate directly with the frame number used in the corresponding JSON file.
  - {free_string}: Any string of length greater than one character. This part of the filename can be used for additional descriptive information or identifiers.
  - {extension}: The file extension, indicating the image format. Accepted formats are .jpg, .jpeg, or .png.

JSON Files

- Format: `record_{frame_number}.json`
- Components:
  - record_: A fixed prefix for all JSON files in the dataset.
  - {frame_number}: The same frame number as used in the corresponding image file. This ensures that each JSON file is correctly associated with its respective image.
  - .json: The file extension for JSON files.
- JSON File Content
  - Each JSON file must contain the following ground truth values:
    - `user/angle`: The steering angle.
    - `user/throttle`: The throttle value.

## Open-SBT Adapter

This library can easily be run together with [OpenSBT](https://git.fortiss.org/opensbt) by creating a wrapping the simulator and perturbation drive object in the Open SBT Simulator.

Creating an OpenSBT Wrapper boils down to three steps:

1. Create a Simulator Subclass
2. Create a Fittness and Criticality Function
3. Create an ADASProblem, SearchConfiguration and run the search

All example folder contain examples in the `main.py` file on running OpenSBT with the simulator and this library.

## CycleGANs

This library partly relies on CycleGANs to translate images into another domain, such as from simulated images into real images. You can specify such perturbations such as:

```Python
# Instantiate a perturbation object
funcs = [
    "sim2real", 
]
benchmarking_obj.grid_seach(
    perturbation_functions=funcs,
)
```

In order to use these perturbations you will either download the models vie `curl` or train your own CycleGAN

### Download the models

1) Navigate into the directory via `cd perturbationdrive/Generative`.
2) Make the setup script executable `chmod +x setup.sh`.
3) Execute the setup script `./setup.sh`. This will download the generative models `donkey_sim2real.h5` and `donkey_sim2sim.h5`.

### Train you own model

You will need two folders containing the unlabeled images from your two domains. It is important that the input domain is equivalent to your simualtor input. Then you can train your model such as.

```Python
from perturbationdrive import train_cycle_gan

train(
    input_dir="./relative/path/to/folder",
    output_dir="./relative/path/to/folder",
    image_extension_input="jpg",
    image_extension_output="jpg",
    buffer_size=100,
    batch_size=2,
    early_stop_patience=None,
    epochs=50,
    steps_per_epoch=300,
)
```

This will automatically save the two generative models `donkey_sim2real.h5` and `donkey_sim2sim.h5`.

After each epoch, two images will be generated in a `generated_sim` and `generated_real` folder which show the performance of the generators.

## Neural Style Transfer

If you want to perturb your images using neural style transfer based on the ideas and models of `Perceptual Losses for Real-Time Style Transfer and Super- Resolution, Johnson et al., 2016`, you first need to download the models via a setup script.

1) Navigate into the directory via `cd perturbationdrive/NeuralStyleTransfer`
2) Make the setup script executable `chmod +x setup.sh`
3) Execute the setup script `./setup.sh`. This will create the folders `perturbationdrive/NeuralStyleTransfer/models/instance_norm` and `perturbationdrive/NeuralStyleTransfer/models/eccv16` with all relevant Neural Style Transfer models.

Happy Testing!

## Local setup

To set this library up locally, navigate into the folder of this library.

### Contributing and extending this library

Make sure you have all requirements for this library setup.

```Shell
pip install -r requirements.txt
```

### Installing this library locally

```Shell
pip install .
```
