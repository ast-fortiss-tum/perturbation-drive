# How to run sdsandbox_perturbations

To run this example you will need the sdsandbox/sdsim unity project on your device. You can find this unity project in [this GitHub repo](https://github.com/ast-fortiss-tum/sdsandbox).

The script provides a command-line interface for running simulations in various modes (e.g., offline, grid search). It requires configuring parameters like simulator path, perturbation types, and attention maps.
Users can customize the simulation by selecting different perturbation functions and adjusting the simulation settings according to their testing needs.

We recommend to use this [fork](https://github.com/HannesLeonhard/sdsandbox_perturbations) of the official sdsandbox simulator as it will allways contain the latest updates on the main branch.

Dependencies

- Make sure to install all requirements for this example using `pip install -r requirements.txt`
- Make sure to install Open-SBT if you want to use it
- Make sure to install perturbation drive

## Run Instructions

1. Launch a Unity Test App and open the USI Track.
2. Hit the start button to launch. Then the "Use NN Steering". When you hit this button, the car will disappear. This is normal. You will see one car per client that connects.
3. Run the script `python3 examples/sdsandbox_perturbations/main.py`
    - You can specify multiple argument when running the scrip such as the model to use, the sim server or the server port to bind to as well as details on the perturbations to use.
    - Use the argument `--host` to specify the host server. The default is `127.0.0.1`
    - Use the argument `--port` to specify the port of your host server. The default is `9091`, however other processes such as JupyterNotebook might occupy this port. If you strugle with finding a port, try using port `9090`.
    - User the argument `--perturbation` to specify which perturbation you want to use, e.g. `--perturbation=defocus_blur`. This parameter can be used an arbitrary amount of times and per default all perturbations which are fast enought for the simulator will be used.
    - Use the argument `--attention_map` to spedify if you want to perturb the image only in regions of the image where the attention map has a value greater than the `attention_threshold`. You can use the options `vanilla` or `grad_cam`. Per default the whole image is perturbed.
    - Use the argument `--attention_threshold` to specicy your threshold for perturbating based on the attention map.
    - Use the argument `--attehtion_layer` to specify which layer should be used to calculate the Grad Cam map. Per default we use the layer `conv2d_5`. It is recommended to use the last convolutional layer.
4. You can choose to run either grid search, offline perturbations or open sbt. This is done by either running `go` or `open_sbt` in the main statement.

## Specifications on the sdsandbox_perturbations simulator and Unity Version

1. Clone the [fortiss-tum sdsandbox_perturbations repository](https://github.com/ast-fortiss-tum/sdsandbox) and install all requirements.
2. Checkout the `crossroad`-branch using `git checkout crossroad`
3. Open Unity and load the folder `sdsandbox_perturbations/sdsim`. Make sure to install all missing packages, such as the `Unity UI`-package. The unity version used during the development of this project is Unity 2022.3.10f1.
4. Click `File` ➜ `Build and Run` to build the test app. This should launch a Unity Test App in a new window (this app will be called `unity-test.app`).
5. If you want to log your driving you will need to choose a `Log dir` in the Start Menu.
6. Open the `USI Track`.
