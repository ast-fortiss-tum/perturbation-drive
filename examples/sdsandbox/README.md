# How to run sdsandbox

To run this example you will need the sdsandbox/sdsim unity project on your device. You can find this unity project in [this GitHub repo](https://github.com/ast-fortiss-tum/sdsandbox).

We recommend to use this [fork](https://github.com/HannesLeonhard/sdsandbox_perturbations) of the official sdsandbox simulator as it will allways contain the latest updates on the main branch.

Make sure to install all requirements for this example using `pip install -r requirements.txt`

## Run Instructions

1. Launch a Unity Test App and open the USI Track.
2. Hit the start button to launch. Then the "Use NN Steering". When you hit this button, the car will disappear. This is normal. You will see one car per client that connects.
3. Run either the predict_client `python3 examples/sdsandbox/predict_client.py` or the monitor_client `python3 examples/sdsandbox/monitor_client.py`
    - You can specify multiple argument when running the scrip such as the model to use, the sim server or the server port to bind to as well as details on the perturbations to use.
    - Use the argument `--model` to specify the path to the model to test, e.g. `--model=./examples/sdsandbox/model_1_11.h5`. This argument is required.
    - Use the argument `--host` to specify the host server. The default is `127.0.0.1`
    - Use the argument `--port` to specify the port of your host server. The default is `9091`, however other processes such as JupyterNotebook might occupy this port. If you strugle with finding a port, try using port `9090`.
    - User the argument `--perturbation` to specify which perturbation you want to use, e.g. `--perturbation=defocus_blur`. This parameter can be used an arbitrary amount of times and per default all perturbations which are fast enought for the simulator will be used.
    - Use the argument `--attention_map` to spedify if you want to perturb the image only in regions of the image where the attention map has a value greater than the `attention_threshold`. You can use the options `vanilla` or `grad_cam`. Per default the whole image is perturbed.
    - Use the argument `--attention_threshold` to specicy your threshold for perturbating based on the attention map.
    - Use the argument `--attehtion_layer` to specify which layer should be used to calculate the Grad Cam map. Per default we use the layer `conv2d_5`. It is recommended to use the last convolutional layer.

This list contains a couple of example configurations:

1. `python3 examples/sdsandbox/monitor_client.py --model=./examples/sdsandbox/model_1_11.h5 --perturbation=defocus_blur --perturbation=increase_brightness --perturbation=pixelate`
2. `python3 examples/sdsandbox/monitor_client.py --model=./examples/sdsandbox/model_1_11.h5 --attention_map=vanilla --attention_threshold=0.4`
3. `python3 examples/sdsandbox/monitor_client.py --model=./examples/sdsandbox/model_1_11.h5 --perturbation=defocus_blur --attention_map=grad_cam --port=9090`
4. `python3 examples/sdsandbox/predict_client.py --model=./examples/sdsandbox/model_1_11.h5 --perturbation=defocus_blur`. Please note, that by running only the predict client, you will not be able to view the perturbated images in a seperate monitor.

## Specifications on the sdsandbox simulator and Unity Version

1. Clone the [fortiss-tum sdsandbox repository](https://github.com/ast-fortiss-tum/sdsandbox) and install all requirements.
2. Checkout the `crossroad`-branch using `git checkout crossroad`
3. Open Unity and load the folder `sdsandbox/sdsim`. Make sure to install all missing packages, such as the `Unity UI`-package. The unity version used during the development of this project is Unity 2022.3.10f1.
4. Click `File` ➜ `Build and Run` to build the test app. This should launch a Unity Test App in a new window (this app will be called `unity-test.app`).
5. If you want to log your driving you will need to choose a `Log dir` in the Start Menu.
6. Open the `USI Track`.

## Train your own model

1. Create your dataset by following `Specifications on the sdsandbox simulator`.
2. Run the script to train you model:
    - You need to specify your model name via `--model`
    - You need to specify the relative path to your input logs `--inputs`. This path needs to end with '*.*'.
    - You can specify the amount of epochs. The default is 200, however training will stop if there is no improvement.

This is an example script execution.

```bash
python3 examples/sdsandbox/train_sdsandbox_model.py --model=your_name --epochs=200 --inputs=../relative/path/to/your/inputs/*.*

python3 examples/sdsandbox/train_sdsandbox_model.py --model=your_name --epochs=200 --inputs="./../../../../Desktop/dataset_1.11./*.*"
```

## Download the models

You can download pretrained models if you do not want to train you own models.

1) Navigate into the directory via `cd examples/sdsandbox`
2) Make the setup script executable `chmod +x setup.sh`
3) Execute the setup script `./setup.sh`. This will download the models `model_1_11.h5`, `mymodel.h5` and `mymodel_15_10.h5`
