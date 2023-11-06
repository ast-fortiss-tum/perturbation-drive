# How to run sdsandbox

To run this example you will need the sdsandbox/sdsim unity project on your device. You can find this unity project in [this GitHub repo](https://github.com/ast-fortiss-tum/sdsandbox).

Make sure to install all requirements for this example using `pip install -r requirements.txt`

1. Load the Unity project sdsandbox/sdsim in Unity. Double click on Assets/Scenes/road_generator to open that scene.
2. Hit the start button to launch. Then the "Use NN Steering". When you hit this button, the car will disappear. This is normal. You will see one car per client that connects.
3. Run either the predict_client `python3 examples/sdsandbox/predict_client.py` or the monitor_client `python3 examples/sdsandbox/monitor_client.py`

## Download the models

You can download pretrained models if you do not want to train you own models.

1) Navigate into the directory via `cd examples/sdsandbox`
2) Make the setup script executable `chmod +x setup.sh`
3) Execute the setup script `./setup.sh`. This will download the models `model_1_11.h5`, `mymodel.h5` and `mymodel_15_10.h5`
