# How to run sdsandbox

Make sure to install all requirements for this example using `pip install -r requirements.txt`

1. Load the Unity project sdsandbox/sdsim in Unity. Double click on Assets/Scenes/road_generator to open that scene.
2. Hit the start button to launch. Then the "Use NN Steering". When you hit this button, the car will disappear. This is normal. You will see one car per client that connects.
3. Run either the predict_client `python3 examples/donkeyGym/predict_client.py` or the monitor_client `python3 examples/donkeyGym/monitor_client.py`
