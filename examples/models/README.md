# Models

This folder provides examples on implementing the `ADS` interface and training models to test with this framework.
We offer the possibility to train the `Dave2` model using your own dataset.

## Table of Contents

- [Example Agent](#example-agent)
- [Train Dave-2](#train-dave-2)
  - [Requirements on the training data](#requirements-on-the-training-data)

## Example Agent

The script `example_agent.py` contains the example implementation of an agent based on the Dave2 architecture trained on donkey and udacity images.
This agent is used in the `udacity` and `sdsandbox` examples.

### ExampleAgent.Class

The initializer loads the model from the `"./examples/models/generatedRoadModel.h5"` file and compiles the model.

### ExampleAgent.Action

Takes one action step given the input, here the input is a cv2 image.
First the input is converted from uint8 to float32, and then the input image is reshaped to add a batch dimension.
Adding a batch dimension allows to to make predictions without relying on the `.predict` method which has an significant overhead compared to passing the input through the model. This overhead occurs, due to `.predict` wrapping the input into a tf.Dataset before making predictions.

Parameters

- `input: ndarray[Any, dtype[uint8]]`: Observation from a single front facing camera

Returns

- `List[List[float, float]]`: Returns a list containing the steering angle and the throttle value.

## Train DAVE-2

To train your model you will need your own dataset. You can generate a dataset for the supported simulators by following the specifications in the simulator examples.

1. Create your dataset by following sdsandbox specifications.
2. Run the script to train you model:
    - You need to specify your model name via `--model`
    - You need to specify the relative path to your input logs `--inputs`. This path needs to end with '*.jpg'.
    - You can specify the amount of epochs. The default is 200, however training will stop if there is no improvement.

When defining your model name you can opt for the different [tensorflow model](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model) formats of `.keras`, `.h5` or saved model format.

The training will also automatically create a loss graph with the file name loss.png.

Here you can find an example script execution.

```bash
python3 examples/models/train_dave2.py --model=your_name.h5 --epochs=200 --inputs="./relative/path/to/your/inputs/*.jpg"
```

### Requirements on the training data

The training data in the `inputs` folder needs to follow these specifications.

- For each data points, you will need a .jpg iamge, and a .json file containing the label.
- The names of the files should be 'record_x.json', 'x_cam-image_array_.jpg', with x representing the frame of this training input.
- The json file needs to contain the fields `user/angel` and `user/throttle`, detailing the throttle and steering angle during this frame.
