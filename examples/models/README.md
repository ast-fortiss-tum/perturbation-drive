# How to train your own models

To train your model you will need your own dataset. You can generate a dataset for the supported simulators by following the specifications in the simulator examples.

## Train DAVE-2

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

## Example Agent

The script `example_agent.py` contains the example implementation of an agent based on the dave2 architecture trained on donkey images.
This agent is used in the `udacity` and `sdsandbox` examples.
