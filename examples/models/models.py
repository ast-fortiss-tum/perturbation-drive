"""
Models
Define the different NN models we will use
Author: Tawn Kramer
"""
from __future__ import print_function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, SpatialDropout2D
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers.experimental import SGD

try:
    import examples.self_driving_sandbox_donkey.conf as conf
except:
    print("trying to resolve conf import through relative path")
    from ..self_driving_sandbox_donkey import conf


def show_model_summary(model):
    model.summary()
    for layer in model.layers:
        print(layer.output_shape)


def get_nvidia_model(num_outputs):
    """
    this model is inspired by the NVIDIA paper
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    Activation is RELU
    """
    row, col, ch = conf.row, conf.col, conf.ch

    drop = 0.1

    img_in = Input(shape=(row, col, ch), name="img_in")
    x = img_in
    # x = Cropping2D(cropping=((10,0), (0,0)))(x) #trim 10 pixels off top
    # x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
    x = Lambda(lambda x: x / 255.0)(x)
    x = Conv2D(24, (5, 5), strides=(2, 2), activation="relu", name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Conv2D(32, (5, 5), strides=(2, 2), activation="relu", name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (5, 5), strides=(2, 2), activation="relu", name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation="relu", name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation="relu", name="conv2d_5")(x)
    x = Dropout(drop)(x)

    x = Flatten(name="flattened")(x)
    x = Dense(100, activation="relu")(x)
    # x = Dropout(drop)(x)
    x = Dense(50, activation="relu")(x)
    # x = Dropout(drop)(x)

    outputs = []
    outputs.append(Dense(num_outputs, activation="linear", name="steering_throttle")(x))

    model = Model(inputs=[img_in], outputs=outputs)
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss="mse", metrics=["acc"])
    return model


def epoch_model(num_outputs):
    # https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/cg23/epoch_model.py
    row, col, ch = conf.row, conf.col, conf.ch

    img_input = Input(shape=(row, col, ch), name="img_in")

    x = Conv2D(32, 3, 3, activation="relu", border_mode="same")(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, 3, 3, activation="relu", border_mode="same")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, 3, 3, activation="relu", border_mode="same")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    y = Flatten()(x)
    y = Dense(1024, activation="relu")(y)
    y = Dropout(0.5)(y)
    y = Dense(num_outputs)(y)

    model = Model(input=img_input, output=y)
    model.compile(optimizer=Adam(lr=1e-4), loss="mse")

    return model


def chauffeur_model(num_outputs):
    row, col, ch = conf.row, conf.col, conf.ch
    input_shape = (row, col, ch)
    learning_rate = 0.01
    use_adadelta = True
    W_l2 = 0.0001
    scale = 16

    # https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/chauffeur/README.md
    model = Sequential()
    model.add(
        Conv2D(
            16,
            5,
            5,
            input_shape=input_shape,
            init="he_normal",
            activation="relu",
            border_mode="same",
        )
    )
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(20, 5, 5, init="he_normal", activation="relu", border_mode="same"))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(40, 3, 3, init="he_normal", activation="relu", border_mode="same"))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(60, 3, 3, init="he_normal", activation="relu", border_mode="same"))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(80, 2, 2, init="he_normal", activation="relu", border_mode="same"))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(128, 2, 2, init="he_normal", activation="relu", border_mode="same")
    )
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=num_outputs, init="he_normal", W_regularizer=L2(W_l2)))
    optimizer = "adadelta" if use_adadelta else SGD(lr=learning_rate, momentum=0.9)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["rmse"])
    return model
