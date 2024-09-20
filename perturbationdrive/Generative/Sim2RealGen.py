import tensorflow as tf
import numpy as np
import cv2

# we need tfa here because the model uses tfa.layers.InstanceNormalization
# and without the import we cannot compile the model
import tensorflow_addons as tfa
import os
from ..utils import download_file


class Sim2RealGen:
    def __init__(self) -> None:
        if not os.path.exists("perturbationdrive/Generative/donkey_sim2real.h5"):
            # download and move file. Note that the link is valid until 1.1.2030
            print("Fetching generative model real2sim")
            download_file(
                "https://syncandshare.lrz.de/dl/fiHFK1MTb6aAK2JE2osQeL/donkey_real2sim.h5",
                "perturbationdrive/Generative",
            )
        self.sim2real = tf.keras.models.load_model(
            "perturbationdrive/Generative/donkey_sim2real.h5"
        )
        if not os.path.exists("perturbationdrive/Generative/donkey_real2sim.h5"):
            print("Fetching generative model sim2real")
            # Note that the link is valid until 1.1.2030
            download_file(
                "https://syncandshare.lrz.de/dl/fi5udoYWWeHS5zajmjXGha/donkey_sim2real.h5",
                "perturbationdrive/Generative",
            )
        self.real2sim = tf.keras.models.load_model(
            "perturbationdrive/Generative/donkey_real2sim.h5"
        )
        print("\n\nsetup generative models\n\n")
        # warum up the models with empty tensor
        dummy = tf.expand_dims(tf.zeros([256, 256, 3]), axis=0)
        for _ in range(10):
            self.serve2Real(dummy)
            self.serve2Sim(dummy)

    @tf.function
    def preprocess_image(self, image):
        img_arr = tf.image.resize(
            image, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        img_arr = tf.cast(img_arr, tf.float32)
        img_arr = (img_arr / 127.5) - 1
        # Add an extra dimension to represent the batch size.
        image_tensor_batch = tf.expand_dims(img_arr, axis=0)
        return image_tensor_batch

    @tf.function
    def serve2Sim(self, image):
        return self.real2sim(image, training=False)

    def toSim(self, image):
        # expect an image uints
        (h, w) = image.shape[:2]

        # prprocess the image
        image_tensor_batch = self.preprocess_image(image)
        generated_real = self.serve2Sim(image_tensor_batch)
        # move image to original shape and datatype
        generated_real = generated_real[0] * 0.5 + 0.5
        generated_real = np.clip(generated_real * 255, 0, 255).astype(np.uint8)
        return cv2.resize(generated_real, (w, h), interpolation=cv2.INTER_AREA)

    @tf.function
    def serve2Real(self, image):
        return self.sim2real(image, training=False)

    def toReal(self, image):
        # expect an image uints
        (h, w) = image.shape[:2]

        # prprocess the image
        image_tensor_batch = self.preprocess_image(image)
        generated_real = self.serve2Real(image_tensor_batch)

        # move image to original shape and datatype
        generated_real = generated_real[0] * 0.5 + 0.5
        generated_real = np.clip(generated_real * 255, 0, 255).astype(np.uint8)
        return cv2.resize(generated_real, (w, h), interpolation=cv2.INTER_AREA)

    def real2real(self, image):
        # expect an image uints
        (h, w) = image.shape[:2]

        # prprocess the image
        image_tensor_batch = self.preprocess_image(image)
        generated = self.serve2Sim(image_tensor_batch)
        real2real = self.serve2Real(generated)

        # move image to original shape and datatype
        generated_real = real2real[0] * 0.5 + 0.5
        generated_real = np.clip(generated_real * 255, 0, 255).astype(np.uint8)
        return cv2.resize(generated_real, (w, h), interpolation=cv2.INTER_AREA)

    def sim2sim(self, image):
        # expect an image uints
        (h, w) = image.shape[:2]

        # prprocess the image
        image_tensor_batch = self.preprocess_image(image)
        generated = self.serve2Real(image_tensor_batch)
        sim2sim = self.serve2Sim(generated)

        # move image to original shape and datatype
        generated_real = sim2sim[0] * 0.5 + 0.5
        generated_real = np.clip(generated_real * 255, 0, 255).astype(np.uint8)
        return cv2.resize(generated_real, (w, h), interpolation=cv2.INTER_AREA)
