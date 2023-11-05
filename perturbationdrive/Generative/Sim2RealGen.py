import tensorflow as tf
import numpy as np
import cv2


class Sim2RealGen:
    def __init__(self) -> None:
        self.sim2real = tf.keras.models.load_model(
            "perturbationdrive/Generative/donkey_sim2real.keras"
        )
        self.real2sim = tf.keras.models.load_model(
            "perturbationdrive/Generative/donkey_real2sim.keras"
        )

    def toSim(self, image):
        # expect an image uints
        (h, w) = image.shape[:2]
        img_arr = tf.image.resize(
            image, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        img_arr = tf.cast(img_arr, tf.float32)
        img_arr = (img_arr / 127.5) - 1
        image_tensor = tf.constant(img_arr)

        # Add an extra dimension to represent the batch size.
        image_tensor_batch = tf.expand_dims(image_tensor, axis=0)
        generated_real = self.real2sim.predict(image_tensor_batch, verbose=0)

        # move image to original shape and datatype
        generated_real = generated_real[0] * 0.5 + 0.5
        generated_real = np.clip(generated_real * 255, 0, 255).astype(np.uint8)
        return cv2.resize(generated_real, (h, w), interpolation=cv2.INTER_AREA)

    def toReal(self, image):
        # expect an image uints
        (h, w) = image.shape[:2]
        img_arr = tf.image.resize(
            image, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        img_arr = tf.cast(img_arr, tf.float32)
        img_arr = (img_arr / 127.5) - 1
        image_tensor = tf.constant(img_arr)

        # Add an extra dimension to represent the batch size.
        image_tensor_batch = tf.expand_dims(image_tensor, axis=0)
        generated_real = self.sim2real.predict(image_tensor_batch, verbose=0)

        # move image to original shape and datatype
        generated_real = generated_real[0] * 0.5 + 0.5
        generated_real = np.clip(generated_real * 255, 0, 255).astype(np.uint8)
        return cv2.resize(generated_real, (h, w), interpolation=cv2.INTER_AREA)

    def real2real(self, image):
        # expect an image uints
        (h, w) = image.shape[:2]
        img_arr = tf.image.resize(
            image, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        img_arr = tf.cast(img_arr, tf.float32)
        img_arr = (img_arr / 127.5) - 1
        image_tensor = tf.constant(img_arr)

        # Add an extra dimension to represent the batch size.
        image_tensor_batch = tf.expand_dims(image_tensor, axis=0)
        generated = self.real2sim.predict(image_tensor_batch, verbose=0)
        real2real = self.sim2real.predict(generated, verbose=0)

        # move image to original shape and datatype
        generated_real = real2real[0] * 0.5 + 0.5
        generated_real = np.clip(generated_real * 255, 0, 255).astype(np.uint8)
        return cv2.resize(generated_real, (h, w), interpolation=cv2.INTER_AREA)

    def sim2sim(self, image):
        # expect an image uints
        (h, w) = image.shape[:2]
        img_arr = tf.image.resize(
            image, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        img_arr = tf.cast(img_arr, tf.float32)
        img_arr = (img_arr / 127.5) - 1
        image_tensor = tf.constant(img_arr)

        # Add an extra dimension to represent the batch size.
        image_tensor_batch = tf.expand_dims(image_tensor, axis=0)
        generated = self.sim2real.predict(image_tensor_batch, verbose=0)
        sim2sim = self.real2sim.predict(generated, verbose=0)

        # move image to original shape and datatype
        generated_real = sim2sim[0] * 0.5 + 0.5
        generated_real = np.clip(generated_real * 255, 0, 255).astype(np.uint8)
        return cv2.resize(generated_real, (h, w), interpolation=cv2.INTER_AREA)
