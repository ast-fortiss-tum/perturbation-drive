import tensorflow as tf
from glob import glob
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import os
import numpy as np
import PIL


def discriminator_loss(real, generated):
    real_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.ones_like(real), real)

    generated_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def generator_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.ones_like(generated), generated)


def load_image(image_file):
    """load a image file"""
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    return image


def generator_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(tf.ones_like(generated), generated)


def identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


def random_crop(image):
    """randomly crop image into defined size"""
    cropped_image = tf.image.random_crop(image, size=[256, 256, 3])

    return cropped_image


def normalize(image):
    """normalizing the images to [-1, 1]"""
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def de_normalize(image):
    """De normalize the image to be in range (0,1)"""

    return (image * 0.5) + 0.5


def image_augmentations(image):
    """perform spatial augmentations (rotation and flips) on input image

    from : https://www.kaggle.com/code/dimitreoliveira/improving-cyclegan-monet-paintings
    """

    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    # 90º rotations
    if p_rotate > 0.8:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > 0.6:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > 0.4:
        image = tf.image.rot90(image, k=1)  # rotate 90º
    # ----------------------Flips---------------------
    p_flip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    if p_flip > 0.7:
        image = tf.image.random_flip_left_right(image)
    elif p_flip < 0.3:
        image = tf.image.random_flip_up_down(image)

    return image


def random_jitter(image):
    """resize and randommly crop the input image"""

    # resizing image
    image = tf.image.resize(
        image,
        size=(256, 256),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    # randomly cropping to 512,512
    image = random_crop(image)

    return image


def preprocess_image_train(image):
    image = load_image(image)
    image = random_jitter(image)
    image = image_augmentations(image)
    image = normalize(image)
    return image


# same function, withou the augemntation
def preprocess_image_eval(image):
    image = load_image(image)
    image = random_jitter(image)
    image = normalize(image)
    return image


def create_img_dataset(
    directory,
    image_preprocess_fn,
    buffer_size=100,
    batch_size=2,
    image_extension="jpg",
    repeat=True,
):
    """create a tf dataset object from a directory of images"""
    img_list = glob(directory + f"/*{image_extension}")

    dataset = tf.data.Dataset.list_files(img_list)

    dataset = dataset.map(image_preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset


def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(
        layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_instancenorm:
        # TODO: Custom implementation of Instance Normalization Layer
        # from here https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(
        layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    # TODO: Custom implementation of Instance Normalization Layer
    # from here https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result


def Generator():
    inputs = layers.Input(shape=[256, 256, 3])

    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = layers.Conv2DTranspose(
        3,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name="input_image")

    x = inp

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (bs, 31, 31, 512)

    # TODO: Custom implementation of Instance Normalization Layer
    # from here https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)


class CycleGan(keras.Model):
    def __init__(
        self,
        monet_generator,
        photo_generator,
        monet_discriminator,
        photo_discriminator,
        lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle

    def compile(
        self,
        m_gen_optimizer,
        p_gen_optimizer,
        m_disc_optimizer,
        p_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, batch_data):
        real_monet, real_photo = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(
                real_monet, cycled_monet, self.lambda_cycle
            ) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss = (
                monet_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)
            )
            total_photo_gen_loss = (
                photo_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)
            )

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # Calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(
            total_monet_gen_loss, self.m_gen.trainable_variables
        )
        photo_generator_gradients = tape.gradient(
            total_photo_gen_loss, self.p_gen.trainable_variables
        )

        monet_discriminator_gradients = tape.gradient(
            monet_disc_loss, self.m_disc.trainable_variables
        )
        photo_discriminator_gradients = tape.gradient(
            photo_disc_loss, self.p_disc.trainable_variables
        )

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(
            zip(monet_generator_gradients, self.m_gen.trainable_variables)
        )

        self.p_gen_optimizer.apply_gradients(
            zip(photo_generator_gradients, self.p_gen.trainable_variables)
        )

        self.m_disc_optimizer.apply_gradients(
            zip(monet_discriminator_gradients, self.m_disc.trainable_variables)
        )

        self.p_disc_optimizer.apply_gradients(
            zip(photo_discriminator_gradients, self.p_disc.trainable_variables)
        )

        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss,
        }


# Callback
class GANMonitor(tf.keras.callbacks.Callback):
    """
    A callback to generate and save num_img images after each epoch
    """

    def __init__(
        self,
        Sim_Dataset,
        Real_Dataset,
        real_generator,
        sim_generator,
        num_img=1,
        sim_paths="generated_sim",
        real_paths="generated_real",
    ):
        self.num_img = num_img
        self.day_paths = sim_paths
        self.night_paths = real_paths
        self.Sim_Dataset = Sim_Dataset
        self.Real_Dataset = Real_Dataset
        self.real_generator = real_generator
        self.sim_generator = sim_generator

        # dir to save genereated day images
        if not os.path.exists(self.day_paths):
            os.makedirs(self.day_paths)
            # dir to save genereated night images
        if not os.path.exists(self.night_paths):
            os.makedirs(self.night_paths)

    def on_epoch_end(self, epoch, logs=None):
        # generated night
        for i, img in enumerate(self.Sim_Dataset.take(self.num_img)):
            prediction = self.real_generator(img, training=False)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            prediction = PIL.Image.fromarray(prediction)
            prediction.save(f"{self.night_paths}/generated_{i}_{epoch+1}.png")

        # generated day images
        for i, img in enumerate(self.Real_Dataset.take(self.num_img)):
            prediction = self.sim_generator(img, training=False)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            prediction = PIL.Image.fromarray(prediction)
            prediction.save(f"{self.day_paths}/generated_{i}_{epoch+1}.png")


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """
    Stops training if both the d2n_loss loss and the n2d_loss does not improve
    for patience epochs

    from https://stackoverflow.com/questions/64556120/early-stopping-with-multiple-conditions
    """

    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.n2d_loss = np.Inf
        self.d2n_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        n2d_loss = np.max(logs.get("monet_gen_loss"))
        d2n_loss = np.max(logs.get("photo_gen_loss"))

        # If both the conditions are met, continue training
        if np.less(n2d_loss, self.n2d_loss) and np.less(d2n_loss, self.d2n_loss):
            self.d2n_loss = d2n_loss
            self.n2d_loss = n2d_loss
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()

        # if above xondition not met, continue training till patiance epochs
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def train_cycle_gan(
    input_dir,
    output_dir,
    image_extension_input="jpg",
    image_extension_output="jpg",
    buffer_size=100,
    batch_size=2,
    early_stop_patience=None,
    epochs=50,
    steps_per_epoch=300,
):
    """
    Trains a CycleGan to use with the PerturbationDrive Project.
    Saves two generators to cycle back and forth between the two image dataset domains.

    Parameters:
        - input_dir: Relative path to the input images
        - output_dir: Relative path to the ouput images
        - image_extension_input="jpg": File extension for input images
        - image_extension_output="jpg": File extension for output images
        - buffer_size=100: Buffer Size of Datasets
        - batch_size=2: Batch Size of Datasets
        - early_stop_patience=None: Stops the training process if the loss does not decrease for `early_stop_patience` epochs.
            If `None`, the training will continue until the end of all epochs.
        - epochs=50: Amount of Epochs to train for.
        - steps_per_epoch=300: Steps per epoch to take.
    """
    # create datasets
    # sim_list = glob(input_dir + f"/*{image_extension_input}")
    Sim_Dataset = create_img_dataset(
        directory=input_dir,
        image_preprocess_fn=preprocess_image_train,
        buffer_size=buffer_size,
        batch_size=batch_size,
        image_extension=image_extension_input,
    )

    # real_list = glob(output_dir + f"/*{image_extension_output}")
    Real_Dataset = create_img_dataset(
        directory=output_dir,
        image_preprocess_fn=preprocess_image_train,
        buffer_size=buffer_size,
        batch_size=batch_size,
        image_extension=image_extension_output,
    )

    sim_generator = Generator()  # transforms real photos to simulation photos
    real_generator = Generator()  # transforms simulation photos to real pictures

    sim_discriminator = Discriminator()  # differentiates real sim and generated sim
    real_discriminator = Discriminator()  # differentiates real and generated real

    monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    cycle_gan_model = CycleGan(
        real_generator, sim_generator, real_discriminator, sim_discriminator
    )

    cycle_gan_model.compile(
        m_gen_optimizer=monet_generator_optimizer,
        p_gen_optimizer=photo_generator_optimizer,
        m_disc_optimizer=monet_discriminator_optimizer,
        p_disc_optimizer=photo_discriminator_optimizer,
        gen_loss_fn=generator_loss,
        disc_loss_fn=discriminator_loss,
        cycle_loss_fn=calc_cycle_loss,
        identity_loss_fn=identity_loss,
    )

    def scheduler(epoch, lr, decay_rate=0.05, warm_up_period=10):
        if epoch < warm_up_period:
            return lr
        elif epoch > warm_up_period and epoch < 40:
            return lr * tf.math.exp(decay_rate)
        else:
            return lr * tf.math.exp(decay_rate * 2)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    gan_monitor = GANMonitor(Sim_Dataset, Real_Dataset, real_generator, sim_generator)

    callbacks = [lr_scheduler, gan_monitor]
    # early stopping
    if early_stop_patience is not None:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="gen_N2D_loss",
            mode="min",
            patience=early_stop_patience,
            restore_best_weights=True,
        )
        callbacks.append(early_stop)

    cycle_gan_model.fit(
        tf.data.Dataset.zip((Real_Dataset, Sim_Dataset)),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )

    real_generator.save("sim2real_gen_e50.h5")
    sim_generator.save("real2sim_gen_e50.h5")
