import tensorflow as tf

def fgsm_attack(model, image, y, epsilon):
    """
    Creataes an adver

    Args:
    - model: The target model.
    - x: The original inputs.
    - y: The true labels.
    - epsilon: The maximum perturbation.

    Returns:
    - Adversarial examples.
    """
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    x = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
      tape.watch(x)
      prediction = model(x)
      loss = loss_object(y, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, x)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return image + epsilon * signed_grad