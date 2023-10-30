import tensorflow as tf

def pgd_attack(model, image, y, epsilon, alpha, num_iterations):
    """
    Creataes an adver

    Args:
    - model: The target model.
    - x: The original inputs.
    - y: The true labels.
    - epsilon: The maximum perturbation.
    - alpha: The step size per iteration
    - num_iterations: The number of PGD iterations.

    Returns:
    - Adversarial examples.
    """
    # Convert x to a tensor and normalize
    x = tf.convert_to_tensor(image, dtype=tf.float32)
    x = x / 255.0  # normalize if values are in [0, 255]
    x_adv = x
    for _ in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            prediction = model(x_adv)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, prediction)
        gradient = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(gradient)
        x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    x_adv = x_adv * 255
    return x_adv