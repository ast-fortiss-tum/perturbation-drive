from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

def gradCam(model, img_array, layer_name="conv2d_5"):
    """
    Creates a grad-cam heatmap for a given model and a layer name contained with that model. 
    We recommend using the last convolutional layer of the model. Eg in terms of the underlying Dave2 model
    the `conv2d_5` layer.

    Inspired from this [GradCam Implementation](https://colab.research.google.com/drive/1rxmXus_nrGEhxlQK_By38AjwDxwmLn9S()

    Parameters:
      - model tf.model: Underlying tf model
      - img_array (numpy array): Image the calculate Grad Cam Map for
      - layer_name str: Layer name to use


    Returns
      uint8 numpy array with shape (img_height, img_width)

    """

    gradModel = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        (convOutputs, predictions) = gradModel(inputs)
        loss = predictions[:, ]
    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)

    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))
    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min())
    heatmap = numer / denom

    # heatmap = (heatmap * 255).astype("uint8")
    # return the resulting heatmap to the calling function
    return heatmap
