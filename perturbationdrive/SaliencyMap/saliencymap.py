from tensorflow import Variable, GradientTape
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


def getActivationMap(model, img_array, layer_name="conv2d_5"):
    """
    Returns the activation based saliency map for a tensorflow cnn.
    We recommend to use the last convolutional layer for your map, e.g. for
    Dave2 you should use the `conv2d_5` layer

    Parameters:
      - model tf.model: Underlying tf model
      - img_array (numpy array): Image the calculate Grad Cam Map for
      - layer_name str: Layer name to use


    Returns
      uint8 numpy array with shape (img_height, img_width)
    """
    out_layer = model.get_layer(layer_name)
    activation_model = tf.keras.models.Model(
    			inputs=model.inputs,
    			outputs=out_layer.output)
    inputs = tf.cast(img_array, tf.float32)
    activations = activation_model(inputs)

    output = np.abs(activations)
    output = np.sum(output, axis = -1).squeeze()

    #resize and convert to image 
    (w, h) = (img_array.shape[2], img_array.shape[1])
    output = cv2.resize(output, (w, h))
    output /= output.max()
    output *= 255
    return output.astype('uint8')


def getSaliencyMap(model, img, _=None):
    """
    Returns the gradient based saliency map of a tensorflow cnn

    Parameters:
        - img (numpy array): The input image.
        - model: The tensorflow model to evaluate
        - _: Throwaway param needed to make the function sig identical to other attention maps

    Returns: numpy array:
    """
    # calculat the gradient with respect to the top class score to see which
    # pixels contribute the most
    images = Variable(img, dtype=float)

    with GradientTape() as tape:
        pred = model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]

    grads = tape.gradient(loss, images)
    dgrad_abs = tf.math.abs(grads)
    # find the max of the absolutes values of the gradient along each RGB channel
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
    # normalize to range between 0 and 1
    arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
    # this is the grad map
    return grad_eval


def getSaliencyPixels(saliency_map, boundary=0.5):
    """
    Returns boolean masks for a given saliency map

    Parameters:
        - saliency_map (numpy array): Two dimensional saliency map of an image
        - boundary float=0.5: The boundary value for the boolean mask. Needs to be in the range of [0, 1]

    Return: numpy array:
    """
    if boundary < 0.0 or boundary > 1.0:
        raise ValueError("The boundary value needs to be in the range of [0, 1]")
    return saliency_map > boundary


def getSaliencyRegions(saliency_map, boundary):
    """
    Returns a list of all the contours in the image which are higher than the boundary value.
    Each individual contour is represented by a list of points.

    Parameters:
        - saliency_map (numpy array): Two dimensional saliency map of an image
        - boundary float=0.5: The boundary value for the boolean mask. Needs to be in the range of [0, 1]

    Returns: Sequence[MatLike] | Sequence[UMat]
    """
    if boundary < 0.0 or boundary > 1.0:
        raise ValueError("The boundary value needs to be in the range of [0, 1]")
    mask = saliency_map > boundary
    mask_uint8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def plotSaliencyRegions(saliency_map, boundary=0.5):
    """
    Plots all regions for a given saliency map which have a value higher than the boundary condition.

    Parameters:
        - saliency_map (numpy array): Two dimensional saliency map of an image
        - boundary float=0.5: The boundary value for the boolean mask. Needs to be in the range of [0, 1]

    Returns: void
    """
    if boundary < 0.0 or boundary > 1.0:
        raise ValueError("The boundary value needs to be in the range of [0, 1]")
    contours = getSaliencyRegions(saliency_map, boundary)
    # Create a copy of the original array to draw rectangles
    saliency_map_with_rect = np.copy(saliency_map)

    # Iterate over all found contours
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a rectangle on the original image (or a copy of it)
        cv2.rectangle(
            saliency_map_with_rect, (x, y), (x + w, y + h), (1), 2
        )  # (1) for white color in a [0, 1] image, 2 for thickness

    # Plot
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(saliency_map, cmap="jet", alpha=0.8)
    axes[1].imshow(saliency_map_with_rect, cmap="jet", alpha=0.8)
    axes[0].set_title("Original Array")
    axes[1].set_title("Array with Bounding Boxes")
    for ax in axes:
        ax.axis("off")
    plt.show()


def plotImageAndSaliencyMap(image, model):
    """
    Plots an image and its saliency map for a given tensorflow model

    Parameters:
        - saliency_map (numpy array): Two dimensional saliency map of an image
        - model: The tensorflow model to evaluate

    Returns: void
    """
    map = getSaliencyMap(image, model)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    image__ = image / 255.0
    axes[0].imshow(image__)
    i = axes[1].imshow(map, cmap="jet", alpha=0.8)
    fig.colorbar(i)
