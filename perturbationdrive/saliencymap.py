from tensorflow import Variable, GradientTape
import tensorflow.math as tf_math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def getSaliencyMap(img, model):
    """given an image and a model this returns the saliency map"""
    # calculat the gradient with respect to the top class score to see which
    # pixels contribute the most
    images = Variable(img, dtype=float)

    with GradientTape() as tape:
        pred = model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]

    grads = tape.gradient(loss, images)
    dgrad_abs = tf_math.abs(grads)
    # find the max of the absolutes values of the gradient along each RGB channel
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
    # normalize to range between 0 and 1
    arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
    # this is the grad map
    return grad_eval


def getSaliencyPixels(saliency_map, boundary):
    """Returns boolean masks for a given saliency map"""
    return saliency_map > boundary


def getSaliencyRegions(saliency_map, boundary):
    mask = saliency_map > boundary
    mask_uint8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def plotSaliencyRegions(saliency_map, boundary):
    """Plots """
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
    axes[0].imshow(saliency_map, cmap="jet",alpha=0.8)
    axes[1].imshow(saliency_map_with_rect, cmap="jet",alpha=0.8)
    axes[0].set_title("Original Array")
    axes[1].set_title("Array with Bounding Boxes")
    for ax in axes:
        ax.axis("off")
    plt.show()


def perturbSaliencyRegions(saliency_map, image, boundary, perturbation, scale):
    """Perturbs the regions of an image where the saliency map has an value greater than boundary"""
    # Create a binary mask from the array
    mask = saliency_map > boundary
    # Apply the gaussian noise to the whole image
    noise_img = perturbation(scale, image)
    # Now apply the mask: replace the original image pixels with noisy pixels where mask is True
    image[mask] = noise_img[mask]
    return image


def plotImageAndSaliencyMap(image, model):
    """Plots an image and its saliency map"""
    map = getSaliencyMap(image, model)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    image__ = image / 255.0
    axes[0].imshow(image__)
    i = axes[1].imshow(map, cmap="jet", alpha=0.8)
    fig.colorbar(i)
