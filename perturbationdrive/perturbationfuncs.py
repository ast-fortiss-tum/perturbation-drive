import numpy as np
import cv2
from scipy.ndimage import zoom as scizoom
from io import BytesIO


def gaussian_noise(scale, img):
    """
    Adds unfirom distributed gausian noise to an image

    Parameters:
        - img (numpy array): The input image.
         - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array
    """
    factor = [0.06, 0.12, 0.22, 0.30, 0.42][scale]
    # scale to a number between 0 and 1
    x = np.array(img, dtype=np.float32) / 255.0
    # add random between 0 and 1
    return (
        np.clip(x + np.random.normal(size=x.shape, scale=factor), 0, 1).astype(
            np.float32
        )
        * 255
    )


def poisson_noise(scale, img):
    """
    Adds poisson noise to an image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array: Image with salt and pepper noise.
    """
    factor = [80, 30, 10, 5, 2][scale]
    x = np.array(img) / 255.0
    return np.clip(np.random.poisson(x * factor) / float(factor), 0, 1) * 255


def impulse_noise(scale, img):
    """
    Add salt and pepper noise to an image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array: Image with salt and pepper noise.
    """
    factor = [0.02, 0.08, 0.10, 0.19, 0.30][scale]
    # Number of salt noise pixels
    num_salt = np.ceil(factor * img.size * 0.5)
    # Add salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    img[tuple(coords)] = 255
    # Number of pepper noise pixels
    num_pepper = np.ceil(factor * img.size * 0.5)
    # Add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    img[tuple(coords)] = 0
    return img


def _create_disk_kernel(radius):
    """Create a disk-shaped kernel with the given radius."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    kernel[mask] = 1
    # Normalize the kernel so that the sum of its elements is 1.
    kernel /= kernel.sum()
    return kernel


def defocus_blur(scale, image):
    """
    Applies a defocus blur to the given image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [2, 5, 6, 9, 12][scale]
    # Create the disk-shaped kernel.
    kernel = _create_disk_kernel(factor)
    # Convolve the image with the kernel.
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


def glass_blur(scale, image):
    """
    Applies glass blur effect to the given image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [2, 5, 6, 9, 12][scale]
    # Get the height and width of the image.
    height, width = image.shape[:2]
    # Generate random offsets for each pixel in the image.
    rand_x = np.random.randint(-factor, factor + 1, size=(height, width))
    rand_y = np.random.randint(-factor, factor + 1, size=(height, width))
    # Compute the new coordinates for each pixel after adding the random offsets.
    # Ensure that the new coordinates are within the image boundaries.
    coord_x = np.clip(np.arange(width) + rand_x, 0, width - 1)
    coord_y = np.clip(np.arange(height).reshape(-1, 1) + rand_y, 0, height - 1)
    # Create the glass-blurred image using the new coordinates.
    glass_blurred_image = image[coord_y, coord_x]
    return glass_blurred_image


def _create_motion_blur_kernel(size, angle):
    """
    Create a motion blur kernel of the given size and angle.
    """
    # Create an empty kernel
    kernel = np.zeros((size, size))
    # Convert angle to radian
    angle = np.deg2rad(angle)
    # Calculate the center of the kernel
    center = size // 2
    # Calculate the slope of the line
    slope = np.tan(angle)
    # Fill in the kernel
    for y in range(size):
        x = int(slope * (y - center) + center)
        if 0 <= x < size:
            kernel[y, x] = 1
    # Normalize the kernel
    kernel /= kernel.sum()
    return kernel


def motion_blur(scale, image, size=10, angle=45):
    """
    Apply motion blur to the given image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    size, angle = [(2, 5), (4, 12), (6, 20), (10, 30), (15, 45)][scale]
    # Create the motion blur kernel.
    kernel = _create_motion_blur_kernel(size, angle)
    # Convolve the image with the kernel.
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


def _clipped_zoom(img, zoom_factor):
    h, w = img.shape[:2]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))
    cw = int(np.ceil(w / float(zoom_factor)))
    top = (h - ch) // 2
    right = (w - cw) // 2
    img = scizoom(
        img[top : top + ch, right : right + cw],
        (zoom_factor, zoom_factor, 1),
        order=1,
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_right = (img.shape[1] - w) // 2
    return img[trim_top : trim_top + h, trim_right : trim_right + w]


def zoom_blur(scale, img):
    """
    Applies a zoom blur effect on an image.\n
    This perturbation has an avereage duration of 36ms on an input image of 256*256*3

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    c = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][scale]
    img = (np.array(img) / 255.0).astype(np.float32)
    out = np.zeros_like(img)
    for zoom_factor in c:
        out += _clipped_zoom(img, zoom_factor)
    img = (img + out) / (len(c) + 1)
    return np.clip(img, 0, 1) * 255


def increase_brightness(scale, image):
    """
    Increase the brightness of the image using HSV color space

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [1.1, 1.2, 1.3, 1.5, 1.7][scale]
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Adjust the V channel
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * factor, 0, 255)
    # Convert the image back to RGB color space
    brightened_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return brightened_image


def contrast(scale, img):
    """
    Increase or decrease the conrast of the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [1.1, 1.2, 1.3, 1.5, 1.7][scale]
    pivot = 127.5
    return np.clip(pivot + (img - pivot) * factor, 0, 255)


def elastic(scale, img):
    """
    Applies an elastic deformation on the image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    alpha, sigma = [(2, 0.4), (3, 0.75), (5, 0.9), (7, 1.2), (10, 1.5)][scale]
    # Generate random displacement fields
    dx = np.random.uniform(-1, 1, img.shape[:2]) * alpha
    dy = np.random.uniform(-1, 1, img.shape[:2]) * alpha
    # Smooth the displacement fields
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)
    # Create a meshgrid of indices and apply the displacements
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    # Map the distorted image back using linear interpolation
    distorted_image = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return distorted_image


def pixelate(scale, img):
    """
    Pixelates the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [0.85, 0.75, 0.55, 0.35, 0.2][scale]
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * factor), int(h * factor)), cv2.INTER_AREA)
    return cv2.resize(img, (w, h), cv2.INTER_NEAREST)


def jpeg_filter(scale, image):
    """
    Introduce JPEG compression artifacts to the image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [30, 18, 15, 10, 5][scale]
    # Encode the image as JPEG with the specified quality
    _, jpeg_encoded_image = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), factor]
    )
    # Convert the JPEG encoded bytes to an in-memory binary stream
    jpeg_stream = BytesIO(jpeg_encoded_image.tobytes())
    # Decode the JPEG stream back to an image
    jpeg_artifact_image = cv2.imdecode(
        np.frombuffer(jpeg_stream.read(), np.uint8), cv2.IMREAD_COLOR
    )
    return jpeg_artifact_image


def fog_filter(scale, image):
    """
    Apply a fog effect to the image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    intensity, noise_amount = [
        (0.1, 0.05),
        (0.2, 0.1),
        (0.3, 0.2),
        (0.45, 0.3),
        (0.65, 0.45),
    ][scale]
    # Create a white overlay of the same size as the image
    fog_overlay = np.ones_like(image) * 255
    # Optionally, introduce some noise to the fog overlay
    noise = np.random.normal(scale=noise_amount * 255, size=image.shape).astype(
        np.uint8
    )
    fog_overlay = cv2.addWeighted(fog_overlay, 1 - noise_amount, noise, noise_amount, 0)
    # Blend the fog overlay with the original image
    foggy_image = cv2.addWeighted(image, 1 - intensity, fog_overlay, intensity, 0)
    return foggy_image


def frost_filter(scale, image):
    """
    Apply a frost effect to the image using an overlay image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    intensity = [0.05, 0.15, 0.275, 0.45, 0.6][scale]
    frost_image_path = "./perturbationdrive/OverlayImages/frostImg.png"
    # Load the frost overlay image
    frost_overlay = cv2.imread(frost_image_path, cv2.IMREAD_UNCHANGED)
    assert (
        frost_overlay is not None
    ), "file could not be read, check with os.path.exists()"
    # Resize the frost overlay to match the input image dimensions
    frost_overlay_resized = cv2.resize(frost_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = frost_overlay_resized[:, :, :3]
    alpha = frost_overlay_resized[:, :, 3] / 255.0  # Normalize to [0, 1]
    # Blend the frost overlay with the original image using the alpha channel for transparency
    frosted_image = (1 - (intensity * alpha[:, :, np.newaxis])) * image + (
        intensity * bgr
    )
    frosted_image = np.clip(frosted_image, 0, 255).astype(np.uint8)
    # Decrease saturation to give a cold appearance
    hsv = cv2.cvtColor(frosted_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.8
    frosted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frosted_image


def snow_filter(scale, image):
    """
    Apply a snow effect to the image using an overlay image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    intensity = [0.05, 0.15, 0.275, 0.45, 0.6][scale]
    frost_image_path = "./perturbationdrive/OverlayImages/snow.png"
    # Load the frost overlay image
    frost_overlay = cv2.imread(frost_image_path, cv2.IMREAD_UNCHANGED)
    assert (
        frost_overlay is not None
    ), "file could not be read, check with os.path.exists()"
    # Resize the frost overlay to match the input image dimensions
    frost_overlay_resized = cv2.resize(frost_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = frost_overlay_resized[:, :, :3]
    alpha = frost_overlay_resized[:, :, 3] / 255.0  # Normalize to [0, 1]
    # Blend the frost overlay with the original image using the alpha channel for transparency
    frosted_image = (1 - (intensity * alpha[:, :, np.newaxis])) * image + (
        intensity * bgr
    )
    frosted_image = np.clip(frosted_image, 0, 255).astype(np.uint8)
    # Decrease saturation to give a cold appearance
    hsv = cv2.cvtColor(frosted_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.8
    frosted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frosted_image


def dynamic_snow_filter(scale, image, iterator):
    """
    Apply a dynamic snow effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.05, 0.15, 0.275, 0.45, 0.6][scale]
    # Load the next frame from the iterator
    snow_overlay = next(iterator)
    if (
        snow_overlay.shape[0] != image.shape[0]
        and snow_overlay.shape[1] != image.shape[1]
    ):
        snow_overlay = cv2.resize(snow_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = snow_overlay[:, :, :3]
    mask = snow_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dynamic_rain_filter(scale, image, iterator):
    """
    Apply a dynamic rain effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.05, 0.15, 0.275, 0.45, 0.6][scale]
    rain_overlay = next(iterator)
    # Load the next frame from the iterator
    if (
        rain_overlay.shape[0] != image.shape[0]
        and rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def object_overlay(scale, img1):
    c = [10, 5, 3, 2, 1.5]
    overlay_path = "./perturbationdrive/OverlayImages/Logo_of_the_Technical_University_of_Munichpng.png"
    img2 = cv2.imread(overlay_path)
    assert img2 is not None, "file could not be read, check with os.path.exists()"
    img1_shape0_div_c_scale = int(img1.shape[0] / c[scale])
    img1_shape1_div_2 = int(img1.shape[1] / 2)
    img1_shape0_div_2 = int(img1.shape[0] / 2)

    # Calculate scale factor and target image width directly without extra division
    targetImageWidth = int(
        img1.shape[1] * (img1_shape0_div_c_scale * 100.0 / img2.shape[0]) / 100
    )

    # Resize img2 in a more efficient manner
    img2 = cv2.resize(
        img2,
        (img1_shape0_div_c_scale, targetImageWidth),
        interpolation=cv2.INTER_NEAREST,
    )

    # Precompute reused expressions
    img2_shape0_div_2 = int(img2.shape[0] / 2)
    img2_shape1_div_2 = int(img2.shape[1] / 2)

    # Calculate the start of the ROI
    height_roi = img1_shape0_div_2 - img2_shape0_div_2
    width_roi = img1_shape1_div_2 - img2_shape1_div_2

    rows, cols, _ = img2.shape
    roi = img1[height_roi : height_roi + rows, width_roi : width_roi + cols]

    # Now create a mask of the logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of the logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only the region of the logo from the logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put the logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[height_roi : height_roi + rows, width_roi : width_roi + cols] = dst

    return img1


def dynamic_object_overlay(scale, image, iterator):
    """
    Apply a dynamic bird flying effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.05, 0.15, 0.275, 0.45, 0.6][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        and rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dynamic_frost_filter():
    pass


def dynamic_sun_filter(scale, image, iterator):
    """
    Apply a dynamic sun effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.05, 0.15, 0.275, 0.45, 0.6][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        and rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dynamic_lightning_filter(scale, image, iterator):
    """
    Apply a dynamic lightning effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.05, 0.15, 0.275, 0.45, 0.6][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        and rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dynamic_smoke_filter(scale, image, iterator):
    """
    Apply a dynamic smoke effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.05, 0.15, 0.275, 0.45, 0.6][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        and rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dynamic_hail_filter():
    pass


def perturb_high_attention_regions(
    saliency_map, image, perturbation, boundary=0.5, scale=0
):
    """
    Perturbs the regions of an image where the saliency map has an value greater than boundary

    Parameters:
        - saliency_map (numpy array): Two dimensional saliency map
        - img (numpy array): The input image. Needs to have the same dimensions as the image
        - perturbation func: The perturbation to apply to the image
        - boundary float=0.5: The boundary value above which to perturb the image regions. Needs to be in the range of [0, 1]
        - scale int=0: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    if boundary < 0 or boundary > 1:
        raise ValueError("The boundary value needs to be in the range of [0, 1]")
    # Create a binary mask from the array
    mask = saliency_map > boundary
    # Apply the gaussian noise to the whole image
    noise_img = perturbation(scale, image)
    # Now apply the mask: replace the original image pixels with noisy pixels where mask is True
    image[mask] = noise_img[mask]
    return image
