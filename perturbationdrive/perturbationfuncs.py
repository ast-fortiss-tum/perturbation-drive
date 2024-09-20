import numpy as np
import cv2
from io import BytesIO
from perturbationdrive.AttentionMasks.raindrops_generator.raindrop.dropgenerator import (
    generateDrops,
    generate_label,
)
from perturbationdrive.AttentionMasks.raindrops_generator.raindrop.config import cfg
from .kernels.kernels import (
    diamond_square,
    create_disk_kernel,
    create_motion_blur_kernel,
    clipped_zoom,
)
from .utils.utilFuncs import (
    round_to_nearest_odd,
    scramble_channel,
    simple_white_balance,
)


def empty(scale, img):
    return img


def _check_scale(func):
    """
    A decorator that checks if the scale value is in the range of [0, 4]

    Raises:
        - ValueError: If the scale value is not in the range of [0, 4]
    """

    def wrapper(scale, *args, **kwargs):
        if scale < 0 or scale > 4:
            raise ValueError("The scale value needs to be in the range of [0, 4]")
        return func(scale, *args, **kwargs)

    return wrapper


@_check_scale
def gaussian_noise(scale, img):
    """
    Adds unfirom distributed gausian noise to an image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array
    """
    factor = [0.03, 0.06, 0.12, 0.18, 0.22][scale]
    # scale to a number between 0 and 1
    x = np.array(img, dtype=np.float32) / 255.0
    # add random between 0 and 1
    return (
        np.clip(x + np.random.normal(size=x.shape, scale=factor), 0, 1).astype(
            np.float32
        )
        * 255
    )


@_check_scale
def poisson_noise(scale, img):
    """
    Adds poisson noise to an image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array: Image with salt and pepper noise.
    """
    factor = [120, 105, 87, 55, 30][scale]
    x = np.array(img) / 255.0
    return np.clip(np.random.poisson(x * factor) / float(factor), 0, 1) * 255


@_check_scale
def impulse_noise(scale, img):
    """
    Add salt and pepper noise to an image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array: Image with salt and pepper noise.
    """
    factor = [0.01, 0.02, 0.04, 0.065, 0.10][scale]
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
    kernel = create_disk_kernel(factor)
    # Convolve the image with the kernel.
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


@_check_scale
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


@_check_scale
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
    kernel = create_motion_blur_kernel(size, angle)
    # Convolve the image with the kernel.
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


@_check_scale
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
        np.arange(1, 1.01, 0.01),
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.15, 0.02),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.31, 0.03),
    ][scale]
    img = (np.array(img) / 255.0).astype(np.float32)
    out = np.zeros_like(img)
    for zoom_factor in c:
        out += clipped_zoom(img, zoom_factor)
    img = (img + out) / (len(c) + 1)
    return np.clip(img, 0, 1) * 255


@_check_scale
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


@_check_scale
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


@_check_scale
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


@_check_scale
def pixelate(scale, img):
    """
    Pixelates the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    factor = [0.85, 0.55, 0.35, 0.2, 0.1][scale]
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * factor), int(h * factor)), cv2.INTER_AREA)
    return cv2.resize(img, (w, h), cv2.INTER_NEAREST)


@_check_scale
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


@_check_scale
def shear_image(scale, image):
    """
    Apply horizontal shear to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    shear_factor = [0.12, 0.2, 0.32, 0.45, 0.6][scale]
    # Load the image
    if image is None:
        raise ValueError("Image not found at the given path.")

    rows, cols, _ = image.shape

    # Define the shear matrix
    M = np.array([[1, shear_factor, 0], [0, 1, 0]])

    sheared = cv2.warpAffine(image, M, (cols, rows))

    return sheared


@_check_scale
def translate_image(scale, image):
    """
    Apply translation to an image with different severities in both x and y directions.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    tx, ty = [(-0.1, 0.1), (25, -25), (40, -40), (65, -65), (90, -90)][scale]
    # Load the image
    if image is None:
        raise ValueError("Image not found at the given path.")

    rows, cols, _ = image.shape

    # Define the translation matrix
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

    translated = cv2.warpAffine(image, M, (cols, rows))
    return translated


@_check_scale
def scale_image(scale, image):
    """
    Apply scaling to an image with different severities while maintaining source dimensions.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    scale_factor = [0.96, 0.9, 0.8, 0.68, 0.5][scale]
    rows, cols, _ = image.shape

    # Resize the image
    new_dimensions = (int(cols * scale_factor), int(rows * scale_factor))
    scaled = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

    # If scaled image is smaller, pad it
    if scale_factor < 1:
        top_pad = (rows - scaled.shape[0]) // 2
        bottom_pad = rows - scaled.shape[0] - top_pad
        left_pad = (cols - scaled.shape[1]) // 2
        right_pad = cols - scaled.shape[1] - left_pad
        scalled_image = cv2.copyMakeBorder(
            scaled,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
    # If scaled image is larger, crop it
    else:
        start_row = (scaled.shape[0] - rows) // 2
        start_col = (scaled.shape[1] - cols) // 2
        scalled_image = scaled[
            start_row : start_row + rows, start_col : start_col + cols
        ]

    return scalled_image


@_check_scale
def rotate_image(scale, image):
    """
    Apply rotation to an image with different severities while maintaining source dimensions.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    angle = [10, 20, 45, 90, 180][scale]
    rows, cols, _ = image.shape
    center = (cols / 2, rows / 2)

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # Apply the rotation
    rotated = cv2.warpAffine(image, M, (cols, rows), borderValue=(0, 0, 0))

    return rotated


@_check_scale
def fog_mapping(scale, image):
    """
    Apply fog effect to an image with different severities using Diamond-Square algorithm.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.05, 0.12, 0.22, 0.35, 0.6][scale]
    rows, cols, _ = image.shape
    # Determine size for diamond-square algorithm (closest power of 2 plus 1)
    size = 2 ** int(np.ceil(np.log2(max(rows, cols)))) + 1

    # Generate fog pattern
    fog_pattern = diamond_square(size, 0.6)
    # Resize fog pattern to image size and normalize to [0, 255]
    fog_pattern_resized = cv2.resize(fog_pattern, (cols, rows))
    fog_pattern_resized = (
        (fog_pattern_resized - fog_pattern_resized.min())
        / (fog_pattern_resized.max() - fog_pattern_resized.min())
        * 255
    ).astype(np.uint8)
    fog_pattern_rgb = cv2.merge(
        [fog_pattern_resized, fog_pattern_resized, fog_pattern_resized]
    )  # Convert grayscale to RGB

    # Blend fog with image
    foggy = cv2.addWeighted(image, 1 - severity, fog_pattern_rgb, severity, 0)

    return foggy


@_check_scale
def splatter_mapping(scale, image):
    """
    Apply splatter effect to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0.2, 0.3, 0.4, 0.5][scale]
    rows, cols, _ = image.shape

    # Determine number and size of splatters based on severity
    num_splotches = int(severity * 50)
    max_splotch_size = max(6, int(severity * 50))

    splattered = image.copy()
    for _ in range(num_splotches):
        center_x = np.random.randint(0, cols)
        center_y = np.random.randint(0, rows)
        splotch_size = np.random.randint(5, max_splotch_size)

        # Generate a mask for splotch and apply to the image
        y, x = np.ogrid[-center_y : rows - center_y, -center_x : cols - center_x]
        mask = x * x + y * y <= splotch_size * splotch_size
        splattered[mask] = [0, 0, 0]  # Obscuring the region with black color

    return splattered


@_check_scale
def dotted_lines_mapping(scale, image):
    """
    Apply dotted lines effect to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0.2, 0.3, 0.4, 0.5][scale]

    rows, cols, _ = image.shape

    # Determine parameters based on severity
    num_lines = int(scale * 10)
    distance_between_dots = max(10, int(50 * (1 - severity)))
    dot_thickness = int(severity * 5)

    dotted = image.copy()
    for _ in range(num_lines):
        start_x = np.random.randint(0, cols)
        start_y = np.random.randint(0, rows)
        direction = np.random.rand(2) * 2 - 1  # Random direction
        direction /= np.linalg.norm(direction)  # Normalize to unit vector

        current_x, current_y = start_x, start_y
        while 0 <= current_x < cols and 0 <= current_y < rows:
            cv2.circle(
                dotted, (int(current_x), int(current_y)), dot_thickness, (0, 0, 0), -1
            )  # Draw dot
            current_x += direction[0] * distance_between_dots
            current_y += direction[1] * distance_between_dots

    return dotted


@_check_scale
def zigzag_mapping(scale, image):
    """
    Apply zigzag effect to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0.2, 0.3, 0.4, 0.6][scale]

    rows, cols, _ = image.shape

    # Determine parameters based on severity
    num_lines = int(severity * 10)
    amplitude = int(20 * severity)
    frequency = int(10 * severity)

    zigzag = image.copy()
    for _ in range(num_lines):
        start_x = np.random.randint(0, cols)
        start_y = np.random.randint(0, rows)
        direction = np.random.rand(2) * 2 - 1  # Random direction
        direction /= np.linalg.norm(direction)  # Normalize to unit vector

        current_x, current_y = start_x, start_y
        step = 0
        while 0 <= current_x < cols and 0 <= current_y < rows:
            # Calculate offset for zigzag
            offset = amplitude * np.sin(frequency * step)
            current_x += direction[0]
            current_y += direction[1] + offset
            if 0 <= current_x < cols and 0 <= current_y < rows:
                zigzag[int(current_y), int(current_x)] = [0, 0, 0]  # Draw on image
            step += 1
    return zigzag


@_check_scale
def canny_edges_mapping(scale, image):
    """
    Apply Canny edge detection to an image with different severities.
    The detected edges are highlited and put on top of the input image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.01, 0.1, 0.25, 0.4, 0.7][scale]
    edge_color = (255, 0, 0)

    # Convert the image to grayscale for edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate low and high thresholds based on severity
    low_threshold = int(50 + severity * 100)
    high_threshold = int(150 + severity * 100)

    canny = cv2.Canny(gray_image, low_threshold, high_threshold)
    colored_edges = np.zeros_like(image)

    # Color the detected edges with the specified color
    colored_edges[canny > 0] = edge_color
    # Merge the colored edges with the original image
    merged_image = cv2.addWeighted(image, 0.7, colored_edges, 0.3, 0)

    return merged_image


@_check_scale
def speckle_noise_filter(scale, image):
    """
    Apply speckle noise to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.02, 0.05, 0.09, 0.14, 0.2][scale]

    rows, cols, _ = image.shape
    # Generate noise pattern
    noise = np.random.normal(1, severity, (rows, cols, 3))

    # Apply speckle noise by multiplying original image with noise pattern
    speckled = (image * noise).clip(0, 255).astype(np.uint8)
    return speckled


@_check_scale
def false_color_filter(scale, image):
    """
    Apply false color effect to an image with different severities.
    Severity 1: The Red and Blue channels are swapped.
    Severity 2: The Red and Green channels are swapped.
    Severity 3: The Blue and Green channels are swapped.
    Severity 4: Each channel is inverted.
    Severity 5: Channels are averaged with their adjacent channels.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    false_color = image.copy()

    # Depending on the severity, we swap or mix channels in different ways
    if scale == 0:
        false_color[:, :, 0] = image[:, :, 1]
        false_color[:, :, 1] = image[:, :, 2]
        false_color[:, :, 2] = image[:, :, 0]
    elif scale == 1:
        false_color[:, :, 0] = image[:, :, 1]
        false_color[:, :, 1] = image[:, :, 0]
        false_color[:, :, 2] = image[:, :, 2]
    elif scale == 2:
        false_color[:, :, 0] = image[:, :, 2]
        false_color[:, :, 1] = image[:, :, 1]
        false_color[:, :, 2] = image[:, :, 0]
    elif scale == 3:
        false_color[:, :, 0] = 255 - image[:, :, 0]
        false_color[:, :, 1] = 255 - image[:, :, 1]
        false_color[:, :, 2] = 255 - image[:, :, 2]
    elif scale == 4:
        false_color[:, :, 0] = (image[:, :, 0] + image[:, :, 1]) // 2
        false_color[:, :, 1] = (image[:, :, 1] + image[:, :, 2]) // 2
        false_color[:, :, 2] = (image[:, :, 2] + image[:, :, 0]) // 2

    return false_color


@_check_scale
def high_pass_filter(scale, image):
    """
    Apply high pass filter to an image with different severities.
    This filter highlightes edges and fine details in an image as well
    as darkens the input image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    kernel_size = [35, 59, 83, 107, 113][scale]

    image_float32 = np.float32(image)
    # Blur the image to get the low frequency components
    low_freq = cv2.GaussianBlur(image_float32, (kernel_size, kernel_size), 0)

    high_freq = image_float32 - low_freq
    sharpened = image_float32 + high_freq

    sharpened = np.clip(sharpened, 0, 255).astype("uint8")
    return sharpened


@_check_scale
def low_pass_filter(scale, image):
    """
    Apply low pass filter to an image with different severities while preserving color.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    kernel_size = [15, 23, 30, 36, 40][scale]

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Split the channels
    H, S, V = cv2.split(hsv_image)

    # Apply Gaussian blur to the V channel (value/brightness)
    blurred_V = cv2.GaussianBlur(
        V,
        (
            round_to_nearest_odd(int(kernel_size)),
            round_to_nearest_odd(int(kernel_size)),
        ),
        0,
    )

    # Merge the blurred V channel with the original H and S channels
    merged_hsv = cv2.merge([H, S, blurred_V])

    # Convert back to RGB color space
    low_pass_rgb = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB)

    return low_pass_rgb


@_check_scale
def phase_scrambling(scale, image):
    """
    Apply power scrambling (phase scrambling) to an image with different severities.
    Phase scrambling involves manipulating the phase information of an image's
    Fourier transform while keeping the magnitude intact. This results in an
    image that retains its overall power spectrum but has its content scrambled.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.05, 0.15, 0.26, 0.38, 0.55][scale]
    # Scramble each channel
    R, G, B = cv2.split(image)
    scrambled_R = scramble_channel(R, severity)
    scrambled_G = scramble_channel(G, severity)
    scrambled_B = scramble_channel(B, severity)

    # Merging the scrambled channels
    scrambled_rgb = cv2.merge([scrambled_R, scrambled_G, scrambled_B])

    return scrambled_rgb


@_check_scale
def histogram_equalisation(scale, image):
    """
    Apply histogram equalisation to an image with different severities while
    preserving color.
    We enhance the contrast of an image by effectively spreading out the
    pixel intensities in an image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    clip_limit = [1, 3, 5, 7, 10][scale]
    equalised_images = []

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv_image)

    # Apply CLAHE to the V channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    equalised_V = clahe.apply(V)

    # Merging the equalised V channel with original H and S
    equalised_hsv = cv2.merge([H, S, equalised_V])

    # Convert back to RGB color space
    equalised_rgb = cv2.cvtColor(equalised_hsv, cv2.COLOR_HSV2RGB)

    return equalised_rgb


@_check_scale
def reflection_filter(scale, image):
    """
    Apply a reflection effect to an image with different intensity.
    Creates a mirror effect to the input image and appends the mirrored image to
    the bottom of the image. The intensity refers to the share of the image
    which is appended at the bottom (20%, 30%, 45%, 60% or 90%).

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.2, 0.3, 0.45, 0.6, 0.9][scale]
    # Calculate the portion of the image to reflect
    portion_to_reflect = int(image.shape[0] * severity)

    # Create the reflection by flipping vertically
    reflection = cv2.flip(image[-portion_to_reflect:], 0)

    # Stack the original image and its reflection
    reflected_img = np.vstack((image, reflection[:portion_to_reflect]))

    # Resize the image to maintain original dimensions
    reflected_img = cv2.resize(reflected_img, (image.shape[1], image.shape[0]))

    return reflected_img


@_check_scale
def white_balance_filter(scale, image):
    """
    Apply a white balance effect to an image with different intensities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0.25, 0.5, 0.75, 0.99][scale]
    return cv2.addWeighted(
        image, 1 - severity, simple_white_balance(image.copy()), severity, 0
    )


@_check_scale
def sharpen_filter(scale, image):
    """
    Apply a sharpening effect to an image with different severities using a
    simple sharpen konvolution via a kernel matrix.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [1, 2, 3, 4, 5][scale]
    weight = [0.9, 0.8, 0.7, 0.6, 0.5][scale]

    # Base sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 8 + severity, -1], [-1, -1, -1]])

    # Convolve the image with the sharpening kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    return cv2.addWeighted(image, weight, sharpened, 1 - weight, 0)


@_check_scale
def grayscale_filter(scale, image):
    """
    Apply a grayscale effect to an image with different intensities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    severity = [0.1, 0.2, 0.35, 0.55, 0.85][scale]
    # Convert the image to grayscale
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayscale_img_colored = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)

    # Interpolate between the original and grayscale image based on severity
    grayed_img = cv2.addWeighted(
        image, 1 - severity, grayscale_img_colored, severity, 0
    )

    return grayed_img


@_check_scale
def posterize_filter(scale, image):
    """
    Reduces the number of distinct colors while mainting essential image features

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    scale = [128, 64, 32, 8, 4][scale]

    # Posterize the image
    indices = np.arange(0, 256)
    divider = np.linspace(0, 255, scale + 1)[1]
    quantiz = np.int0(np.linspace(0, 255, scale))
    color_levels = (indices / divider).astype(int) * (255 // (scale - 1))
    color_levels = np.clip(color_levels, 0, 255).astype(int)

    # Apply posterization for each channel
    posterized = np.zeros_like(image)
    for i in range(3):  # For each channel: B, G, R
        posterized[:, :, i] = color_levels[image[:, :, i]]

    return posterized


@_check_scale
def cutout_filter(scale, image):
    """
    Creates random cutouts on the picture and makes the random cutouts black

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    scale = [1, 2, 4, 6, 10][scale]

    h, w, _ = image.shape

    # Apply patches to the image
    for _ in range(scale):
        # Determine patch size
        patch_size_x = np.random.randint(h * 0.05, h * 0.2)
        patch_size_y = np.random.randint(w * 0.05, w * 0.2)

        # Determine top-left corner of the patch
        x = np.random.randint(0, h - patch_size_x)
        y = np.random.randint(0, w - patch_size_y)

        # Apply the patch
        image[x : x + patch_size_x, y : y + patch_size_y, :] = 0  # set to black

    return image


@_check_scale
def sample_pairing_filter(scale, image):
    """
    Randomly sample to regions of the image together

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    alpha = [0.9, 0.7, 0.5, 0.3, 0.1][scale]

    # Randomly select a section of the image
    h, w, _ = image.shape
    start_x = np.random.randint(0, w // 2)
    start_y = np.random.randint(0, h // 2)
    end_x = start_x + w // 2
    end_y = start_y + h // 2

    random_section = image[start_y:end_y, start_x:end_x]

    # Resize the section to the size of the original image
    random_section_resized = cv2.resize(random_section, (w, h))

    # Blend the image and the section
    blended = cv2.addWeighted(image, alpha, random_section_resized, 1 - alpha, 0)

    return blended


@_check_scale
def gaussian_blur(scale, image):
    """
    Applies gaussian blur to the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    kernel_size = [(3, 3), (7, 7), (15, 15), (25, 25), (41, 41)][scale]

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, kernel_size, 0)

    return blurred


@_check_scale
def saturation_filter(scale, image):
    """
    Increases the saturation of the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    multiplier = [1.05, 1.15, 1.4, 1.65, 1.9][scale]

    # Adjust the saturation channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * multiplier, 0, 255)

    # Convert the modified HSV image back to the RGB color space
    saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return saturated


@_check_scale
def saturation_decrease_filter(scale, image):
    """
    Decreases the saturation of the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    multiplier = [0.9, 0.85, 0.6, 0.35, 0.1][scale]

    # Adjust the saturation channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * multiplier, 0, 255)

    # Convert the modified HSV image back to the RGB color space
    saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return saturated


@_check_scale
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


@_check_scale
def frost_filter(scale, image):
    """
    Apply a frost effect to the image using an overlay image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    intensity = [0.15, 0.19, 0.25, 0.32, 0.4][scale]
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


@_check_scale
def snow_filter(scale, image):
    """
    Apply a snow effect to the image using an overlay image.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    intensity = [0.15, 0.22, 0.3, 0.45, 0.6][scale]
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


@_check_scale
def dynamic_snow_filter(scale, image, iterator):
    """
    Apply a dynamic snow effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    snow_overlay = next(iterator)
    snow_overlay = shift_color(snow_overlay, [71, 253, 135], [255, 255, 255])

    if (
        snow_overlay.shape[0] != image.shape[0]
        or snow_overlay.shape[1] != image.shape[1]
    ):
        snow_overlay = cv2.resize(snow_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = snow_overlay[:, :, :3]
    mask = snow_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def static_snow_filter(scale, image, snow_overlay):
    """
    Apply a static snow effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    snow_overlay = shift_color(snow_overlay, [71, 253, 135], [255, 255, 255])

    if (
        snow_overlay.shape[0] != image.shape[0]
        or snow_overlay.shape[1] != image.shape[1]
    ):
        snow_overlay = cv2.resize(snow_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = snow_overlay[:, :, :3]
    mask = snow_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def dynamic_rain_filter(scale, image, iterator):
    """
    Apply a dynamic rain effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = next(iterator)
    rain_overlay = shift_color(rain_overlay, [31, 146, 59], [191, 35, 0])

    # Load the next frame from the iterator
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1.0 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dynamic_raindrop_filter(scale, image, iterator):
    """
    Apply a dynamic rain dropeffect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    overlay = next(iterator)
    overlay = shift_color(overlay, [71, 253, 135], [255, 255, 255])

    # Load the next frame from the iterator
    if overlay.shape[0] != image.shape[0] or overlay.shape[1] != image.shape[1]:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = overlay[:, :, :3]
    mask = overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1.0 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def static_rain_filter(scale, image, rain_overlay):
    """
    Apply a dynamic rain effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = shift_color(rain_overlay, [31, 146, 59], [191, 35, 0])

    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1.0 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def object_overlay(scale, img1):
    """
    Apply an overlay effect to the image using an overlay image read from
    the file system. The overlay image is placed at the center of the input image.
    The file resides at the path "./perturbationdrive/OverlayImages/Logo_of_the_Technical_University_of_Munichpng.png"

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
    """
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


@_check_scale
def dynamic_object_overlay(scale, image, iterator):
    """
    Apply a dynamic bird flying effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    rain_overlay = shift_color(rain_overlay, [175, 221, 202], [0, 0, 0])

    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def static_object_overlay(scale, image, rain_overlay):
    """
    Apply a dynamic bird flying effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = shift_color(rain_overlay, [175, 221, 202], [0, 0, 0])

    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def dynamic_sun_filter(scale, image, iterator):
    """
    Apply a dynamic sun effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    rain_overlay = shift_color(rain_overlay, [223, 234, 212], [28, 202, 255])

    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def static_sun_filter(scale, image, rain_overlay):
    """
    Apply a dynamic sun effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    rain_overlay = shift_color(rain_overlay, [223, 234, 212], [28, 202, 255])

    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def dynamic_lightning_filter(scale, image, iterator):
    """
    Apply a dynamic lightning effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    rain_overlay = shift_color(rain_overlay, [5, 122, 101], [8, 152, 188])

    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def static_lightning_filter(scale, image, rain_overlay):
    """
    Apply a dynamic lightning effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = shift_color(rain_overlay, [5, 122, 101], [8, 152, 188])

    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def dynamic_smoke_filter(scale, image, iterator):
    """
    Apply a dynamic smoke effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
    rain_overlay = shift_color(rain_overlay, [30, 112, 65], [132, 132, 132])

    # Resize the frost overlay to match the input image dimensions
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def static_smoke_filter(scale, image, rain_overlay):
    """
    Apply a dynamic smoke effect to the image using an overlay image iterator.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4
        - iterator cycle: Cyclic iterator over all the frames of the dynamic overlay mask

    Returns: numpy array:
    """
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = shift_color(rain_overlay, [30, 112, 65], [132, 132, 132])
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1 - intensity) * image[mask] + intensity * bgr[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@_check_scale
def perturb_high_attention_regions(
    saliency_map, image, perturbation, boundary=0.5, scale=0
):
    """
    Perturbs the regions of an image where the saliency map has an value greater than boundary
    Can be used with either vanilla saliency map or grad-cam map

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


@_check_scale
def perturb_highest_n_attention_regions(
    saliency_map, image, perturbation, n=30, scale=0
):
    """
    Perturbs the highest n% of the regions of an image where the saliency map has an value greater than threshold
    """
    if n < 0 or n > 100:
        raise ValueError("The threshold value needs to be in the range of [0, 100]")
    # Create a binary mask from the array
    mask = saliency_map > np.percentile(saliency_map, n)
    # Apply the gaussian noise to the whole image
    noise_img = perturbation(scale, image)
    # Now apply the mask: replace the original image pixels with noisy pixels where mask is True
    image[mask] = noise_img[mask]
    return image


@_check_scale
def perturb_lowest_n_attention_regions(
    saliency_map, image, perturbation, n=30, scale=0
):
    """
    Perturbs the lowest n% of the regions of an image where the saliency map has an value greater than threshold

    Parameters:
        - saliency_map (numpy array): Two dimensional saliency map
        - img (numpy array): The input image. Needs to have the same dimensions as the image
        - perturbation func: The perturbation to apply to the image
        - n int=30: The percentage of the lowest saliency regions to perturb
        - scale int=0: The severity of the perturbation on a scale from 0 to 4
    """
    if n < 0 or n > 100:
        raise ValueError("The threshold value needs to be in the range of [0, 100]")
    # Create a binary mask from the array
    thres = np.percentile(saliency_map, n)
    if thres == 0:
        mask = saliency_map <= thres
    else:
        mask = saliency_map < thres
    # Apply the gaussian noise to the whole image
    noise_img = perturbation(scale, image)
    # Now apply the mask: replace the original image pixels with noisy pixels where mask is True
    image[mask] = noise_img[mask]
    return image


@_check_scale
def perturb_random_n_attention_regions(
    saliency_map, image, perturbation, n=30, scale=0
):
    """
    Perturbs n% of the regions of an image where the saliency map has an value greater than threshold
    """
    if n < 0 or n > 100:
        raise ValueError("The n value needs to be in the range of [0, 100]")
    # Create a binary mask from the array
    mask = np.random.choice(
        [True, False], size=saliency_map.shape, p=[n / 100, 1 - n / 100]
    )
    # Apply the gaussian noise to the whole image
    noise_img = perturbation(scale, image)
    # Now apply the mask: replace the original image pixels with noisy pixels where mask is True
    image[mask] = noise_img[mask]
    return image


def clamp_values(tuples_list, min1, max1, min2, max2):
    """
    Adjusts the values in each tuple to be within the specified range.

    :param tuples_list: List of tuples to adjust
    :param min1: Minimum limit for the first element of the tuple
    :param max1: Maximum limit for the first element of the tuple
    :param min2: Minimum limit for the second element of the tuple
    :param max2: Maximum limit for the second element of the tuple
    :return: List of tuples with values adjusted to be within the specified range
    """
    clamped_list = []
    for t in tuples_list:
        # Clamp the first value
        val1 = max(min(t[0], max1), min1)
        # Clamp the second value
        val2 = max(min(t[1], max2), min2)
        # Add the clamped tuple to the new list
        clamped_list.append((val1, val2))
    return clamped_list


def effects_attention_regions(saliency_map, scale, image, type):
    mask = saliency_map > np.percentile(saliency_map, 90)
    coordinates = np.argwhere(mask)
    selected_coords = coordinates[
        np.random.choice(coordinates.shape[0], scale + 1, replace=False)
    ]
    selected_coords_tuples = [tuple(row) for row in selected_coords]
    selected_coords_tuples = clamp_values(
        selected_coords_tuples, 5, image.shape[1] - 5, 5, image.shape[0] - 5
    )
    List_of_Drops, _, _ = generate_label(
        image.shape[1], image.shape[0], selected_coords_tuples, cfg
    )
    output_image = generateDrops(image, cfg, List_of_Drops)
    return output_image


def shift_color(image, source_color, target_color):
    # Check if the image has an alpha channel
    has_alpha = image.shape[2] == 4

    if has_alpha:
        # Split the image into BGR and alpha channels
        bgr, alpha = image[:, :, :3], image[:, :, 3]
    else:
        bgr = image

    # Convert source and target colors to numpy arrays
    source_color = np.array(source_color, dtype=np.int16)
    target_color = np.array(target_color, dtype=np.int16)

    # Calculate the difference
    color_diff = target_color - source_color

    # Apply the difference to each pixel in the BGR channels
    shifted_bgr = np.clip(bgr.astype(np.int16) + color_diff, 0, 255).astype(np.uint8)

    if has_alpha:
        # Recombine the shifted BGR channels with the untouched alpha channel
        shifted_image = cv2.merge((shifted_bgr, alpha))
    else:
        shifted_image = shifted_bgr

    return shifted_image
