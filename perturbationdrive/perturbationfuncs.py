import numpy as np
import cv2
from io import BytesIO
from .kernels.kernels import (
    diamond_square,
    create_disk_kernel,
    create_motion_blur_kernel,
    clipped_zoom,
)
from .utils.utilFuncs import (
    round_to_nearest_odd,
    scramble_channel,
    equalise_power,
    simple_white_balance,
)


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
        out += clipped_zoom(img, zoom_factor)
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


def shear_image(scale, image):
    """
    Apply horizontal shear to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    shear_factor = [-0.3, -0.15, 0.01, 0.15, 0.3][scale]
    # Load the image
    if image is None:
        raise ValueError("Image not found at the given path.")

    rows, cols, _ = image.shape

    # Define the shear matrix
    M = np.array([[1, shear_factor, 0], [0, 1, 0]])

    sheared = cv2.warpAffine(image, M, (cols, rows))

    return sheared


def translate_image(scale, image):
    """
    Apply translation to an image with different severities in both x and y directions.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    tx, ty = [(-50, 50), (-25, 25), (-0.01, 0.01), (25, -25), (50, -50)][scale]
    # Load the image
    if image is None:
        raise ValueError("Image not found at the given path.")

    rows, cols, _ = image.shape

    # Define the translation matrix
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

    translated = cv2.warpAffine(image, M, (cols, rows))
    return translated


def scale_image(scale, image):
    """
    Apply scaling to an image with different severities while maintaining source dimensions.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    scale_factor = [0.5, 0.75, 0.99, 1.25, 2][scale]
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


def rotate_image(scale, image):
    """
    Apply rotation to an image with different severities while maintaining source dimensions.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    angle = [-45, -20, 0.01, 20, 45][scale]
    rows, cols, _ = image.shape
    center = (cols / 2, rows / 2)

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # Apply the rotation
    rotated = cv2.warpAffine(image, M, (cols, rows), borderValue=(0, 0, 0))

    return rotated


def stripe_mapping(scale, image):
    """
    Apply Stripe mapping to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    width = [10, 20, 30, 40, 50][scale]
    rows, cols, _ = image.shape

    # Clone the original image
    striped = image.copy()

    # Define stripe boundaries
    start_col = (cols - width) // 2
    end_col = start_col + width

    # Invert the pixel values in the stripe
    striped[:, start_col:end_col] = 255 - striped[:, start_col:end_col]

    return striped


def fog_mapping(scale, image):
    """
    Apply fog effect to an image with different severities using Diamond-Square algorithm.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0, 2, 0.3, 0.4, 0.5][scale]
    rows, cols, _ = image.shape
    # Determine size for diamond-square algorithm (closest power of 2 plus 1)
    size = 2 ** int(np.ceil(np.log2(max(rows, cols)))) + 1

    # Generate fog pattern
    fog_pattern = diamond_square(size, severity)
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


def zigzag_mapping(scale, image):
    """
    Apply zigzag effect to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0.2, 0.3, 0.4, 0.5][scale]

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


def canny_edges_mapping(scale, image):
    """
    Apply Canny edge detection to an image with different severities.
    The detected edges are highlited and put on top of the input image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0.2, 0.3, 0.4, 0.5][scale]
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


def speckle_noise_filter(scale, image):
    """
    Apply speckle noise to an image with different severities.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [0.1, 0.2, 0.3, 0.4, 0.5][scale]

    rows, cols, _ = image.shape
    # Generate noise pattern
    noise = np.random.normal(1, severity, (rows, cols, 3))

    # Apply speckle noise by multiplying original image with noise pattern
    speckled = (image * noise).clip(0, 255).astype(np.uint8)
    return speckled


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
    kernel_size = [233, 75, 63, 49, 15][scale]
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Split the channels
    H, S, V = cv2.split(hsv_image)

    # Apply Gaussian blur to the V channel (value/brightness)
    blurred_V = cv2.GaussianBlur(V, (int(kernel_size), int(kernel_size)), 0)

    # Subtract the blurred V channel from the original to get the high-pass filtered result
    high_pass_V = cv2.subtract(V, blurred_V)

    # Merge the high-pass filtered V channel with the original H and S channels
    merged_hsv = cv2.merge([H, S, high_pass_V])

    # Convert back to RGB color space
    high_pass_rgb = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB)

    return high_pass_rgb


def low_pass_filter(scale, image):
    """
    Apply low pass filter to an image with different severities while preserving color.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    kernel_size = [15, 40, 63, 75, 113][scale]

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


def power_equalisation(scale, image):
    """
    Apply power equalisation to an image with different severities while
    preserving color.
    We equalize or modify the energy distribution of the image across different frequencies.
    By adjustingthe magnitude of the Fourier transform (which represents the frequency
    content of the image), we can change how the energy (or power) is spread
    across frequencies.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    alpha = [1.15, 1.08, 1.02, 0.98, 0.92][scale]
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv_image)
    # Equalise each channel
    equalised_V = equalise_power(V, alpha)

    # Merging the equalised channel with original H and S
    equalised_hsv = cv2.merge([H, S, equalised_V])

    # Convert back to RGB color space
    equalised_rgb = cv2.cvtColor(equalised_hsv, cv2.COLOR_HSV2RGB)
    return equalised_rgb


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


def sharpen_filter(scale, image):
    """
    Apply a sharpening effect to an image with different severities using a
    simple sharpen konvolution via a kernel matrix.

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    severity = [1.15, 1.2, 1.25, 1.5, 2.0][scale]

    # Base sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel[1, 1] = 8 * severity

    # Convolve the image with the sharpening kernel
    return cv2.filter2D(image, -1, kernel)


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


def silhouette_filter(scale, image):
    """
    Applies a silhouette filter using canny edge detection to highlight edges and
    return a gray scale image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    thresholds = [(10, 60), (20, 80), (30, 100), (40, 120), (50, 150)]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lower_threshold, upper_threshold = thresholds[scale]

    # Apply Canny edge detection
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)

    # Invert the binary output to get the silhouette
    silhouette = cv2.bitwise_not(edges)
    silhouette = cv2.cvtColor(silhouette, cv2.COLOR_GRAY2BGR)

    return silhouette


def invert_filter(scale, image):
    """
    Applies a invert filter, inverting each color channel seperately

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    inverted = cv2.bitwise_not(image)
    original_weight, inverted_weight = [
        (0.9, 0.1),
        (0.7, 0.3),
        (0.4, 0.6),
        (0.3, 0.7),
        (0.0, 1.0),
    ][scale]
    blended = cv2.addWeighted(image, original_weight, inverted, inverted_weight, 0)

    return blended


def solarite_filter(scale, image):
    """
    Inverts the tonesnof the image pixels which are above a certain threshold

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """
    threshold = [230, 200, 170, 140, 110][scale]

    solarized = np.where(image > threshold, 255 - image, image)

    return solarized


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


def gaussian_blur(scale, image):
    """
    Applies gaussian blur to the image

    Parameters:
        - img (numpy array): The input image.
        - scale int: The severity of the perturbation on a scale from 0 to 4

    Returns: numpy array:
    """

    kernel_size = [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)][scale]

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, kernel_size, 0)

    return blurred


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
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
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
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
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
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    snow_overlay = next(iterator)
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
    # Load the next frame from the iterator
    if (
        rain_overlay.shape[0] != image.shape[0]
        or rain_overlay.shape[1] != image.shape[1]
    ):
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))
    cv2.imwrite("output_image.jpg", image)
    cv2.imwrite("output_image2.jpg", rain_overlay)
    # Extract the 3 channels (BGR) and the alpha (transparency) channel
    bgr = rain_overlay[:, :, :3]
    mask = rain_overlay[:, :, 3] != 0
    # mash the mask areas together
    image[mask] = (1.0 - intensity) * image[mask] + intensity * bgr[mask]
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
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    # Load the next frame from the iterator
    rain_overlay = next(iterator)
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
