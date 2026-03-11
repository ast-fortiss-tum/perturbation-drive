import numpy as np
import cv2
from io import BytesIO
from perturbationdrive.AttentionMasks.raindrops_generator.raindrop.dropgenerator import generateDrops, generate_label
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
    equalise_power,
    simple_white_balance,
    clamp_values,
)


def _to_uint8_image(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def _as_float01(x):
    return np.asarray(x, dtype=np.float32) / 255.0


def gaussian_noise(scale, img):
    """
    Adds uniform gaussian noise to an image.
    """
    factor = [0.03, 0.06, 0.12, 0.18, 0.22][scale]
    x = _as_float01(img)
    noisy = np.clip(
        x + np.random.normal(loc=0.0, scale=factor, size=x.shape).astype(np.float32),
        0,
        1,
    )
    return _to_uint8_image(noisy * 255.0)


def poisson_noise(scale, img):
    """
    Adds poisson noise to an image.
    """
    factor = [120, 105, 87, 55, 30][scale]
    x = _as_float01(img)
    noisy = np.random.poisson(x * factor) / float(factor)
    noisy = np.clip(noisy, 0, 1)
    return _to_uint8_image(noisy * 255.0)


def impulse_noise(scale, img):
    """
    Add salt and pepper noise to an image.
    """
    img = _to_uint8_image(np.asarray(img).copy())
    factor = [0.01, 0.02, 0.04, 0.065, 0.10][scale]

    num_salt = int(np.ceil(factor * img.size * 0.5))
    coords = [np.random.randint(0, i, num_salt) for i in img.shape]
    img[tuple(coords)] = 255

    num_pepper = int(np.ceil(factor * img.size * 0.5))
    coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
    img[tuple(coords)] = 0

    return img


def defocus_blur(scale, image):
    factor = [2, 5, 6, 9, 12][scale]
    image = _to_uint8_image(image)
    kernel = create_disk_kernel(factor)
    return cv2.filter2D(image, -1, kernel)


def glass_blur(scale, image):
    factor = [2, 5, 6, 9, 12][scale]
    image = _to_uint8_image(image)
    height, width = image.shape[:2]

    rand_x = np.random.randint(-factor, factor + 1, size=(height, width))
    rand_y = np.random.randint(-factor, factor + 1, size=(height, width))

    coord_x = np.clip(np.arange(width) + rand_x, 0, width - 1)
    coord_y = np.clip(np.arange(height).reshape(-1, 1) + rand_y, 0, height - 1)

    return image[coord_y, coord_x]


def motion_blur(scale, image, size=10, angle=45):
    size, angle = [(2, 5), (4, 12), (6, 20), (10, 30), (15, 45)][scale]
    image = _to_uint8_image(image)
    kernel = create_motion_blur_kernel(size, angle)
    return cv2.filter2D(image, -1, kernel)


def zoom_blur(scale, img):
    c = [
        np.arange(1, 1.01, 0.01),
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.15, 0.02),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.31, 0.03),
    ][scale]

    img = _as_float01(img)
    out = np.zeros_like(img, dtype=np.float32)

    for zoom_factor in c:
        out += clipped_zoom(img, zoom_factor)

    img = (img + out) / (len(c) + 1)
    return _to_uint8_image(np.clip(img, 0, 1) * 255.0)


def increase_brightness(scale, image):
    factor = [1.1, 1.2, 1.3, 1.5, 1.7][scale]
    image = _to_uint8_image(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = np.clip(
        hsv_image[:, :, 2].astype(np.float32) * factor, 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def contrast(scale, img):
    factor = [1.1, 1.2, 1.3, 1.5, 1.7][scale]
    pivot = 127.5
    x = np.asarray(img, dtype=np.float32)
    out = pivot + (x - pivot) * factor
    return _to_uint8_image(out)


def elastic(scale, img):
    img = _to_uint8_image(img)
    alpha, sigma = [(2, 0.4), (3, 0.75), (5, 0.9), (7, 1.2), (10, 1.5)][scale]

    dx = np.random.uniform(-1, 1, img.shape[:2]).astype(np.float32) * alpha
    dy = np.random.uniform(-1, 1, img.shape[:2]).astype(np.float32) * alpha

    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)

    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    map_x = (x.astype(np.float32) + dx).astype(np.float32)
    map_y = (y.astype(np.float32) + dy).astype(np.float32)

    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def pixelate(scale, img):
    img = _to_uint8_image(img)
    factor = [0.85, 0.55, 0.35, 0.2, 0.1][scale]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))), cv2.INTER_AREA)
    return cv2.resize(small, (w, h), cv2.INTER_NEAREST)


def jpeg_filter(scale, image):
    factor = [30, 18, 15, 10, 5][scale]
    image = _to_uint8_image(image)
    _, jpeg_encoded_image = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), factor]
    )
    jpeg_stream = BytesIO(jpeg_encoded_image.tobytes())
    jpeg_artifact_image = cv2.imdecode(
        np.frombuffer(jpeg_stream.read(), np.uint8), cv2.IMREAD_COLOR
    )
    return jpeg_artifact_image


def shear_image(scale, image):
    shear_factor = [0.12, 0.2, 0.32, 0.45, 0.6][scale]
    image = _to_uint8_image(image)
    if image is None:
        raise ValueError("Image not found at the given path.")

    rows, cols, _ = image.shape
    M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(image, M, (cols, rows))


def translate_image(scale, image):
    tx, ty = [(-0.1, 0.1), (25, -25), (40, -40), (65, -65), (90, -90)][scale]
    image = _to_uint8_image(image)
    if image is None:
        raise ValueError("Image not found at the given path.")

    rows, cols, _ = image.shape
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    return cv2.warpAffine(image, M, (cols, rows))


def scale_image(scale, image):
    image = _to_uint8_image(image)
    scale_factor = [0.96, 0.9, 0.8, 0.68, 0.5][scale]
    rows, cols, _ = image.shape

    new_dimensions = (max(1, int(cols * scale_factor)), max(1, int(rows * scale_factor)))
    scaled = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

    if scale_factor < 1:
        top_pad = (rows - scaled.shape[0]) // 2
        bottom_pad = rows - scaled.shape[0] - top_pad
        left_pad = (cols - scaled.shape[1]) // 2
        right_pad = cols - scaled.shape[1] - left_pad
        scaled_image = cv2.copyMakeBorder(
            scaled,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
    else:
        start_row = (scaled.shape[0] - rows) // 2
        start_col = (scaled.shape[1] - cols) // 2
        scaled_image = scaled[start_row:start_row + rows, start_col:start_col + cols]

    return scaled_image


def rotate_image(scale, image):
    image = _to_uint8_image(image)
    angle = [10, 20, 45, 90, 180][scale]
    rows, cols, _ = image.shape
    center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, M, (cols, rows), borderValue=(0, 0, 0))


def fog_mapping(scale, image):
    image = _to_uint8_image(image)
    severity = [0.05, 0.12, 0.22, 0.35, 0.6][scale]
    rows, cols, _ = image.shape
    size = 2 ** int(np.ceil(np.log2(max(rows, cols)))) + 1

    fog_pattern = diamond_square(size, 0.6)
    fog_pattern_resized = cv2.resize(fog_pattern, (cols, rows))
    fog_pattern_resized = (
        (fog_pattern_resized - fog_pattern_resized.min())
        / max((fog_pattern_resized.max() - fog_pattern_resized.min()), 1e-8)
        * 255
    ).astype(np.uint8)

    fog_pattern_rgb = cv2.merge(
        [fog_pattern_resized, fog_pattern_resized, fog_pattern_resized]
    )

    return cv2.addWeighted(image, 1 - severity, fog_pattern_rgb, severity, 0)


def splatter_mapping(scale, image):
    image = _to_uint8_image(image)
    severity = [0.1, 0.2, 0.3, 0.4, 0.5][scale]
    rows, cols, _ = image.shape

    num_splotches = int(severity * 50)
    max_splotch_size = max(6, int(severity * 50))

    splattered = image.copy()
    for _ in range(num_splotches):
        center_x = np.random.randint(0, cols)
        center_y = np.random.randint(0, rows)
        splotch_size = np.random.randint(5, max_splotch_size)

        y, x = np.ogrid[-center_y: rows - center_y, -center_x: cols - center_x]
        mask = x * x + y * y <= splotch_size * splotch_size
        splattered[mask] = [0, 0, 0]

    return splattered


def dotted_lines_mapping(scale, image):
    image = _to_uint8_image(image)
    severity = [0.1, 0.2, 0.3, 0.4, 0.5][scale]
    rows, cols, _ = image.shape

    num_lines = int((scale + 1) * 10)
    distance_between_dots = max(10, int(50 * (1 - severity)))
    dot_thickness = max(1, int(severity * 5))

    dotted = image.copy()
    for _ in range(num_lines):
        start_x = np.random.randint(0, cols)
        start_y = np.random.randint(0, rows)
        direction = np.random.rand(2) * 2 - 1
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction /= norm

        current_x, current_y = float(start_x), float(start_y)
        while 0 <= current_x < cols and 0 <= current_y < rows:
            cv2.circle(
                dotted, (int(current_x), int(current_y)), dot_thickness, (0, 0, 0), -1
            )
            current_x += direction[0] * distance_between_dots
            current_y += direction[1] * distance_between_dots

    return dotted


def zigzag_mapping(scale, image):
    image = _to_uint8_image(image)
    severity = [0.1, 0.2, 0.3, 0.4, 0.6][scale]
    rows, cols, _ = image.shape

    num_lines = int(max(1, severity * 10))
    amplitude = int(20 * severity)
    frequency = max(1, int(10 * severity))

    zigzag = image.copy()
    for _ in range(num_lines):
        start_x = np.random.randint(0, cols)
        start_y = np.random.randint(0, rows)
        direction = np.random.rand(2) * 2 - 1
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction /= norm

        current_x, current_y = float(start_x), float(start_y)
        step = 0
        while 0 <= current_x < cols and 0 <= current_y < rows:
            offset = amplitude * np.sin(frequency * step)
            current_x += direction[0]
            current_y += direction[1] + offset
            if 0 <= current_x < cols and 0 <= current_y < rows:
                zigzag[int(current_y), int(current_x)] = [0, 0, 0]
            step += 1

    return zigzag


def canny_edges_mapping(scale, image):
    image = _to_uint8_image(image)
    severity = [0.01, 0.1, 0.25, 0.4, 0.7][scale]
    edge_color = (255, 0, 0)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    low_threshold = int(50 + severity * 100)
    high_threshold = int(150 + severity * 100)

    canny = cv2.Canny(gray_image, low_threshold, high_threshold)
    colored_edges = np.zeros_like(image)
    colored_edges[canny > 0] = edge_color

    return cv2.addWeighted(image, 0.7, colored_edges, 0.3, 0)


def speckle_noise_filter(scale, image):
    image = _to_uint8_image(image)
    severity = [0.02, 0.05, 0.09, 0.14, 0.2][scale]
    rows, cols, _ = image.shape
    noise = np.random.normal(1, severity, (rows, cols, 3)).astype(np.float32)
    speckled = image.astype(np.float32) * noise
    return _to_uint8_image(speckled)


def false_color_filter(scale, image):
    image = _to_uint8_image(image)
    false_color = image.copy()

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
        false_color[:, :, 0] = ((image[:, :, 0].astype(np.uint16) + image[:, :, 1].astype(np.uint16)) // 2).astype(np.uint8)
        false_color[:, :, 1] = ((image[:, :, 1].astype(np.uint16) + image[:, :, 2].astype(np.uint16)) // 2).astype(np.uint8)
        false_color[:, :, 2] = ((image[:, :, 2].astype(np.uint16) + image[:, :, 0].astype(np.uint16)) // 2).astype(np.uint8)

    return false_color


def high_pass_filter(scale, image):
    image = _to_uint8_image(image)
    kernel_size = [35, 59, 83, 107, 113][scale]

    image_float32 = np.float32(image)
    low_freq = cv2.GaussianBlur(image_float32, (kernel_size, kernel_size), 0)
    high_freq = image_float32 - low_freq
    sharpened = image_float32 + high_freq

    return _to_uint8_image(sharpened)


def low_pass_filter(scale, image):
    image = _to_uint8_image(image)
    kernel_size = [15, 23, 30, 36, 40][scale]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv_image)

    blurred_V = cv2.GaussianBlur(
        V,
        (
            round_to_nearest_odd(int(kernel_size)),
            round_to_nearest_odd(int(kernel_size)),
        ),
        0,
    )

    merged_hsv = cv2.merge([H, S, blurred_V])
    return cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB)


def phase_scrambling(scale, image):
    image = _to_uint8_image(image)
    severity = [0.05, 0.15, 0.26, 0.38, 0.55][scale]
    R, G, B = cv2.split(image)
    scrambled_R = scramble_channel(R, severity)
    scrambled_G = scramble_channel(G, severity)
    scrambled_B = scramble_channel(B, severity)
    return cv2.merge([scrambled_R, scrambled_G, scrambled_B])


def histogram_equalisation(scale, image):
    image = _to_uint8_image(image)
    clip_limit = [1, 3, 5, 7, 10][scale]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv_image)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    equalised_V = clahe.apply(V)

    equalised_hsv = cv2.merge([H, S, equalised_V])
    return cv2.cvtColor(equalised_hsv, cv2.COLOR_HSV2RGB)


def reflection_filter(scale, image):
    image = _to_uint8_image(image)
    severity = [0.2, 0.3, 0.45, 0.6, 0.9][scale]
    portion_to_reflect = max(1, int(image.shape[0] * severity))

    reflection = cv2.flip(image[-portion_to_reflect:], 0)
    reflected_img = np.vstack((image, reflection[:portion_to_reflect]))
    reflected_img = cv2.resize(reflected_img, (image.shape[1], image.shape[0]))

    return reflected_img


def white_balance_filter(scale, image):
    image = _to_uint8_image(image)
    severity = [0.1, 0.25, 0.5, 0.75, 0.99][scale]
    return cv2.addWeighted(
        image, 1 - severity, _to_uint8_image(simple_white_balance(image.copy())), severity, 0
    )


def sharpen_filter(scale, image):
    image = _to_uint8_image(image)
    severity = [1, 2, 3, 4, 5][scale]
    weight = [0.9, 0.8, 0.7, 0.6, 0.5][scale]

    kernel = np.array([[-1, -1, -1], [-1, 8 + severity, -1], [-1, -1, -1]], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    return cv2.addWeighted(image, weight, sharpened, 1 - weight, 0)


def grayscale_filter(scale, image):
    image = _to_uint8_image(image)
    severity = [0.1, 0.2, 0.35, 0.55, 0.85][scale]
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayscale_img_colored = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(image, 1 - severity, grayscale_img_colored, severity, 0)


def posterize_filter(scale, image):
    image = _to_uint8_image(image)
    levels = [128, 64, 32, 8, 4][scale]

    indices = np.arange(256)
    divider = np.linspace(0, 255, levels + 1)[1]
    color_levels = (indices / divider).astype(int) * (255 // max(levels - 1, 1))
    color_levels = np.clip(color_levels, 0, 255).astype(np.uint8)

    posterized = np.zeros_like(image)
    for i in range(3):
        posterized[:, :, i] = color_levels[image[:, :, i]]

    return posterized


def cutout_filter(scale, image):
    image = _to_uint8_image(np.asarray(image).copy())
    num_patches = [1, 2, 4, 6, 10][scale]

    h, w, _ = image.shape

    for _ in range(num_patches):
        patch_size_x = np.random.randint(max(1, int(h * 0.05)), max(2, int(h * 0.2)))
        patch_size_y = np.random.randint(max(1, int(w * 0.05)), max(2, int(w * 0.2)))

        x = np.random.randint(0, max(1, h - patch_size_x))
        y = np.random.randint(0, max(1, w - patch_size_y))

        image[x:x + patch_size_x, y:y + patch_size_y, :] = 0

    return image


def sample_pairing_filter(scale, image):
    image = _to_uint8_image(image)
    alpha = [0.9, 0.7, 0.5, 0.3, 0.1][scale]

    h, w, _ = image.shape
    start_x = np.random.randint(0, max(1, w // 2))
    start_y = np.random.randint(0, max(1, h // 2))
    end_x = start_x + max(1, w // 2)
    end_y = start_y + max(1, h // 2)

    random_section = image[start_y:end_y, start_x:end_x]
    random_section_resized = cv2.resize(random_section, (w, h))

    return cv2.addWeighted(image, alpha, random_section_resized, 1 - alpha, 0)


def gaussian_blur(scale, image):
    image = _to_uint8_image(image)
    kernel_size = [(3, 3), (7, 7), (15, 15), (25, 25), (41, 41)][scale]
    return cv2.GaussianBlur(image, kernel_size, 0)


def saturation_filter(scale, image):
    image = _to_uint8_image(image)
    multiplier = [1.05, 1.15, 1.4, 1.65, 1.9][scale]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32) * multiplier, 0, 255
    ).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def saturation_decrease_filter(scale, image):
    image = _to_uint8_image(image)
    multiplier = [0.9, 0.85, 0.6, 0.35, 0.1][scale]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32) * multiplier, 0, 255
    ).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def fog_filter(scale, image):
    image = _to_uint8_image(image)
    intensity, noise_amount = [
        (0.1, 0.05),
        (0.2, 0.1),
        (0.3, 0.2),
        (0.45, 0.3),
        (0.65, 0.45),
    ][scale]

    fog_overlay = np.ones_like(image, dtype=np.uint8) * 255
    noise = np.random.normal(loc=0.0, scale=noise_amount * 255, size=image.shape)
    noise = _to_uint8_image(noise)
    fog_overlay = cv2.addWeighted(fog_overlay, 1 - noise_amount, noise, noise_amount, 0)

    return cv2.addWeighted(image, 1 - intensity, fog_overlay, intensity, 0)


def frost_filter(scale, image):
    image = _to_uint8_image(image)
    intensity = [0.15, 0.19, 0.25, 0.32, 0.4][scale]
    frost_image_path = "./perturbationdrive/OverlayImages/frostImg.png"

    frost_overlay = cv2.imread(frost_image_path, cv2.IMREAD_UNCHANGED)
    assert frost_overlay is not None, "file could not be read, check with os.path.exists()"

    frost_overlay_resized = cv2.resize(frost_overlay, (image.shape[1], image.shape[0]))
    bgr = frost_overlay_resized[:, :, :3].astype(np.float32)
    alpha = frost_overlay_resized[:, :, 3].astype(np.float32) / 255.0

    frosted_image = (1 - (intensity * alpha[:, :, np.newaxis])) * image.astype(np.float32) + (intensity * bgr)
    frosted_image = _to_uint8_image(frosted_image)

    hsv = cv2.cvtColor(frosted_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def snow_filter(scale, image):
    image = _to_uint8_image(image)
    intensity = [0.15, 0.22, 0.3, 0.45, 0.6][scale]
    snow_image_path = "./perturbationdrive/OverlayImages/snow.png"

    snow_overlay = cv2.imread(snow_image_path, cv2.IMREAD_UNCHANGED)
    assert snow_overlay is not None, "file could not be read, check with os.path.exists()"

    snow_overlay_resized = cv2.resize(snow_overlay, (image.shape[1], image.shape[0]))
    bgr = snow_overlay_resized[:, :, :3].astype(np.float32)
    alpha = snow_overlay_resized[:, :, 3].astype(np.float32) / 255.0

    snowed_image = (1 - (intensity * alpha[:, :, np.newaxis])) * image.astype(np.float32) + (intensity * bgr)
    snowed_image = _to_uint8_image(snowed_image)

    hsv = cv2.cvtColor(snowed_image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def dynamic_snow_filter(scale, image, iterator):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    snow_overlay = next(iterator)
    snow_overlay = shift_color(snow_overlay, [71, 253, 135], [255, 255, 255])

    if snow_overlay.shape[:2] != image.shape[:2]:
        snow_overlay = cv2.resize(snow_overlay, (image.shape[1], image.shape[0]))

    bgr = snow_overlay[:, :, :3].astype(np.float32)
    mask = snow_overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def static_snow_filter(scale, image, snow_overlay):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    snow_overlay = shift_color(snow_overlay, [71, 253, 135], [255, 255, 255])

    if snow_overlay.shape[:2] != image.shape[:2]:
        snow_overlay = cv2.resize(snow_overlay, (image.shape[1], image.shape[0]))

    bgr = snow_overlay[:, :, :3].astype(np.float32)
    mask = snow_overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def dynamic_rain_filter(scale, image, iterator):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = next(iterator)
    rain_overlay = shift_color(rain_overlay, [31, 146, 59], [191, 35, 0])

    if rain_overlay.shape[:2] != image.shape[:2]:
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))

    bgr = rain_overlay[:, :, :3].astype(np.float32)
    mask = rain_overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1.0 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def dynamic_raindrop_filter(scale, image, iterator):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    overlay = next(iterator)
    overlay = shift_color(overlay, [71, 253, 135], [255, 255, 255])

    if overlay.shape[:2] != image.shape[:2]:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))

    bgr = overlay[:, :, :3].astype(np.float32)
    mask = overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1.0 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def static_rain_filter(scale, image, rain_overlay):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    rain_overlay = shift_color(rain_overlay, [31, 146, 59], [191, 35, 0])

    if rain_overlay.shape[:2] != image.shape[:2]:
        rain_overlay = cv2.resize(rain_overlay, (image.shape[1], image.shape[0]))

    bgr = rain_overlay[:, :, :3].astype(np.float32)
    mask = rain_overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1.0 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def object_overlay(scale, img1):
    img1 = _to_uint8_image(np.asarray(img1).copy())
    c = [10, 5, 3, 2, 1.5]
    overlay_path = "./perturbationdrive/OverlayImages/Logo_of_the_Technical_University_of_Munichpng.png"
    img2 = cv2.imread(overlay_path)
    assert img2 is not None, "file could not be read, check with os.path.exists()"

    img1_shape0_div_c_scale = int(img1.shape[0] / c[scale])
    img1_shape1_div_2 = int(img1.shape[1] / 2)
    img1_shape0_div_2 = int(img1.shape[0] / 2)

    target_image_width = int(
        img1.shape[1] * (img1_shape0_div_c_scale * 100.0 / img2.shape[0]) / 100
    )

    img2 = cv2.resize(
        img2,
        (img1_shape0_div_c_scale, target_image_width),
        interpolation=cv2.INTER_NEAREST,
    )

    img2_shape0_div_2 = int(img2.shape[0] / 2)
    img2_shape1_div_2 = int(img2.shape[1] / 2)

    height_roi = img1_shape0_div_2 - img2_shape0_div_2
    width_roi = img1_shape1_div_2 - img2_shape1_div_2

    rows, cols, _ = img2.shape
    roi = img1[height_roi: height_roi + rows, width_roi: width_roi + cols]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    img1[height_roi: height_roi + rows, width_roi: width_roi + cols] = dst

    return img1


def dynamic_object_overlay(scale, image, iterator):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    overlay = next(iterator)
    overlay = shift_color(overlay, [175, 221, 202], [0, 0, 0])

    if overlay.shape[:2] != image.shape[:2]:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))

    bgr = overlay[:, :, :3].astype(np.float32)
    mask = overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def static_object_overlay(scale, image, overlay):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    overlay = shift_color(overlay, [175, 221, 202], [0, 0, 0])

    if overlay.shape[:2] != image.shape[:2]:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))

    bgr = overlay[:, :, :3].astype(np.float32)
    mask = overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def dynamic_sun_filter(scale, image, iterator):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    overlay = next(iterator)
    overlay = shift_color(overlay, [223, 234, 212], [28, 202, 255])

    if overlay.shape[:2] != image.shape[:2]:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))

    bgr = overlay[:, :, :3].astype(np.float32)
    mask = overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def static_sun_filter(scale, image, overlay):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    overlay = shift_color(overlay, [223, 234, 212], [28, 202, 255])

    if overlay.shape[:2] != image.shape[:2]:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))

    bgr = overlay[:, :, :3].astype(np.float32)
    mask = overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def dynamic_lightning_filter(scale, image, iterator):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    overlay = next(iterator)
    overlay = shift_color(overlay, [5, 122, 101], [8, 152, 188])

    if overlay.shape[:2] != image.shape[:2]:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))

    bgr = overlay[:, :, :3].astype(np.float32)
    mask = overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def static_lightning_filter(scale, image, overlay):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    overlay = shift_color(overlay, [5, 122, 101], [8, 152, 188])

    if overlay.shape[:2] != image.shape[:2]:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))

    bgr = overlay[:, :, :3].astype(np.float32)
    mask = overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def dynamic_smoke_filter(scale, image, iterator):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    overlay = next(iterator)
    overlay = shift_color(overlay, [30, 112, 65], [132, 132, 132])

    if overlay.shape[:2] != image.shape[:2]:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))

    bgr = overlay[:, :, :3].astype(np.float32)
    mask = overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def static_smoke_filter(scale, image, overlay):
    image = _to_uint8_image(np.asarray(image).copy())
    intensity = [0.15, 0.25, 0.4, 0.6, 0.85][scale]
    overlay = shift_color(overlay, [30, 112, 65], [132, 132, 132])

    if overlay.shape[:2] != image.shape[:2]:
        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))

    bgr = overlay[:, :, :3].astype(np.float32)
    mask = overlay[:, :, 3] != 0
    image_f = image.astype(np.float32)
    image_f[mask] = (1 - intensity) * image_f[mask] + intensity * bgr[mask]
    return _to_uint8_image(image_f)


def perturb_high_attention_regions(
    saliency_map, image, perturbation, boundary=0.5, scale=0
):
    if boundary < 0 or boundary > 1:
        raise ValueError("The boundary value needs to be in the range of [0, 1]")

    image = _to_uint8_image(np.asarray(image).copy())
    mask = saliency_map > boundary
    noise_img = _to_uint8_image(perturbation(scale, image.copy()))
    image[mask] = noise_img[mask]
    return image


def perturb_highest_n_attention_regions(
    saliency_map, image, perturbation, n=30, scale=0
):
    if n < 0 or n > 100:
        raise ValueError("The threshold value needs to be in the range of [0, 100]")

    image = _to_uint8_image(np.asarray(image).copy())
    mask = saliency_map > np.percentile(saliency_map, n)
    noise_img = _to_uint8_image(perturbation(scale, image.copy()))
    image[mask] = noise_img[mask]
    return image


def perturb_lowest_n_attention_regions(
    saliency_map, image, perturbation, n=30, scale=0
):
    if n < 0 or n > 100:
        raise ValueError("The threshold value needs to be in the range of [0, 100]")

    image = _to_uint8_image(np.asarray(image).copy())
    thres = np.percentile(saliency_map, n)
    if thres == 0:
        mask = saliency_map <= thres
    else:
        mask = saliency_map < thres

    noise_img = _to_uint8_image(perturbation(scale, image.copy()))
    image[mask] = noise_img[mask]
    return image


def perturb_random_n_attention_regions(
    saliency_map, image, perturbation, n=30, scale=0
):
    if n < 0 or n > 100:
        raise ValueError("The n value needs to be in the range of [0, 100]")

    image = _to_uint8_image(np.asarray(image).copy())
    mask = np.random.choice(
        [True, False], size=saliency_map.shape, p=[n / 100, 1 - n / 100]
    )
    noise_img = _to_uint8_image(perturbation(scale, image.copy()))
    image[mask] = noise_img[mask]
    return image


def effects_attention_regions(saliency_map, scale, image, type):
    image = _to_uint8_image(np.asarray(image).copy())
    mask = saliency_map > np.percentile(saliency_map, 90)
    coordinates = np.argwhere(mask)

    if coordinates.shape[0] == 0:
        return image

    num_select = min(scale + 1, coordinates.shape[0])
    selected_coords = coordinates[
        np.random.choice(coordinates.shape[0], num_select, replace=False)
    ]
    selected_coords_tuples = [tuple(row) for row in selected_coords]
    selected_coords_tuples = clamp_values(
        selected_coords_tuples, 5, image.shape[1] - 5, 5, image.shape[0] - 5
    )

    List_of_Drops, _, _ = generate_label(
        image.shape[1], image.shape[0], selected_coords_tuples, cfg
    )
    output_image = generateDrops(image, cfg, List_of_Drops)
    return _to_uint8_image(output_image)


def shift_color(image, source_color, target_color):
    image = _to_uint8_image(image) if image.shape[2] == 3 else image.copy()
    has_alpha = image.shape[2] == 4

    if has_alpha:
        bgr, alpha = image[:, :, :3], image[:, :, 3]
    else:
        bgr = image

    source_color = np.array(source_color, dtype=np.int16)
    target_color = np.array(target_color, dtype=np.int16)
    color_diff = target_color - source_color

    shifted_bgr = np.clip(bgr.astype(np.int16) + color_diff, 0, 255).astype(np.uint8)

    if has_alpha:
        return cv2.merge((shifted_bgr, alpha))
    return shifted_bgr