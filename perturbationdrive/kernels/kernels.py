import numpy as np
from scipy.ndimage import zoom as scizoom

def diamond_square(size, roughness):
    """
    Generate a fog pattern using the Diamond-Square algorithm.

    Parameters:
    - size: The size of the final grid. Should be 2^n + 1.
    - roughness: Determines the randomness of the pattern.

    Returns:
    - grid: Generated pattern.
    """
    # Initialize grid with zeros
    grid = np.zeros((size, size), dtype=float)

    # Set initial random corners
    grid[0, 0] = np.random.random()
    grid[0, size - 1] = np.random.random()
    grid[size - 1, 0] = np.random.random()
    grid[size - 1, size - 1] = np.random.random()

    step = size - 1
    while step > 1:
        half_step = step // 2

        # Diamond step
        for x in range(0, size - 1, step):
            for y in range(0, size - 1, step):
                average = (
                    grid[x, y]
                    + grid[x + step, y]
                    + grid[x, y + step]
                    + grid[x + step, y + step]
                ) / 4.0
                grid[x + half_step, y + half_step] = (
                    average + np.random.random() * roughness
                )

        # Square step
        for x in range(0, size, half_step):
            start = 0 if (x + half_step) % step == 0 else half_step
            for y in range(start, size, step):
                total = 0
                count = 0
                # Check left
                if x - half_step >= 0:
                    total += grid[x - half_step, y]
                    count += 1
                # Check right
                if x + half_step < size:
                    total += grid[x + half_step, y]
                    count += 1
                # Check above
                if y - half_step >= 0:
                    total += grid[x, y - half_step]
                    count += 1
                # Check below
                if y + half_step < size:
                    total += grid[x, y + half_step]
                    count += 1
                average = total / count
                grid[x, y] = average + np.random.random() * roughness

        step //= 2

    return grid

def create_disk_kernel(radius):
    """Create a disk-shaped kernel with the given radius."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    kernel[mask] = 1
    # Normalize the kernel so that the sum of its elements is 1.
    kernel /= kernel.sum()
    return kernel

def create_motion_blur_kernel(size, angle):
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

def clipped_zoom(img, zoom_factor):
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