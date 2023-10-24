import numpy as np

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
