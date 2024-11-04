import numpy as np
import os
import requests
from typing import Tuple


def round_to_nearest_odd(n):
    """
    Round an integer to the nearest odd number.

    Parameters:
    - n: Integer to be rounded.

    Returns:
    - Rounded odd integer.
    """

    return n + 1 if n % 2 == 0 else n

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


def scramble_channel(channel, severity):
    """Helper function to scramble a single color channel"""
    f = np.fft.fft2(channel)
    magnitude = np.abs(f)
    angle = np.angle(f)

    random_phase = np.exp(
        1j * (angle + severity * np.pi * np.random.rand(*angle.shape))
    )
    scrambled_freq = magnitude * random_phase
    scrambled_img = np.fft.ifft2(scrambled_freq).real
    return np.clip(scrambled_img, 0, 255).astype(np.uint8)


def equalise_power(channel, alpha):
    """Helper function to equalise power of a single channel"""
    # Compute the 2D Fourier transform of the channel
    f = np.fft.fftshift(np.fft.fft2(channel))
    magnitude = np.abs(f)
    angle = np.angle(f)

    # Equalize the power
    equalised_magnitude = magnitude**alpha

    # Combine equalised magnitude and original phase
    equalised_freq = equalised_magnitude * np.exp(1j * angle)

    # Compute inverse 2D Fourier transform
    equalised_img = np.fft.ifft2(np.fft.ifftshift(equalised_freq)).real
    return np.clip(equalised_img, 0, 255).astype(np.uint8)


def simple_white_balance(img):
    """Helper function to perform simple white balance on an image"""
    # Calculate the mean of each channel
    r_mean, g_mean, b_mean = (
        np.mean(img[..., 0]),
        np.mean(img[..., 1]),
        np.mean(img[..., 2]),
    )

    # Calculate the global mean
    global_mean = np.mean([r_mean, g_mean, b_mean])

    # Calculate the scaling factors
    r_scale = global_mean / r_mean
    g_scale = global_mean / g_mean
    b_scale = global_mean / b_mean

    # Scale the channels
    img[..., 0] = np.clip(img[..., 0] * r_scale, 0, 255)
    img[..., 1] = np.clip(img[..., 1] * g_scale, 0, 255)
    img[..., 2] = np.clip(img[..., 2] * b_scale, 0, 255)

    return img


def download_file(url, target_folder):
    """Downloads file from url and moves it to target folder"""
    local_filename = os.path.join(target_folder, url.split("/")[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def calculate_velocities(positions, speeds) -> Tuple[float, float, float]:
    """
    Calculate velocities given a list of positions and corresponding speeds.
    """
    velocities = []

    for i in range(len(positions) - 1):
        displacement = np.array(positions[i + 1]) - np.array(positions[i])
        direction = displacement / np.linalg.norm(displacement)
        velocity = direction * speeds[i]
        velocities.append(velocity)

    return velocities
