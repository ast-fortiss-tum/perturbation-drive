import numpy as np


def round_to_nearest_odd(n):
    """
    Round an integer to the nearest odd number.

    Parameters:
    - n: Integer to be rounded.

    Returns:
    - Rounded odd integer.
    """

    return n + 1 if n % 2 == 0 else n


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
