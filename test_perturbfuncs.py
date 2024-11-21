from perturbationdrive import gaussian_noise,poisson_noise,impulse_noise,defocus_blur,glass_blur,motion_blur,zoom_blur,increase_brightness,contrast,elastic,pixelate,jpeg_filter,shear_image,translate_image,scale_image,rotate_image,fog_mapping,splatter_mapping,dotted_lines_mapping,zigzag_mapping,canny_edges_mapping,speckle_noise_filter,false_color_filter,high_pass_filter,low_pass_filter,phase_scrambling,histogram_equalisation,reflection_filter,white_balance_filter,sharpen_filter,grayscale_filter,posterize_filter,cutout_filter,sample_pairing_filter,gaussian_blur,saturation_filter,saturation_decrease_filter,fog_filter,frost_filter,snow_filter,object_overlay
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = '0001_0.png'  # Replace with the path to your image
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

scale = 4
functions = [
    gaussian_noise, poisson_noise, impulse_noise, defocus_blur, glass_blur, motion_blur,
    zoom_blur, increase_brightness, contrast, elastic, pixelate, jpeg_filter, shear_image,
    translate_image, scale_image, rotate_image, fog_mapping, splatter_mapping,
    dotted_lines_mapping, zigzag_mapping, canny_edges_mapping, speckle_noise_filter,
    false_color_filter, high_pass_filter, low_pass_filter, phase_scrambling,
    histogram_equalisation, reflection_filter, white_balance_filter, sharpen_filter,
    grayscale_filter, posterize_filter, cutout_filter, sample_pairing_filter, gaussian_blur,
    saturation_filter, saturation_decrease_filter, fog_filter, frost_filter, snow_filter,object_overlay
]


perturbed_images = []
for func in functions:
    try:
        image_to_test=image.copy()
        perturbed_image = func(scale,image_to_test)  # Assuming scale=2 is a valid input for all functions
        perturbed_image = cv2.cvtColor(perturbed_image, cv2.COLOR_BGR2RGB)
        perturbed_images.append(perturbed_image) 
    except Exception as e:
        print(f"Error applying {func.__name__}: {e}")

# Plot all perturbed images
num_images = len(perturbed_images)
print(num_images)
num_cols = 10
num_rows = 5

plt.figure(figsize=(num_cols, num_rows))
for i, img in enumerate(perturbed_images):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(img)
    plt.title(functions[i].__name__)
    plt.axis('off')
plt.tight_layout()
plt.show()


