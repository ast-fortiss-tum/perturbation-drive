import perturbationdrive
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path ='./examples/dataset_dummy/0001_0.png'  # Replace with the path to your image
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

scale = 4
functions = [
    perturbationdrive.gaussian_noise,
    perturbationdrive.poisson_noise, 
    perturbationdrive.impulse_noise, 
    perturbationdrive.defocus_blur, 
    perturbationdrive.motion_blur, 
    perturbationdrive.glass_blur,
    perturbationdrive.zoom_blur, 
    perturbationdrive.increase_brightness, 
    perturbationdrive.contrast, 
    perturbationdrive.elastic, 
    perturbationdrive.pixelate, 
    perturbationdrive.jpeg_filter, 
    perturbationdrive.shear_image,
    perturbationdrive.translate_image, 
    perturbationdrive.scale_image, 
    perturbationdrive.rotate_image, 
    perturbationdrive.fog_mapping, 
    perturbationdrive.splatter_mapping,
    perturbationdrive.dotted_lines_mapping, 
    perturbationdrive.zigzag_mapping, 
    perturbationdrive.canny_edges_mapping, 
    perturbationdrive.speckle_noise_filter,
    perturbationdrive.false_color_filter, 
    perturbationdrive.high_pass_filter, 
    perturbationdrive.low_pass_filter, 
    perturbationdrive.phase_scrambling,
    perturbationdrive.histogram_equalisation, 
    perturbationdrive.reflection_filter, 
    perturbationdrive.white_balance_filter, 
    perturbationdrive.sharpen_filter,
    perturbationdrive.grayscale_filter, 
    perturbationdrive.posterize_filter, 
    perturbationdrive.cutout_filter, 
    perturbationdrive.sample_pairing_filter, 
    perturbationdrive.gaussian_blur,
    perturbationdrive.saturation_filter, 
    perturbationdrive.saturation_decrease_filter, 
    perturbationdrive.fog_filter, 
    perturbationdrive.frost_filter, 
    perturbationdrive.snow_filter,
    perturbationdrive.object_overlay, 
]


perturbed_images = []
for func in functions:
    try:
        image_to_test=image.copy()
        perturbed_image = func(scale,image_to_test)
        
        perturbed_image = cv2.cvtColor(perturbed_image, cv2.COLOR_BGR2RGB)
        perturbed_images.append(perturbed_image) 
    except Exception as e:
        print(f"Error applying {func.__name__}: {e}")


num_images = len(perturbed_images)
num_cols = 7
num_rows = (num_images + num_cols - 1) // num_cols  # Round up

plt.figure(figsize=(num_cols * 2, num_rows * 2))  # Bigger figure
for i, img in enumerate(perturbed_images):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(img)
    plt.title(functions[i].__name__, fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.savefig("perturbed_outputs.png", dpi=300)
plt.show()
