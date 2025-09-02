from perturbationdrive import ImagePerturbation
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = '0001_0.png'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

intensity = 2
perturbation_names =["gaussian_noise","poisson_noise","impulse_noise","defocus_blur","glass_blur","motion_blur",
"zoom_blur","increase_brightness","contrast","elastic","pixelate","jpeg_filter","shear_image",
"translate_image", "scale_image", "rotate_image", "fog_mapping", "splatter_mapping",
"dotted_lines_mapping", "zigzag_mapping","canny_edges_mapping","speckle_noise_filter","false_color_filter","high_pass_filter","low_pass_filter","phase_scrambling",
"histogram_equalisation", "reflection_filter", "white_balance_filter", "sharpen_filter",
"grayscale_filter", "posterize_filter", "cutout_filter", "sample_pairing_filter", "gaussian_blur",
"saturation_filter", "saturation_decrease_filter", "fog_filter", "frost_filter", "snow_filter","object_overlay",
"dynamic_snow_filter","dynamic_rain_filter","dynamic_object_overlay","dynamic_sun_filter","dynamic_lightning_filter","dynamic_smoke_filter",
"static_snow_filter","static_rain_filter","static_object_overlay","static_sun_filter","static_lightning_filter","static_smoke_filter",
"candy","la_muse","mosaic","feathers","the_scream","udnie","the_wave","starry_night","la_muse","composition_vii"]
perturbation_controller = ImagePerturbation(funcs=perturbation_names)
perturbed_images = []
perturbed_names=[]
for perturbation in perturbation_names:
    if "dynamic" in perturbation:
        for i in range(0,3):
            test_image=image.copy()
            perturbed_image=perturbation_controller.perturbation(test_image,perturbation,intensity)
            perturbed_images.append(perturbed_image)
            str_name=perturbation+" "+str(i)
            perturbed_names.append(str_name)
    else:
        test_image=image.copy()
        perturbed_image=perturbation_controller.perturbation(test_image,perturbation,intensity)
        perturbed_images.append(perturbed_image)
        perturbed_names.append(perturbation)
    
num_images = len(perturbed_images)
print(num_images)
num_cols = 10
num_rows = 10

plt.figure(figsize=(num_cols, num_rows))
for i, img in enumerate(perturbed_images):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(num_rows, num_cols, i + 1)
    img_rgb = img[..., ::-1]
    plt.imshow(img_rgb)
    plt.title(perturbed_names[i])
    plt.axis('off')
plt.tight_layout()
plt.show()


