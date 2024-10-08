import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import flip_left_right, flip_up_down, rot90, random_brightness, random_contrast
from tensorflow.keras.preprocessing.image import random_shift, random_zoom
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for the loading bar


def sort_by_name(arr, split_at):
    print(arr)
    return sorted([f for f in arr if f.split(split_at)[0].isdigit()], key=lambda x: int(x.split(split_at)[0]))


def load_data(dir_path):
    # dir_path = '../satellite-roads/train/'
    directory = os.listdir(dir_path)
    images = []

    for filename in directory:
        if filename.split('.')[1] == 'jpg':
            images.append(filename)
        elif filename.split('.')[1] == 'png':
            images.append(filename)
        elif filename.split('.')[1] == 'gif':
            images.append(filename)


    return np.array(images)

def rgb_to_hsv(rgb):
    # Ensure the input is a float array
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    max_val = np.max(rgb, axis=-1)
    min_val = np.min(rgb, axis=-1)
    delta = max_val - min_val

    # Initialize HSV arrays
    h = np.zeros_like(max_val)
    s = np.zeros_like(max_val)
    v = max_val

    # Calculate hue
    mask = delta != 0
    h[mask & (max_val == r)] = (60 * ((g[mask & (max_val == r)] - b[mask & (max_val == r)]) / delta[mask & (max_val == r)])) % 360
    h[mask & (max_val == g)] = (60 * ((b[mask & (max_val == g)] - r[mask & (max_val == g)]) / delta[mask & (max_val == g)]) + 120) % 360
    h[mask & (max_val == b)] = (60 * ((r[mask & (max_val == b)] - g[mask & (max_val == b)]) / delta[mask & (max_val == b)]) + 240) % 360

    # Calculate saturation
    s[max_val > 0] = delta[max_val > 0] / max_val[max_val > 0]

    return np.stack([h, s, v], axis=-1)

def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Initialize RGB arrays
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)

    # Convert hue to [0, 6]
    h = h / 60.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i = i % 6

    # Assign RGB values based on hue sector
    r[i == 0], g[i == 0], b[i == 0] = v[i == 0], t[i == 0], p[i == 0]
    r[i == 1], g[i == 1], b[i == 1] = q[i == 1], v[i == 1], p[i == 1]
    r[i == 2], g[i == 2], b[i == 2] = p[i == 2], v[i == 2], t[i == 2]
    r[i == 3], g[i == 3], b[i == 3] = p[i == 3], q[i == 3], v[i == 3]
    r[i == 4], g[i == 4], b[i == 4] = t[i == 4], p[i == 4], v[i == 4]
    r[i == 5], g[i == 5], b[i == 5] = v[i == 5], p[i == 5], q[i == 5]

    return np.stack([r, g, b], axis=-1)


def color_jitter(image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), saturation_range=(0.8, 1.2), hue_range=(-0.1, 0.1)):
    # Apply random brightness
    brightness_factor = np.random.uniform(*brightness_range)
    image = np.clip(image * brightness_factor, 0, 1)

    # Apply random contrast
    contrast_factor = np.random.uniform(*contrast_range)
    mean = np.mean(image)
    image = np.clip((image - mean) * contrast_factor + mean, 0, 1)

    # Convert to HSV for saturation and hue adjustment
    hsv_image = rgb_to_hsv(image)
    
    # Apply random saturation
    saturation_factor = np.random.uniform(*saturation_range)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation_factor, 0, 1)

    # Apply random hue
    hue_factor = np.random.uniform(*hue_range)
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_factor) % 1.0

    # Convert back to RGB
    image = hsv_to_rgb(hsv_image)

    return image


def preprocess_data(img_path, mask_path, sorted_images, sorted_masks, input_size, augmented=False):
    images = []
    masks = []

    # Use tqdm to create a loading bar for the loop
    for img_file, mask_file in tqdm(zip(sorted_images, sorted_masks), total=len(sorted_images), desc="Processing Images"):
        img = load_img(os.path.join(img_path, img_file), target_size=input_size, color_mode='rgb')
        mask = load_img(os.path.join(mask_path, mask_file), target_size=input_size, color_mode='grayscale')

        # Convert image and mask to arrays
        img_array = img_to_array(img) / 255.0
        mask_array = img_to_array(mask, dtype=np.bool_)

        # Append original image and mask
        images.append(img_array)
        masks.append(mask_array)

        if augmented:
            # Flip left-right
            images.append(flip_left_right(img_array))
            masks.append(flip_left_right(mask_array))

            # # Flip up-down
            images.append(flip_up_down(img_array))
            masks.append(flip_up_down(mask_array))

            # 90-degree rotation
            for k in range(1, 4):
                images.append(rot90(img_array, k=k))
                masks.append(rot90(mask_array, k=k))

            # Random brightness adjustment
            images.append(random_brightness(img_array, max_delta=0.6))
            masks.append(mask_array)  # Mask remains the same
            
            # Random brightness adjustment
            images.append(flip_up_down(random_brightness(img_array, max_delta=0.4)))
            masks.append(flip_up_down(mask_array))  # Mask remains the same

            # Random color jitter
            images.append(color_jitter(img_array))
            masks.append(mask_array)  # Mask remains the same

            # Random contrast adjustment
            images.append(random_contrast(img_array, lower=0.1, upper=1.3))
            masks.append(mask_array)  # Mask remains the same

            # Random shifts
            shift_range = 0.4  # Shift by 10% of image size
            shifted_img = random_shift(img_array, wrg=shift_range, hrg=shift_range)
            images.append(shifted_img)
            masks.append(random_shift(mask_array, wrg=shift_range, hrg=shift_range))

            # # Random zoom
            zoom_range = [0.8, 1.2]  # Zoom in or out between 80% and 120%
            zoomed_img = random_zoom(img_array, zoom_range)
            images.append(zoomed_img)
            masks.append(random_zoom(mask_array, zoom_range))

    # Convert lists to numpy arrays
    images = np.array(images)
    masks = np.array(masks)

    return images, masks
def display_data(dir_path, image_paths, mask_paths):

    fig, axes = plt.subplots(5, 2, figsize=(10, 20))

    # Iterate over the image and mask pairs and display them in subplots
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Load the image and mask using your preferred method
        image = plt.imread(dir_path + image_path)
        mask = plt.imread(dir_path + mask_path)

        # Plot the image and mask in the corresponding subplot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask)
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    return

