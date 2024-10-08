import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from util import load_data, preprocess_data
from tensorflow.keras.models import load_model
import random
from tqdm import tqdm  # Add tqdm for the loading bar

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
print('GPU is available' if len(physical_devices) > 0 else 'Not available')

INPUT_SIZE = (256, 256)
INPUT_SHAPE = (256, 256, 3)  # Color images, 3 channels

# Load data
img_path = r"C:\Users\ethan\OneDrive\Desktop\Thesis 2024\Sydney 1903_JS"
mask_path = r"C:\Users\ethan\OneDrive\Desktop\Thesis 2024\Base Images\New folder\1903"
output_folder = r"C:\Users\ethan\OneDrive\Desktop\Thesis 2024\Sydney 1903_JS_Mask"  # Change this path if needed
image_filenames = load_data(img_path)
mask_filenames = load_data(mask_path)

# Preprocess data
with tf.device("/device:GPU:0"):
    images, masks = preprocess_data(img_path, mask_path, image_filenames, mask_filenames, input_size=INPUT_SIZE, augmented=False)

# Print the shapes of the entire dataset
print('Shape of image data: ' + str(images.shape))
print('Shape of mask data: ' + str(masks.shape))

# Load the saved model
model = load_model('./models/save_best_image.keras')

# Predict masks for all images with a loading bar
print("Predicting masks...")
predictions = []
for i in tqdm(range(len(images)), desc="Predicting"):
    pred = model.predict(np.expand_dims(images[i], axis=0))
    predictions.append((pred > 0.5).astype(np.uint8))
predictions = np.array(predictions)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all images and masks with a loading bar for saving results
print("Saving predicted masks and overlays...")
for i in tqdm(range(len(images)), desc="Saving"):
    image_filename = os.path.basename(image_filenames[i])  # Extract the filename from the full path

    image = (images[i] * 255).astype(np.uint8)
    mask = predictions[i]
    ground_truth = masks[i] * np.array([255, 255, 255])  # Convert the foreground into white color

    overlay = image.copy()

    # Repeat mask along channels for visualization
    mask = np.repeat(mask, 3, axis=3)[0]  # Matching the size of the mask and the image to perform an overlay
    inverted_mask = 1 - mask

    yellow_mask = np.array([255, 255, 255]) * mask

    # Apply the mask on the image
    result = image * inverted_mask + yellow_mask
    alpha = 0.2
    predicted_overlay = cv2.addWeighted(overlay, alpha, result.astype(overlay.dtype), 1 - alpha, 0)

    # Save the predicted mask and overlay using the original input image filename
    cv2.imwrite(os.path.join(output_folder, f'predicted_{image_filename}'), yellow_mask)

# Randomly select images for display
random_indices = random.sample(range(0, len(images)), 10)
test_sample = images[random_indices]

# Optionally, create a plot displaying results (you can skip this if you don't need to display them)
fig, axes = plt.subplots(len(test_sample), 4, figsize=(10, 3 * len(test_sample)))

print("Creating plot...")
for i in tqdm(range(len(test_sample)), desc="Plotting"):
    image = (test_sample[i] * 255).astype(np.uint8)
    mask = predictions[random_indices[i]]
    ground_truth = masks[random_indices[i]] * np.array([255, 255, 255])  # Convert the foreground into white color

    overlay = image.copy()

    # Repeat mask along channels for visualization
    mask = np.repeat(mask, 3, axis=2)  # Matching the size of the mask and the image to perform an overlay
    inverted_mask = 1 - mask

    yellow_mask = np.array([255, 255, 255]) * mask

    # Apply the mask on the image
    result = image * inverted_mask + yellow_mask
    alpha = 0.2
    predicted_overlay = cv2.addWeighted(overlay, alpha, result.astype(overlay.dtype), 1 - alpha, 0)

    # Plot the original image, ground truth, predicted mask, and overlay
    axes[i, 0].imshow(image)
    axes[i, 0].set_title('Original')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(ground_truth)
    axes[i, 1].set_title('Ground Truth')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(yellow_mask)
    axes[i, 2].set_title('Predicted')
    axes[i, 2].axis('off')

    axes[i, 3].imshow(predicted_overlay)
    axes[i, 3].set_title('Predicted Overlay')
    axes[i, 3].axis('off')

# Adjust the spacing between subplots and save the plot (optional)
plt.tight_layout()
plt.savefig('result_all_images.png', bbox_inches='tight')

# Show the plot (optional)
plt.show()
