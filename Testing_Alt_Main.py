import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
from util import load_data, preprocess_data
from unet import unet
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import random


# Check if GPU is available and set it to be used
if tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    print("GPU is available and selected.")
else:
    print("No GPU found. Using CPU.")

INPUT_SIZE = (256, 256)
INPUT_SHAPE = (256, 256, 3) # color images, 3 channels


def display_data(img_path,msk_path, image_paths, mask_paths):

    fig, axes = plt.subplots(5, 2, figsize=(10, 15))

    # Iterate over the image and mask pairs and display them in subplots
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Load the image and mask using your preferred method
        image = plt.imread(os.path.join(img_path,image_path))
        mask = plt.imread(os.path.join(msk_path,mask_path))

        # Plot the image and mask in the corresponding subplot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask)
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig('samples.png', bbox_inches='tight')  # Save as PNG image

    # Show the plot
    plt.show()
    return



# load data
img_path = r"C:\Users\ethan\OneDrive\Desktop\Thesis 2024\Code\Python_UNET_Code\Predictions\New folder\imgs"
mask_path = r"C:\Users\ethan\OneDrive\Desktop\Thesis 2024\Code\Python_UNET_Code\Predictions\New folder\masks_thick"
image_filenames = load_data(img_path)
mask_filenames = load_data(mask_path)
# display the first 5 pairs of image and mask
random_indices = random.sample(range(0, len(image_filenames)), 5)
display_data(img_path,mask_path, image_filenames[random_indices], mask_filenames[random_indices])


# preprocess data
with tf.device("/device:GPU:0"):
    # no augmentation due to limited computational resources
    # already have a large amount of data, roughly 12,000 images and masks
    images,masks = preprocess_data(img_path,mask_path, image_filenames, mask_filenames,INPUT_SIZE, augmented=True)
    
# get shape
print('Shape of image data: ' + str(images.shape))
print('Shape of mask data: ' + str(masks.shape))
    

# Split the dataset into training and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.2, random_state=820)
train_images, test_images, train_masks, test_masks = train_test_split(train_images, train_masks, test_size=0.2, random_state=820)

checkpoint = tf.keras.callbacks.ModelCheckpoint('./models/save_best_image.keras', verbose=1, save_best_only=True)

# Initialize the model
model = unet(INPUT_SHAPE, output_layer=1)
model.summary()

# complie the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

epochs = 20
with tf.device("/GPU:0"):
    history = model.fit(train_images, train_masks, batch_size=32, epochs=epochs, validation_data=(val_images, val_masks), callbacks=[checkpoint])
    
    # Accessing training and testing accuracy
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(1, epochs + 1)

    # Plotting
    plt.plot(epochs, train_accuracy, 'r', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    
model.save('./models/image_extraction.keras')  
eval = model.evaluate(test_images, test_masks)
print('Test accuracy: ' + "{:.2f}".format(eval[1]))
