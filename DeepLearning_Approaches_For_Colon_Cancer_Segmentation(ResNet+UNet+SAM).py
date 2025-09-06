#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Required libraries for the project
get_ipython().system('pip install opencv-python')

# Import necessary libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from google.colab import files


# In[ ]:





# In[2]:


import zipfile
import os

# Unzip the file
zip_file = "Kvasir-SEG.zip"  # Replace with the name of your zip file
destination_dir = "/content/your_folder"  # Replace with the desired destination folder

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(destination_dir)

# List the contents of the folder to verify
os.listdir(destination_dir)


# In[3]:


# Specify paths to the images and masks
images_path = "/content/your_folder/Kvasir-SEG/images"  # Update with actual path
masks_path = "/content/your_folder/Kvasir-SEG/masks"  # Update with actual path

image_size = (256, 256)  # You can adjust the size


# In[10]:


def apply_color_mapping(image):
    """
    Convert blue-tone images to skin-tone images using color mapping.
    """
    # Convert image to LAB color space for better control over tone
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Adjust the LAB channels to map blue to a skin tone
    l, a, b = cv2.split(lab)
    b = cv2.add(b, 50)  # Increase the 'b' channel to move towards a warmer tone
    adjusted_lab = cv2.merge((l, a, b))

    # Convert back to BGR color space
    skin_tone_image = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
    return skin_tone_image


# In[11]:


image_size = (256, 256)  # You can adjust the size

def load_images_and_masks_with_color_adjustment(images_path, masks_path):
    images = []
    masks = []

    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(images_path, img_file)
        mask_path = os.path.join(masks_path, mask_file)

        image = cv2.imread(img_path)
        image = apply_color_mapping(image)  # Apply skin-tone transformation

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, image_size)
        mask = cv2.resize(mask, image_size)

        image = image / 255.0
        mask = mask / 255.0

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load images and masks
# The function call has been changed from 'load_images_and_masks' to 'load_images_and_masks_with_color_adjustment'
images, masks = load_images_and_masks_with_color_adjustment(images_path, masks_path)

# Reshape masks to add channel dimension
masks = masks.reshape(*masks.shape, 1)  # Add a channel dimension to the masks

# Print to confirm
print(f"Loaded {len(images)} images and {len(masks)} masks")
print(f"Masks shape: {masks.shape}")  # Print the shape of masks to verify


# In[12]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import ImageDataGenerator
# Data augmentation parameters
datagen_args = dict(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],  # Adjust brightness
    # 'contrast_stretching' is not a valid argument for ImageDataGenerator
    # Consider using 'brightness_range' for similar effect
    # contrast_stretching=True,
    horizontal_flip=True,
    fill_mode='nearest'
)


image_datagen = ImageDataGenerator(**datagen_args)
mask_datagen = ImageDataGenerator(**datagen_args)

# Flow from numpy arrays
image_generator = image_datagen.flow(images, batch_size=32, seed=42)
mask_generator = mask_datagen.flow(masks, batch_size=32, seed=42)

# Combine image and mask generators
train_generator = zip(image_generator, mask_generator)


# In[17]:


import tensorflow as tf
from tensorflow.keras import backend as K

# IoU Metric
def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Dice Coefficient Metric
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


# In[14]:


# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Now you can cast the data types
images = images.astype('float32')
masks = masks.astype('float32')
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_val = x_val.astype('float32')
y_val = y_val.astype('float32')


# In[18]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def integrated_hybrid_model_with_pretrained(input_size=(256, 256, 3)):
    # Pretrained ResNet50 as the encoder backbone
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_size)

    # Use selected intermediate layers for the UNet decoder path
    resnet_output = base_model.get_layer('conv4_block6_out').output  # Example intermediate layer
    resnet_low_level = base_model.get_layer('conv2_block3_out').output  # Low-level feature layer

    # Ensure the shapes match for concatenation in the decoder path
    upsampled_resnet_output = layers.UpSampling2D((2, 2))(resnet_output)  # Upsample ResNet output to match low-level features

    # UNet Decoder Path
    d1 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(upsampled_resnet_output)
    d1 = layers.Concatenate()([d1, resnet_low_level])  # Skip connection from low-level ResNet features
    d1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(d1)

    d2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d1)
    d2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(d2)

    # Print shape of d2 for debugging
    print(f"Shape of d2: {d2.shape}")

    # SAM Path
    sam_c1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(base_model.input)
    sam_p1 = layers.MaxPooling2D((2, 2))(sam_c1)

    sam_c2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(sam_p1)
    sam_p2 = layers.MaxPooling2D((2, 2))(sam_c2)

    sam_bottleneck = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(sam_p2)

    # Align dimensions for SAM bottleneck and UNet decoder
    upsampled_sam_bottleneck = layers.UpSampling2D((2, 2))(sam_bottleneck)  # Upsample SAM bottleneck to match d2 dimensions

    # Reduce channels of upsampled_sam_bottleneck to match d2
    reduced_sam_bottleneck = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(upsampled_sam_bottleneck)

    # Print shape of reduced_sam_bottleneck for debugging
    print(f"Shape of reduced_sam_bottleneck: {reduced_sam_bottleneck.shape}")

    # Merge UNet Decoder and SAM bottleneck
    merged_bottleneck = layers.Concatenate()([d2, reduced_sam_bottleneck])  # Align spatial dimensions

    # Final Decoder Path
    final_decoder = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(merged_bottleneck)
    final_decoder = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(final_decoder)
    final_decoder = layers.Conv2D(1, (1, 1), activation='sigmoid')(final_decoder)  # Output layer

    # Define the full model
    model = models.Model(inputs=base_model.input, outputs=final_decoder)
    return model

    # Create the model with a pretrained backbone
integrated_model = integrated_hybrid_model_with_pretrained()

# Compile the model with additional metrics
integrated_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', iou_metric, dice_coefficient]
)

# Print model summary
integrated_model.summary()


# In[19]:


integrated_model = integrated_hybrid_model_with_pretrained()


# In[20]:


def weighted_binary_crossentropy(y_true, y_pred, pos_weight=2):
    """
    Weighted binary crossentropy for class imbalance.
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    return -tf.reduce_mean(pos_weight * y_true * tf.math.log(y_pred) +
                           (1 - y_true) * tf.math.log(1 - y_pred))


# In[21]:


# Assuming 'images' and 'masks' are loaded as numpy arrays
# Split the dataset into training and validation sets (80% train, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Print the shapes to verify
print(f"Training set shape: {x_train.shape}, {y_train.shape}")
print(f"Validation set shape: {x_val.shape}, {y_val.shape}")


# In[22]:


def image_mask_generator(images_path, masks_path, batch_size, image_size=(256, 256)):
    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    while True:
        batch_images = []
        batch_masks = []

        for i in range(batch_size):
            idx = np.random.randint(0, len(image_files))  # Randomly select an index

            # Load images and masks
            img_path = os.path.join(images_path, image_files[idx])
            mask_path = os.path.join(masks_path, mask_files[idx])

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Resize images and masks
            image = cv2.resize(image, image_size)
            mask = cv2.resize(mask, image_size)

            # Normalize image and mask
            image = image / 255.0
            mask = mask / 255.0

            batch_images.append(image)
            batch_masks.append(np.expand_dims(mask, axis=-1))  # Add an extra dimension for the mask

        yield np.array(batch_images), np.array(batch_masks)

# Set the batch size
batch_size = 16

# Define train generator
train_generator = image_mask_generator(images_path, masks_path, batch_size)


# In[23]:


# Compile the model with additional metrics
integrated_model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy', iou_metric, dice_coefficient])


# In[24]:


# Train the model
history = integrated_model.fit(train_generator,
                                steps_per_epoch=len(x_train) // batch_size,
                                epochs=50,
                                validation_data=(x_val, y_val))  # Validation data as numpy arrays


# In[25]:


# Save the trained model
integrated_model.save('kvasir_colon_cancer_segmentation_unet.h5')


# In[26]:


# Evaluate the model and print all metrics including IoU and Dice coefficient
loss, accuracy, iou, dice = integrated_model.evaluate(x_val, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")
print(f"Validation IoU: {iou}")
print(f"Validation Dice Coefficient: {dice}")


# In[ ]:


# Predict on validation data
preds = integrated_model.predict(x_val)

def visualize_result_with_color_mapping(image, mask, prediction):
    skin_tone_image = apply_color_mapping(image)

    plt.figure(figsize=(12, 6))

    # Original blue-tone image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original (Blue Tone)")

    # Skin-tone converted image
    plt.subplot(1, 4, 2)
    plt.imshow(skin_tone_image)
    plt.title("Skin Tone Image")

    # True mask
    plt.subplot(1, 4, 3)
    plt.imshow(mask[:, :, 0], cmap='gray')
    plt.title("True Mask")

    # Predicted mask
    plt.subplot(1, 4, 4)
    plt.imshow(prediction[:, :, 0], cmap='gray')
    plt.title("Predicted Mask")

    plt.show()


visualize_result_with_color_mapping(x_val[0], y_val[0], preds[0])

visualize_result_with_color_mapping(x_val[1], y_val[1], preds[1])


# In[29]:


def advanced_post_process(prediction, kernel_size=3, threshold=0.5):
    binary_mask = (prediction > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return refined_mask


# In[30]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Advanced Post-Processing Function
def advanced_post_process(prediction, kernel_size=3, threshold=0.5):
    """
    Post-process the prediction using thresholding and morphological operations.
    """
    # Apply thresholding
    binary_mask = (prediction > threshold).astype(np.uint8)

    # Define a structuring element (elliptical kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply morphological closing to refine the mask
    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return refined_mask

# Visualization Function
def visualize_advanced_post_process(image, raw_prediction, refined_mask):
    """
    Visualize the original image, raw predicted mask, and refined mask.
    """
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Raw predicted mask
    plt.subplot(1, 3, 2)
    plt.imshow(raw_prediction, cmap='gray')
    plt.title("Raw Predicted Mask")
    plt.axis("off")

    # Refined mask after post-processing
    plt.subplot(1, 3, 3)
    plt.imshow(refined_mask, cmap='gray')
    plt.title("Refined Mask (Post-Processed)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example Usage
def main():
    # Load a trained model (replace with your trained model path)
    from tensorflow.keras.models import load_model

    # Custom IoU and Dice coefficient functions should be defined before loading the model
    model = load_model('kvasir_colon_cancer_segmentation_unet.h5',
                       custom_objects={'iou_metric': iou_metric, 'dice_coefficient': dice_coefficient})

    # Use an example validation image for prediction
    idx = 0  # Index of the validation sample
    input_image = x_val[idx]  # Shape: (256, 256, 3)
    ground_truth = y_val[idx]  # Shape: (256, 256, 1)

    # Predict the segmentation mask
    raw_prediction = model.predict(input_image[np.newaxis, ...])[0, :, :, 0]  # Shape: (256, 256)

    # Apply post-processing
    refined_mask = advanced_post_process(raw_prediction, kernel_size=3, threshold=0.5)

    # Visualize the results
    visualize_advanced_post_process(input_image, raw_prediction, refined_mask)

# Run the script
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




