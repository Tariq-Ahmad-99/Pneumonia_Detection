import os
import shutil
import random
import numpy as np
import pandas as pd
import cv2
import skimage
import matplotlib.pyplot as plt
import skimage.segmentation
import seaborn as sns


# Labels and image size
labels = ["PNEUMONIA", "NORMAL"]
img_size = 128

# Create a function to load the data
def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)  # Path to each class folder
        class_num = labels.index(label)      # 0 for PNEUMONIA, 1 for NORMAL
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                # Read the image in grayscale
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # Resize the image to a fixed size
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                # Append the processed image and its label
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return data

# Load data for train, test, and validation sets
train_data = get_data("data/train")
test_data = get_data("data/test")
val_data = get_data("data/val")

# Extract images and labels separately
train_images = np.array([item[0] for item in train_data])  # Training images
train_labels = np.array([item[1] for item in train_data])  # Training labels

""" # Visualize class distribution
label_names = ["Pneumonia" if label == 0 else "Normal" for label in train_labels]
sns.countplot(x=label_names)
plt.title("Class Distribution in Training Data")
plt.show() """
