#To create a program that identifies facial expressions using GenAI on Intel AI laptops, we can follow a structured approach:
#Step 1: Data Preparation -


import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset (e.g., FER2013)
def load_dataset(data_path):
    images = []
    labels = []
    for label in os.listdir(data_path):
        for image_file in os.listdir(os.path.join(data_path, label)):
            image_path = os.path.join(data_path, label, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (48, 48))
            images.append(image)
            labels.append(int(label))
    return np.array(images), np.array(labels)

data_path = 'path_to_your_dataset'
images, labels = load_dataset(data_path)

# Normalize images
images = images / 255.0
images = np.expand_dims(images, -1)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
