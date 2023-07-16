# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure each string is either "ODD"
# or "EVEN" (without the quotes) depending on whether the hexadecimal number in
# the image is odd or even. Take care not to make spelling or case mistakes. Make
# sure that the length of the list returned as output is the same as the number of
# filenames that were given as input. The judge may give unexpected results if this
# convention is not followed.
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers
import tensorflow as tf
tf.config.run_functions_eagerly(True)

def extract_background_color(image):
    corners = [(0, 0), (0, image.shape[0]-1), (image.shape[1]-1, 0), (image.shape[1]-1, image.shape[0]-1)]
    corner_colors = [image[y, x] for x, y in corners]
    background_color = np.mean(corner_colors, axis=0)
    return background_color

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 2: Eliminate Obfuscating Lines
def remove_obfuscating_lines(image, background_color):
    mask = cv2.inRange(image, background_color, background_color)
    result = cv2.bitwise_not(image, image, mask=mask)
    return result

# Step 3: Image Segmentation
def segment_image(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # filter out small segments
            segment = image[y:y+h, x:x+w]
            segments.append(segment)

    return segments

def decaptcha(filenames):
    image_directory = 'train/'  # Replace with the actual image directory path
    labels_file = 'train/labels.txt'  # Replace with the actual path to the labels file
    image_size = (500, 100)
    with open(labels_file, 'r') as file:
        labels = [line.strip() for line in file]
    images = []
    for i in range(2000):
        # Assuming the images have a '.jpg' extension
        filename = str(i) + '.png'
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)
        if image is not None and image.size != 0:
            image = cv2.resize(image, image_size)
            image = cv2.imread(image_path)
            background_color = extract_background_color(image)
            image_hsv = convert_to_hsv(image)
            image_without_lines = remove_obfuscating_lines(image_hsv, background_color)
            # Image Segmentation
            segments = segment_image(image_without_lines)
            images.append(image)
    # if len(images) != len(labels):
    #     print("Number of images and labels do not match.")
    #     exit()
    parities = []
    for label in labels:
        if label == 'ODD' or label == 'EVEN':
            parities.append(label)
        else:
            decimal_number = int(label, 16)
            if decimal_number % 2 == 0:
                parities.append('EVEN')
            else:
                parities.append('ODD')
    df = pd.DataFrame({'image': images, 'parity': parities})
    X_train, X_val, y_train, y_val = train_test_split(
        df['image'], df['parity'], test_size=0.2, random_state=42)
    X_train = np.array(X_train.tolist()) / 255.0
    X_val = np.array(X_val.tolist()) / 255.0
    label_dict = {'EVEN': 0, 'ODD': 1}
    y_train_encoded = np.array([label_dict[label] for label in y_train])
    y_val_encoded = np.array([label_dict[label] for label in y_val])
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(500, 100, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')  # 2 classes: Even and Odd
    ])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)
    X_train = np.transpose(X_train, (0, 2, 1, 3))
    X_val = np.transpose(X_val, (0, 2, 1, 3))
    history = model.fit(X_train, y_train_encoded, batch_size=32,
                        epochs=20, validation_data=(X_val, y_val_encoded))
    y_val_pred = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred, axis=1)
    val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
    val_precision = precision_score(
        y_val_encoded, y_val_pred, average='weighted')
    val_recall = recall_score(
        y_val_encoded, y_val_pred, average='weighted')
    val_f1_score = f1_score(y_val_encoded, y_val_pred, average='weighted')
    test_image_directory = ''  # Replace with the actual image directory path
    labels_file = 'test/labels.txt'
    test_images = []
    for i in filenames:
        filename = i
        image_path = os.path.join(test_image_directory, filename)
        image = cv2.imread(image_path)
        if image is not None and image.size != 0:
            image = cv2.resize(image, image_size)
            image = cv2.imread(image_path)
            background_color = extract_background_color(image)
            image_hsv = convert_to_hsv(image)
            image_without_lines = remove_obfuscating_lines(image_hsv, background_color)
            # Image Segmentation
            segments = segment_image(image_without_lines)
            test_images.append(image)
    test_labels = []
    with open(labels_file, 'r') as file:
        test_labels = [line.strip() for line in file]
    
    X_test = np.array(test_images) / 255.0
    X_test = np.transpose(X_test, (0, 2, 1, 3))
    label_dict = {'EVEN': 0, 'ODD': 1}
    y_test_encoded = np.array([label_dict[label] for label in test_labels])
    y_test_pred = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    test_precision = precision_score(
        y_test_encoded, y_test_pred, average='weighted')
    test_recall = recall_score(
        y_test_encoded, y_test_pred, average='weighted')
    test_f1_score = f1_score(
    y_test_encoded, y_test_pred, average='weighted')
    return ['ODD' if pred == 1 else 'EVEN' for pred in y_test_pred]