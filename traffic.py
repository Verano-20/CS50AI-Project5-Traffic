import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Check GPUs
    print(f"\nNum GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))} \n")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    for category in range(NUM_CATEGORIES):
        path = os.path.join(data_dir, str(category))
        for file in os.listdir(path):
            img_file = os.path.join(path, file)
            img = cv2.imread(img_file)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(category)
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Generate sequential model for CNN
    model = tf.keras.models.Sequential()

    # Define input shape
    model.add(tf.keras.Input(shape = (30, 30, 3)))

    # Convolution layer with 3 filters, using 3x3 kernel matrix
    model.add(tf.keras.layers.Conv2D(3, (3, 3), activation="relu"))

    # 2x2 Max Pooling 2D layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten Data from 3D to 1D
    model.add(tf.keras.layers.Flatten())

    # Hidden layer with 4 units
    model.add(tf.keras.layers.Dense(10, activation="relu"))

    # Hidden layer with 3 units
    model.add(tf.keras.layers.Dense(10, activation="relu"))

    # Output layer with NUM_CATEGORIES outputs
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))

    # Print structure of model
    model.summary()

    # Compile model for training
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    main()