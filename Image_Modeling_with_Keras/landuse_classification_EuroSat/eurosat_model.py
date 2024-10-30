from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from keras_tuner import HyperParameters
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam, RMSprop, SGD, AdamW, Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import shutil
import numpy as np
from pathlib import Path

# mail Santiago: nunez.rimedio@gmail.com    
# https://poloclub.github.io/cnn-explainer/


def count_images_in_subdirectories(parent_directory, subdirectories):
    """
    Counts the total number of images in the specified subdirectories of a parent directory.

    :param parent_directory: The main directory containing the subdirectories.
    :param subdirectories: List of subdirectory names to search within.
    :return: Total count of image files across all specified subdirectories.
    """
    # Define common image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif'}

    total_count = 0

    # Iterate over each specified subdirectory
    for subdirectory in subdirectories:
        path = Path(parent_directory) / subdirectory
        # Count the image files with matching extensions
        count = sum(1 for file in path.rglob('*') if file.suffix.lower() in image_extensions)
        print(f"Number of images in '{subdirectory}': {count}")
        total_count += count

    return total_count


# Function to plot images from a dataset
def plot_images(images, labels, class_names, num_images=10):
    """
    Plots a grid of random images with their corresponding labels.
    
    :param images: Array of images.
    :param labels: Array of corresponding labels.
    :param class_names: List of class names corresponding to the label indices.
    :param num_images: Number of images to display.
    """
    # Randomly select num_images indices
    indices = np.random.choice(len(images), num_images, replace=False)
    
    plt.figure(figsize=(20, 10))
    
    for i, idx in enumerate(indices):
        ax = plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(images[idx])
        plt.title(f"Class: {class_names[labels[idx]]}")
        plt.axis("off")
    
    plt.show()


# Function to create the data generators
def create_data_generators(data_dir, img_size=(64, 64), batch_size=32):
    datagen_aug = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255,
        validation_split=0.2
    )

    train_generator_aug = datagen_aug.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen_aug.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    class_names = list(train_generator_aug.class_indices.keys())
    return train_generator_aug, val_generator, class_names


def create_extended_generators(data_dir, img_size=(64, 64), batch_size=32, validation_split=0.2, test_split=0.1):
    """
    Creates training, validation, and test data generators for image datasets.

    :param data_dir: Path to the directory containing the dataset.
    :param img_size: Tuple specifying the target image size (default is (64, 64)).
    :param batch_size: Batch size for the generators (default is 32).
    :param validation_split: Fraction of data to reserve for validation (default is 0.2).
    :param test_split: Fraction of validation data to reserve for testing (default is 0.1).
    :return: Tuple containing training, validation, and test data generators.
    """
    
    # Create a temporary directory to hold validation and test split
    temp_val_dir = './temp_val_test_split'
    if not os.path.exists(temp_val_dir):
        os.makedirs(temp_val_dir)

    # Create a data generator with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],  # New augmentation technique
        channel_shift_range=0.2,      # New augmentation technique
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split  # Split for training and validation
    )

    # Use a separate data generator without augmentation for validation and test data
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split  # Use the same split for validation
    )

    # Load the training set with augmentation
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',  # Load training data
        shuffle=True,
        seed=42
    )

    # Load the validation set without augmentation
    val_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',  # Load validation data
        shuffle=False,
        seed=42
    )

    # Extract file paths and labels for validation data
    val_filepaths = val_generator.filepaths
    val_labels = val_generator.labels

    # Split validation data into validation and test sets
    val_indices, test_indices = train_test_split(
        np.arange(len(val_filepaths)),
        test_size=test_split / (validation_split + test_split),
        random_state=42
    )

    # Copy validation images to temporary directory for split processing
    for idx in val_indices:
        src = val_filepaths[idx]
        dst = os.path.join(temp_val_dir, 'validation', os.path.relpath(src, data_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    # Copy test images to temporary directory for split processing
    for idx in test_indices:
        src = val_filepaths[idx]
        dst = os.path.join(temp_val_dir, 'test', os.path.relpath(src, data_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    # Create validation and test generators without augmentation
    val_split_generator = test_datagen.flow_from_directory(
        os.path.join(temp_val_dir, 'validation'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(temp_val_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    return train_generator, val_split_generator, test_generator

# Build a simple CNN model
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # Define input layer explicitly
    model.add(layers.Input(shape=input_shape))

    # 1st Conv Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 2nd Conv Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 3rd Conv Layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output from the conv layers
    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(512, activation='relu'))

    # Output layer: softmax for multi-class classification
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
                  metrics=['accuracy'])

    return model


# Function to build the CNN model with regularization
def build_cnn_model_with_regularization(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(num_classes, activation='softmax'))

    # Define the learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9
    )

    # Instantiate the optimizer with the learning rate schedule
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

# Function to build the model with adjustable hyperparameters
def build_model_with_hp(hp):
    '''
    Function to build a dynamic network architecture with 
    hyperparameters tuning using [hp] keras tuner object 
    inside the RandomSearch iterator class
    '''
    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))

    # Define kernel size options and map them using indices
    kernel_size_options = {0: (3, 3), 1: (5, 5)}
    kernel_size_choice = hp.Choice('kernel_size_index', values=[0, 1])
    selected_kernel_size = kernel_size_options[kernel_size_choice]

    # Define pool size options and map them using indices
    pool_size_options = {0: (2, 2), 1: (3, 3)}
    pool_size_choice = hp.Choice('pool_size_index', values=[0, 1])
    selected_pool_size = pool_size_options[pool_size_choice]

    # Define the number of convolutional layers
    num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=3)
    for i in range(num_conv_layers):
        model.add(layers.Conv2D(
            filters=hp.Int(f'conv{i+1}_filters', min_value=32, max_value=256, step=32),
            kernel_size=selected_kernel_size,
            activation=hp.Choice(f'conv{i+1}_activation', values=['relu', 'elu']),
            kernel_regularizer=regularizers.l2(hp.Choice('l2_reg', [0.0001, 0.001, 0.01]))
        ))
        model.add(layers.BatchNormalization(momentum=hp.Float('bn_momentum', 0.85, 0.99, step=0.01)))
        model.add(layers.MaxPooling2D(pool_size=selected_pool_size))
        if hp.Boolean(f'use_dropout{i+1}'):
            model.add(layers.Dropout(hp.Float(f'dropout_rate{i+1}', 0.1, 0.5, step=0.1)))

    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=256, max_value=512, step=64),
        activation=hp.Choice('dense_activation', values=['relu', 'tanh']),
        kernel_regularizer=regularizers.l2(hp.Choice('l2_reg', [0.0001, 0.001, 0.01]))
    ))
    model.add(layers.BatchNormalization())

    # Output Layer
    model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for EuroSAT

    # Compile Model with Tuned Optimizer
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')

    # Instantiate the optimizer based on choice
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Function to train the model
def train_model(model, n_epochs, n_patience, train_generator, val_generator):
    # Use only EarlyStopping when using LearningRateSchedule
    early_stopping = EarlyStopping(monitor='val_loss', patience=n_patience, restore_best_weights=True)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=n_epochs,
        callbacks=[early_stopping]  # Only use callbacks that do not alter the learning rate
    )

    return history

# Separate function to smooth curves
def smooth_curve(points, factor=0.8):
    """
    Applies exponential smoothing to a list of points.

    :param points: List of data points to smooth.
    :param factor: Smoothing factor (0 to 1). Higher values result in smoother curves.
    :return: List of smoothed points.
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# Function to plot training history with optional smoothing
def plot_training_history(history, smooth=False, smoothing_factor=0.8):
    """
    Plots training and validation accuracy and loss from the training history.

    :param history: History object from model training (output of model.fit()).
    :param smooth: Whether to smooth the curves for better visualization of trends.
    :param smoothing_factor: Factor used for smoothing; higher values make curves smoother.
    """
    # Extract accuracy and loss values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Apply smoothing if enabled
    if smooth:
        acc = smooth_curve(acc, factor=smoothing_factor)
        val_acc = smooth_curve(val_acc, factor=smoothing_factor)
        loss = smooth_curve(loss, factor=smoothing_factor)
        val_loss = smooth_curve(val_loss, factor=smoothing_factor)
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.show()


def visualize_augmented_images(X_train_sample, y_train_sample, index, datagen_aug, class_names, num_augments=9):
    """
    Visualizes an original image and its augmented versions, with pixel range checks and normalization.

    :param X_train_sample: Batch of images from the data generator.
    :param y_train_sample: Corresponding labels for the batch of images.
    :param index: Index of the image in the batch to visualize and augment.
    :param datagen_aug: ImageDataGenerator to apply augmentations.
    :param class_names: List of class names corresponding to the label indices.
    :param num_augments: Number of augmented images to visualize.
    """

    # Take the image at the given index from the batch
    sample_image = X_train_sample[index]

    # Normalize the original sample image to range [0, 1]
    sample_image_norm = sample_image / np.max(sample_image)

    # Visualize the manually normalized original sample image
    plt.figure(figsize=(5, 5))
    plt.imshow(sample_image_norm, vmin=0, vmax=1)  # Ensure it's normalized for display
    plt.title(f"Original Image (Class: {class_names[np.argmax(y_train_sample[index])]})")
    plt.axis('off')
    plt.show()

    # Generate augmented images from the sample image
    it = datagen_aug.flow(sample_image.reshape((1, 64, 64, 3)))  # Reshape to match input format for the generator

    # Check and print pixel values range for the first augmentation
    for i in range(1):
        batch = next(it)
        image = batch[0]
        print(f"Image {i+1} pixel range: min={image.min()}, max={image.max()}")  # Check pixel value range

    # Visualize multiple augmented images with manual normalization
    plt.figure(figsize=(10, 10))
    for i in range(num_augments):
        plt.subplot(3, 3, i + 1)
        batch = next(it)  # Generate a new augmented image
        image = batch[0]
        image_norm = image / np.max(image)  # Normalize to the range [0, 1] for visualization
        plt.imshow(image_norm, vmin=0, vmax=1)  # Display the normalized image
        plt.axis('off')
    plt.suptitle(f"Manually Normalized Augmented Images (Original Class: {class_names[np.argmax(y_train_sample[index])]})")
    plt.show()
