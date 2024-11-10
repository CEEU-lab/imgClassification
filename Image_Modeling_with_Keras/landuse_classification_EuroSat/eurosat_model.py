from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from keras_tuner import HyperParameters
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import tensorflow as tf
import rasterio

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import shutil
import numpy as np
from pathlib import Path
    
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


def adjust_brightness_contrast(image, label):
    # Ajusta brillo y contraste de la imagen
    image = tf.image.random_brightness(image, max_delta=0.2)  # Variación de brillo hasta ±20%
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Variación de contraste entre 0.8 y 1.2
    return image, label

def create_extended_generators_with_brightness_contrast(data_dir, img_size=(64, 64), batch_size=32, validation_split=0.2, test_split=0.1):
    """
    Creates training, validation, and test data generators with brightness and contrast adjustments for image datasets.
    """
    
    temp_val_dir = './temp_val_test_split'
    if not os.path.exists(temp_val_dir):
        os.makedirs(temp_val_dir)

    # Crear un generador de datos con aumentación para el entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,            
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split  # División para entrenamiento y validación
    )

    # Generador para validación y prueba sin aumentación
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split  
    )

    # Cargar conjunto de entrenamiento con aumentación
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',  
        shuffle=True,
        seed=42
    )

    # Cargar conjunto de validación sin aumentación
    val_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',  
        shuffle=False,
        seed=42
    )

    # Extraer rutas y etiquetas del conjunto de validación
    val_filepaths = val_generator.filepaths
    val_labels = val_generator.labels

    # Dividir en validación y prueba
    val_indices, test_indices = train_test_split(
        np.arange(len(val_filepaths)),
        test_size=test_split / (validation_split + test_split),
        random_state=42
    )

    # Separar las imágenes de validación y prueba en carpetas temporales
    for idx in val_indices:
        src = val_filepaths[idx]
        dst = os.path.join(temp_val_dir, 'validation', os.path.relpath(src, data_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for idx in test_indices:
        src = val_filepaths[idx]
        dst = os.path.join(temp_val_dir, 'test', os.path.relpath(src, data_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    # Crear generadores para validación y prueba sin aumentación
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

    # Convertir a tf.data y aplicar aumentación de brillo y contraste
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, train_generator.num_classes), dtype=tf.float32)
        )
    ).map(adjust_brightness_contrast).prefetch(buffer_size=tf.data.AUTOTUNE)  # Añadir prefetch para mayor velocidad

    # Aplicar cache para validación y prueba
    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_split_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, val_split_generator.num_classes), dtype=tf.float32)
        )
    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)  # Añadir cache y prefetch para evitar recargas

    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, test_generator.num_classes), dtype=tf.float32)
        )
    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)  # Añadir cache y prefetch para evitar recargas

    return train_dataset, val_dataset, test_dataset


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

# Build a rev CNN model
def build_cnn_model_rev(input_shape, num_classes):
    model = models.Sequential()

    # Define input layer explicitly
    model.add(layers.Input(shape=input_shape))

    # 1st Conv Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # 2nd Conv Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # 3rd Conv Layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # GlobalAveragePooling instead of Flatten
    model.add(layers.GlobalAveragePooling2D())

    # Fully connected layer
    model.add(layers.Dense(512, activation='relu'))

    # Output layer: softmax for multi-class classification
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Function to build the CNN model with enhanced regularization and dropout
def build_cnn_model_with_regularization(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    # Primera capa convolucional sin regularización (menos profunda)
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))  # Menor dropout en capas convolucionales

    # Segunda capa convolucional sin regularización (menos profunda)
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))

    # Tercera capa convolucional con regularización ligera (0.01)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Cuarta capa convolucional (más profunda) con regularización más fuerte (0.1)
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    
    # Aplanar las capas para pasar a capas densas
    model.add(layers.Flatten())

    # Capa densa con regularización fuerte y mayor dropout
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))  # Mayor dropout en la capa densa

    # Capa de salida para la clasificación
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Definir el plan de tasa de aprendizaje
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9
    )

    # Instanciar el optimizador con el plan de tasa de aprendizaje
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_cnn_model_lite(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),  # Capa densa más pequeña
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def visualize_weights(model, layer_index):
    layer = model.layers[layer_index]
    
    # Verificar si la capa es convolucional
    if 'conv' not in layer.name:
        print(f"La capa en el índice {layer_index} no es una capa convolucional.")
        return

    # Obtener los pesos de la capa especificada
    weights = layer.get_weights()[0]  # Tomar solo los pesos, sin los sesgos
    
    # Normalizar los pesos entre 0 y 1 para visualización
    min_w = np.min(weights)
    max_w = np.max(weights)
    weights = (weights - min_w) / (max_w - min_w)

    # Determinar el número de filtros en la capa
    num_filters = weights.shape[-1]
    num_channels = weights.shape[-2]  # Número de canales de entrada

    fig, axes = plt.subplots(num_channels, num_filters, figsize=(20, 20))

    # Visualizar cada filtro
    for i in range(num_filters):
        for j in range(num_channels):
            ax = axes[j, i] if num_channels > 1 else axes[i]
            ax.imshow(weights[:, :, j, i], cmap='viridis')
            ax.axis('off')
    plt.show()


from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def create_adapted_resnet(num_classes, input_shape=(64, 64, 4)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    # Congelar capas base
    for layer in base_model.layers:
        layer.trainable = False

    # Adaptar para 4 canales (RGB + MSAVI)
    inputs = layers.Input(shape=input_shape)  # input_shape=(64, 64, 4)
    x = layers.Conv2D(3, (1, 1))(inputs)      # Reducir de 4 a 3 canales
    x = base_model(x)

    # Añadir capas personalizadas
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Crear el modelo adaptado usando ResNet50 preentrenada
def create_resnet_transfer_model(input_shape=(224, 224, 4), num_classes=10):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Congelar capas de la base preentrenada inicialmente
    for layer in base_model.layers:
        layer.trainable = False

    # Adaptar para 4 canales (RGB + MSAVI)
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, (1, 1), activation='relu')(inputs)  # Reducir a 3 canales
    x = base_model(x)

    # Añadir capas personalizadas
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Fine-tuning gradual
def unfreeze_and_finetune(model, layers_to_unfreeze=10, learning_rate=1e-5):
    for layer in model.layers[-layers_to_unfreeze:]:
        layer.trainable = True

    # Compilar con un learning rate menor para fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])


from tensorflow.keras.callbacks import Callback

class ManualLearningRateScheduler(Callback):
    def __init__(self, schedule):
        super(ManualLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        # Check if the current epoch is in the schedule
        if epoch in self.schedule:
            new_lr = self.schedule[epoch]
            if isinstance(new_lr, (float, int)) and new_lr > 0:
                # Determine the learning rate attribute, depending on TensorFlow version
                if hasattr(self.model.optimizer, "learning_rate"):
                    lr_attr = self.model.optimizer.learning_rate
                elif hasattr(self.model.optimizer, "lr"):
                    lr_attr = self.model.optimizer.lr
                else:
                    print(f"Warning: Unable to set learning rate for optimizer: no learning rate attribute found.")
                    return

                # Set the learning rate only if lr_attr is a valid variable
                if isinstance(lr_attr, tf.Variable) or isinstance(lr_attr, tf.Tensor):
                    tf.keras.backend.set_value(lr_attr, float(new_lr))
                    print(f"\nEpoch {epoch+1}: Learning rate is set to {new_lr}")
                else:
                    print(f"Warning: Learning rate attribute {lr_attr} is not a valid Keras variable.")


def preprocess_image(img, label):
    # Redimensionar y normalizar en el rango [-1, 1] (por compatibilidad con ResNet50)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32)
    
    # Normalizar valores de los pixeles al rango [-1, 1]
    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img)) * 2 - 1
    
    return img, label


# Función para calcular MSAVI y combinarlo con RGB
def prepare_rgb_msavi(image):
    # Selecciona las bandas necesarias (ajusta los índices si es necesario)
    red = image[..., 2]  # Banda Roja
    nir = image[..., 7]  # Banda NIR (ajusta según tu archivo)

    # Calcular MSAVI
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2

    # Normalizar RGB entre 0 y 1 (si aún no lo has hecho)
    rgb = image[..., :3] / 255.0  # Escala RGB si es necesario

    # Concatenar RGB + MSAVI para obtener un tensor (64, 64, 4)
    rgb_msavi = np.concatenate([rgb, msavi[..., np.newaxis]], axis=-1)
    return rgb_msavi


from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import seaborn as sns

# Función para evaluar el modelo
def evaluate_model(model, X_test, y_test,classes):
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Convertir y_test de formato one-hot a etiquetas de clase única
    y_test_classes = np.argmax(y_test, axis=1)

    # Matriz de confusión
    cm = confusion_matrix(y_test_classes, y_pred_classes)

    # Coeficiente Kappa
    kappa = cohen_kappa_score(y_test_classes, y_pred_classes)

    # Reporte de clasificación
    class_report = classification_report(y_test_classes, y_pred_classes, target_names=classes)

    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(f"Kappa Score: {kappa}")
    print("\nClassification Report:")
    print(class_report)

# Modified evaluation function to handle batches directly from the dataset
def evaluate_model_in_batches(model, dataset, classes):
    y_pred = []
    y_true = []
    
    # Loop over batches in the dataset
    for images, labels in dataset:
        batch_preds = model.predict(images)  # Predict on batch
        y_pred.extend(np.argmax(batch_preds, axis=1))  # Predicted classes
        y_true.extend(np.argmax(labels.numpy(), axis=1))  # True classes (convert from one-hot if needed)
    
    # Convert lists to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Cohen's Kappa Score
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Classification report
    class_report = classification_report(y_true, y_pred, target_names=classes)
    
    # Display confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    print(f"Kappa Score: {kappa}")
    print("\nClassification Report:")
    print(class_report)



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


    # Función para cargar y preprocesar imágenes satelitales
def load_satellite_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read()
        # Normalizar y reordenar dimensiones
        image = np.transpose(image, (1, 2, 0))
        image = image.astype(np.float32) / 255.0
    return image

# Cargar dataset (ejemplo con EuroSAT)
def load_eurosat_dataset(root_dir, num_samples=1000):
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

    images = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        files = os.listdir(class_dir)[:num_samples // len(classes)]

        for file in files:
            img_path = os.path.join(class_dir, file)
            img = load_satellite_image(img_path)
            images.append(img)
            labels.append(class_idx)

    return np.array(images), np.array(labels)


# Modificar la función de carga para incluir MSAVI
def load_eurosat_dataset_with_msavi(root_dir, num_samples=1000):
    images = []
    labels = []

    # Lista de clases (necesaria para visualización)
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        files = os.listdir(class_dir)[:num_samples // len(classes)]

        for file in files:
            img_path = os.path.join(class_dir, file)
            img = load_satellite_image(img_path)
            img_with_msavi = prepare_rgb_msavi(img)  # Añadir MSAVI a RGB
            images.append(img_with_msavi)
            labels.append(class_idx)

    return np.array(images), np.array(labels)
