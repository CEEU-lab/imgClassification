import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import cv2


# Initial config
def check_GPU_config():
    # Check available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU devices found. Ensure your system recognizes the GPU.")
    else:
        try:
            # Limit TensorFlow to use only the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"Configured TensorFlow to use GPU: {gpus[0].name}")

            # Enable dynamic memory growth on the GPU
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("Memory growth enabled for the first GPU.")

            # Optional: Display additional GPU configuration details
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")

        except RuntimeError as e:
            print(f"RuntimeError during GPU setup: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # Further GPU diagnostics
    print("TensorFlow version:", tf.__version__)
    print("CUDA device detected:", tf.test.is_built_with_cuda())
    print("GPU availability:", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    return gpus

## 1. DATA PREP
# Parámetros globales
IMAGE_SIZE = 512

def read_image(path, target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode="rgb"):
    """
    Lee y procesa una imagen.
    
    Args:
    - path: Ruta de la imagen.
    - target_size: Tamaño de la imagen de salida (default: (512, 512)).
    - color_mode: "rgb" o "grayscale".
    
    Returns:
    - img: Imagen procesada.
    """
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    if color_mode == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode == "grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def plot_images_and_masks(image_dir, mask_dir, file_list, rows=3, cols=3, figsize=(10, 10)):
    """
    Muestra imágenes y sus máscaras correspondientes.

    Args:
    - image_dir: Directorio con las imágenes.
    - mask_dir: Directorio con las máscaras.
    - file_list: Lista de nombres de archivos a mostrar.
    - rows: Número de filas del grid.
    - cols: Número de columnas del grid.
    - figsize: Tamaño de la figura.
    """
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(ax.flat):
        if i < len(file_list):
            # Mostrar imagen
            img_path = os.path.join(image_dir, file_list[i])
            img = read_image(img_path, color_mode="rgb")
            ax.imshow(img)
            ax.set_title(f"Image: {file_list[i]}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(ax.flat):
        if i < len(file_list):
            # Mostrar máscara
            mask_path = os.path.join(mask_dir, file_list[i])
            if os.path.exists(mask_path):
                mask = read_image(mask_path, color_mode="grayscale")
                ax.imshow(mask, cmap="gray")
                ax.set_title(f"Mask: {file_list[i]}")
            else:
                ax.set_title("Mask not found")
            ax.axis("off")
    
    plt.tight_layout()
    plt.show()


def plot_images_and_masks(image_dir, mask_dir, file_list, rows=3, cols=3, figsize=(10, 10)):
    """
    Muestra imágenes y sus máscaras correspondientes.

    Args:
    - image_dir: Directorio con las imágenes.
    - mask_dir: Directorio con las máscaras.
    - file_list: Lista de nombres de archivos a mostrar.
    - rows: Número de filas del grid.
    - cols: Número de columnas del grid.
    - figsize: Tamaño de la figura.
    """
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(ax.flat):
        if i < len(file_list):
            # Mostrar imagen
            img_path = os.path.join(image_dir, file_list[i])
            img = read_image(img_path, color_mode="rgb")
            ax.imshow(img)
            ax.set_title(f"Image: {file_list[i]}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(ax.flat):
        if i < len(file_list):
            # Mostrar máscara
            mask_path = os.path.join(mask_dir, file_list[i])
            if os.path.exists(mask_path):
                mask = read_image(mask_path, color_mode="grayscale")
                ax.imshow(mask, cmap="gray")
                ax.set_title(f"Mask: {file_list[i]}")
            else:
                ax.set_title("Mask not found")
            ax.axis("off")
    
    plt.tight_layout()
    plt.show()

def plot_train_images_and_masks(train_image_dir, train_mask_dir, random_files, rows=3, cols=3):
    print("Train Images and Masks:")
    plot_images_and_masks(train_image_dir, train_mask_dir, random_files, rows, cols)

def plot_val_images_and_masks(val_image_dir, val_mask_dir, random_files, rows=3, cols=3):
    print("Validation Images and Masks:")
    plot_images_and_masks(val_image_dir, val_mask_dir, random_files, rows, cols)

def plot_test_images_and_masks(test_image_dir, test_mask_dir, random_files, rows=3, cols=3):
    print("Test Images and Masks:")
    plot_images_and_masks(test_image_dir, test_mask_dir, random_files, rows, cols)

def read_and_preprocess_images(image_dir, mask_dir, image_size=512):
    """
    Lee y preprocesa imágenes y máscaras desde un directorio.

    Args:
        image_dir (str): Ruta al directorio de imágenes.
        mask_dir (str): Ruta al directorio de máscaras.
        image_size (int): Tamaño de las imágenes (asume imágenes cuadradas).

    Returns:
        tuple: Arrays numpy de imágenes y máscaras.
    """
    image_list = os.listdir(image_dir)
    images = []
    masks = []

    for img_file in image_list:
        try:
            # Leer y procesar la imagen
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

            # Leer y procesar la máscara
            mask_file = img_file
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (image_size, image_size))
            mask = np.expand_dims(mask, axis=-1)
            masks.append(mask)
        except Exception as e:
            print(f"Error procesando {img_file}: {e}")
            continue

    # Convertir a arrays numpy
    images = np.array(images) / 255.0  # Normalizar imágenes
    masks = (np.array(masks) > 0).astype("float32")  # Binarizar máscaras
    return images, masks

'''
def calculate_class_balance(masks):
    """
    Calcula la proporción de píxeles de cada clase en un conjunto de máscaras.

    Args:
        masks (numpy array): Array de máscaras con forma (num_samples, height, width, 1).

    Returns:
        dict: Porcentaje de píxeles para cada clase.
    """
    total_pixels = np.prod(masks.shape)
    class_counts = np.sum(masks, axis=(0, 1, 2))  # Suma de píxeles de la clase 1
    background_count = total_pixels - class_counts  # Píxeles de fondo (clase 0)
    return {
        "background": background_count / total_pixels,
        "buildings": class_counts / total_pixels
    }
'''
def calculate_class_balance(masks):
    """
    Calcula el balance de clases en las máscaras de entrenamiento.
    
    Args:
        masks: Numpy array con las máscaras (shape: [num_samples, height, width, 1]).

    Returns:
        class_ratios: Un diccionario con las proporciones de cada clase (fondo y edificios).
    """
    total_pixels = masks.size
    background_pixels = (masks == 0).sum()
    building_pixels = (masks == 1).sum()
    
    return {
        'background': background_pixels / total_pixels,
        'buildings': building_pixels / total_pixels
    }

# Asegurar que las máscaras sean binarias después de las augmentaciones
def binary_mask_postprocess(mask_batch):
    return (mask_batch > 0.5).astype("float32")  # Reconvierte a binario después de las augmentaciones

def data_generator(datagen, images, masks, batch_size):
    gen = datagen.flow(images, batch_size=batch_size, seed=42)
    mask_gen = datagen.flow(masks, batch_size=batch_size, seed=42)
    for img_batch, mask_batch in zip(gen, mask_gen):
        mask_batch = binary_mask_postprocess(mask_batch)  # Reconvierte las máscaras a binarias
        yield img_batch, mask_batch


#### 2. ARQUITECTURA DEL MODELO ####

def conv_block(input, num_filters):
    conv = tf.keras.layers.Conv2D(
        num_filters, 3, padding="same", kernel_regularizer=l2(0.01)
    )(input)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation("relu")(conv)
    conv = tf.keras.layers.Conv2D(
        num_filters, 3, padding="same", kernel_regularizer=l2(0.01)
    )(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation("relu")(conv)
    return conv

def conv_block_with_dropout(input, num_filters):
    conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.01))(input)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation("relu")(conv)
    conv = tf.keras.layers.Dropout(0.5)(conv)  # Aumenta el dropout
    conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.01))(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation("relu")(conv)
    return conv

def encoder_block(input, num_filters):
    skip = conv_block_with_dropout(input, num_filters) # or try conv_block
    pool = tf.keras.layers.MaxPool2D((2, 2))(skip)
    return skip, pool

def decoder_block(input, skip, num_filters):
    up_conv = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    conv = tf.keras.layers.Concatenate()([up_conv, skip])
    conv = conv_block_with_dropout(conv, num_filters) # or try conv_block
    return conv

# Mas profunda
def Unet(input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    skip1, pool1 = encoder_block(inputs, 32)   # Cambiado de 64 a 32
    skip2, pool2 = encoder_block(pool1, 64)   # Cambiado de 128 a 64
    skip3, pool3 = encoder_block(pool2, 128)  # Cambiado de 256 a 128
    skip4, pool4 = encoder_block(pool3, 256)  # Cambiado de 512 a 256

    # Bridge
    bridge = conv_block(pool4, 512)  # Cambiado de 1024 a 512

    # Decoder
    decode1 = decoder_block(bridge, skip4, 256)
    decode2 = decoder_block(decode1, skip3, 128)
    decode3 = decoder_block(decode2, skip2, 64)
    decode4 = decoder_block(decode3, skip1, 32)

    # Output
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(decode4)
    model = tf.keras.models.Model(inputs, outputs, name="U-Net")
    return model

# Menos profunda
def Unet_optimized(input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    def encoder_block(input, num_filters):
        conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same", activation="relu")(input)
        conv = tf.keras.layers.Dropout(0.3)(conv)
        conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same", activation="relu")(conv)
        pool = tf.keras.layers.MaxPool2D((2, 2))(conv)
        return conv, pool

    # Decoder
    def decoder_block(input, skip, num_filters):
        upsample = tf.keras.layers.Conv2DTranspose(num_filters, 3, strides=2, padding="same")(input)
        concat = tf.keras.layers.Concatenate()([upsample, skip])
        conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same", activation="relu")(concat)
        conv = tf.keras.layers.Dropout(0.3)(conv)
        conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same", activation="relu")(conv)
        return conv

    # Encoder
    skip1, pool1 = encoder_block(inputs, 32)
    skip2, pool2 = encoder_block(pool1, 64)
    skip3, pool3 = encoder_block(pool2, 128)
    skip4, pool4 = encoder_block(pool3, 256)

    # Bridge
    bridge = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(pool4)
    bridge = tf.keras.layers.Dropout(0.3)(bridge)
    bridge = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(bridge)

    # Decoder
    decode1 = decoder_block(bridge, skip4, 256)
    decode2 = decoder_block(decode1, skip3, 128)
    decode3 = decoder_block(decode2, skip2, 64)
    decode4 = decoder_block(decode3, skip1, 32)

    # Output
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(decode4)
    model = tf.keras.models.Model(inputs, outputs, name="U-Net_Optimized")
    return model

#### 3. CONFIGURACION DEL MODELO ####

# Model wrapper
def configure_model(input_shape=(512, 512, 3), 
                    use_lr_schedule=True, 
                    train_size=None, 
                    loss_function=None):
    """
    Configura el modelo U-Net y define los callbacks.

    Args:
        input_shape (tuple): Forma de entrada de las imágenes.
        use_lr_schedule (bool): Si usa un scheduler de learning rate.
        train_size (int): Número de ejemplos en el conjunto de entrenamiento.
        loss_function (callable): Función de pérdida a usar. Si None, utiliza una pérdida por defecto.

    Returns:
        tuple: Modelo U-Net y callbacks.
    """
    # Crear el modelo
    model = Unet_optimized(input_shape)  # arquitectura menos profunda

    # Configuración de callbacks comunes
    early_stopping = EarlyStopping(
        monitor='val_iou_coeff', patience=10, mode='max', restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        filepath='best_model.keras', monitor='val_loss', save_best_only=True, verbose=1
    )
    callbacks = [early_stopping, PrintLearningRate(), model_checkpoint]

    # Configuración del optimizer y scheduler
    if use_lr_schedule:
        print("COMPILING MODEL WITH LR SCHEDULER")
        if train_size is None:
            raise ValueError("Debe proporcionar 'train_size' para calcular el decaimiento del learning rate.")

        from tensorflow.keras.optimizers.schedules import CosineDecay

        # Configuración del scheduler de learning rate
        lr_schedule = CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=50 * train_size // 4 * 50,  # TODO: ajustar según necesidades
            alpha=0.1
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        print("COMPILING MODEL WITHOUT LR SCHEDULER")
        optimizer = tf.keras.optimizers.Adam()

        # Añadir callbacks específicos para el caso sin scheduler
        reduce_lr = ReduceLROnPlateau(
            monitor='val_iou_coeff',  # O 'val_loss'
            factor=0.5,              # Reduce el learning rate a la mitad
            patience=5,              # Espera 5 épocas antes de reducir el LR
            min_lr=1e-6              # Learning rate mínimo
        )
        dynamic_beta_callback = DynamicBetaCallback(initial_beta=2, decay_rate=0.1)
        callbacks.extend([reduce_lr, dynamic_beta_callback])

    # Configuración de la función de pérdida
    if loss_function is None:
        print("Using default combined_focal_dice_loss")
        loss_function = lambda y_true, y_pred: combined_focal_dice_loss(y_true, y_pred, beta=2.0)

    # Compilar el modelo
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy', iou_coeff, f1_score]
    )

    return model, callbacks


# Funciones de pérdida
from tensorflow.keras.losses import BinaryCrossentropy

def get_weighted_loss(class_weights):
    """
    Devuelve una función de pérdida ponderada basada en los pesos de las clases.

    Args:
        class_weights (dict): Pesos para cada clase (ej: {0: peso_fondo, 1: peso_edificios}).

    Returns:
        weighted_loss: Función de pérdida ponderada.
    """
    loss = BinaryCrossentropy(from_logits=False)

    def weighted_loss(y_true, y_pred):
        # Pérdida para cada clase
        loss_bg = class_weights[0] * loss(y_true * (1 - y_pred), y_pred * (1 - y_true))
        loss_buildings = class_weights[1] * loss(y_true * y_pred, y_pred * y_true)
        return loss_bg + loss_buildings

    return weighted_loss


def custom_loss(y_true, y_pred):
    # Binary Crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)  # Reducción total para que sea escalar
    #print("Forma después de binary_crossentropy:", bce.shape)

    # Dice Loss
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = 1 - numerator / (denominator + tf.keras.backend.epsilon())
    dice = tf.reduce_mean(dice)  # Reducción total para que sea escalar
    #print("Forma de dice loss:", dice.shape)

    # Combinar pérdidas
    return bce + dice

def weighted_binary_crossentropy(y_true, y_pred, alpha=0.5):
    """
    y_true: Tensor de máscaras verdaderas.
    y_pred: Tensor de predicciones.
    alpha: Peso que ajusta la importancia de los píxeles activos (positivos).
           Valores típicos: 0.5 <= alpha <= 1.0.
    """
    # Calcula la proporción de píxeles activos en la máscara
    proportion = tf.reduce_mean(y_true)

    # Ajusta el peso de los píxeles activos con un límite superior
    weight = tf.clip_by_value(alpha / (proportion + tf.keras.backend.epsilon()), 1.0, 10.0)

    # Calcula la pérdida de BCE ponderada
    loss = -1 * (
        weight * y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()) +
        (1 - alpha) / (1 - proportion + tf.keras.backend.epsilon()) *
        (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
    )

    return tf.reduce_mean(loss)

def combined_weighted_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    """
    y_true: Tensor de máscaras verdaderas.
    y_pred: Tensor de predicciones.
    alpha: Peso para ajustar la importancia de los píxeles activos en BCE.
    beta: Peso relativo de Dice Loss.
    """
    # Calcula la pérdida ponderada de BCE
    w_bce = weighted_binary_crossentropy(y_true, y_pred, alpha)

    # Calcula Dice Loss
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = 1 - numerator / (denominator + tf.keras.backend.epsilon())
    dice = tf.reduce_mean(dice)  # Reducción total

    # Combina ambas pérdidas
    return w_bce + beta * dice

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_exp = tf.exp(-bce)
        focal = alpha * (1 - bce_exp) ** gamma * bce
        return tf.reduce_mean(focal)
    return loss

# Función dinámica para ajustar beta
def dynamic_beta(epoch, initial_beta=2, decay_rate=0.1):
    return initial_beta / (1 + decay_rate * epoch)

# Modifica la pérdida para aceptar beta dinámico
class DynamicBetaCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_beta=2, decay_rate=0.1):
        self.initial_beta = initial_beta
        self.decay_rate = decay_rate

    def on_epoch_begin(self, epoch, logs=None):
        global beta
        beta = dynamic_beta(epoch, self.initial_beta, self.decay_rate)

# Actualiza la función de pérdida para usar la variable global `beta`
def combined_focal_dice_loss(y_true, y_pred, beta=2.0):
    focal = focal_loss()(y_true, y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = 1 - numerator / (denominator + tf.keras.backend.epsilon())
    dice = tf.reduce_mean(dice)
    return focal + beta * dice

def combined_focal_iou_loss(y_true, y_pred):
    # Focal Loss
    alpha, gamma = 0.25, 2
    focal_loss = -alpha * (1 - y_pred) ** gamma * y_true * tf.math.log(y_pred + 1e-8)
    focal_loss += -(1 - alpha) * y_pred ** gamma * (1 - y_true) * tf.math.log(1 - y_pred + 1e-8)

    # IoU Loss
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou_loss = 1 - intersection / (union + 1e-8)

    return tf.reduce_mean(focal_loss) + iou_loss

# Callback para imprimir el learning rate en cada época
class PrintLearningRate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        print(f"\nLearning rate for epoch {epoch + 1} is {lr.numpy()}")

def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    f1 = 2 * tp / (2 * tp + fp + fn + tf.keras.backend.epsilon())
    return f1

def iou_coeff(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Binariza predicciones
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = intersection / (union + tf.keras.backend.epsilon())
    return tf.reduce_mean(iou)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return 1 - numerator / (denominator + tf.keras.backend.epsilon())

#### 3. EVALUACION ####

def plot_training_history(history):
    """
    Plots model training history
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_acc.plot(history.epoch, history.history["iou_coeff"], label="Train iou")
    ax_acc.plot(history.epoch, history.history["val_iou_coeff"], label="Validation iou")
    ax_acc.legend()

def visualize_predictions(model, images, masks, num_examples=3):
    """
    Visualiza un conjunto de predicciones del modelo.

    Args:
        model: El modelo entrenado.
        images: Imágenes de entrada.
        masks: Máscaras verdaderas.
        num_examples: Número de ejemplos a visualizar.
    """
    predictions = model.predict(images[:num_examples])
    plt.figure(figsize=(15, 5))
    
    for i in range(num_examples):
        # Imagen de entrada
        plt.subplot(3, num_examples, i + 1)
        plt.imshow(images[i])
        plt.title("Input Image")
        plt.axis("off")

        # Máscara verdadera
        plt.subplot(3, num_examples, i + 1 + num_examples)
        plt.imshow(masks[i].squeeze(), cmap="gray")
        plt.title("True Mask")
        plt.axis("off")

        # Máscara predicha
        plt.subplot(3, num_examples, i + 1 + 2 * num_examples)
        plt.imshow(predictions[i].squeeze(), cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def show_result(idx, og, unet, target, p):

    fig, axs = plt.subplots(1, 3, figsize=(12,12))
    axs[0].set_title("Original "+str(idx) )
    axs[0].imshow(og)
    axs[0].axis('off')

    axs[1].set_title("U-Net: p>"+str(p))
    axs[1].imshow(unet)
    axs[1].axis('off')

    axs[2].set_title("Ground Truth")
    axs[2].imshow(target)
    axs[2].axis('off')

    plt.show()


def visualize_predictions_with_thresholds(test_images, test_masks, predictions, thresholds):
    """
    Visualiza las predicciones del modelo para varios umbrales.

    Args:
    - test_images (numpy.ndarray): Imágenes de prueba.
    - test_masks (numpy.ndarray): Máscaras reales de prueba.
    - predictions (numpy.ndarray): Predicciones del modelo.
    - thresholds (list of float): Lista de umbrales para binarizar las predicciones.
    """
    show_test_idx = random.sample(range(len(predictions)), 3)

    for idx in show_test_idx:
        print(f"Mostrando resultados para índice: {idx}")
        for threshold in thresholds:
            # Binariza las predicciones según el umbral
            unet_binarized = (predictions[idx] > threshold).astype(np.uint8)
            # Muestra los resultados
            show_result(idx, test_images[idx], unet_binarized, test_masks[idx], threshold)
        print()
