# ARQUITECTURAS
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, 
    Concatenate, BatchNormalization, Activation, Dropout, Add, Conv2DTranspose
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50, EfficientNetB0

# Bloque convolucional 
def conv_block_unet(filters, input_tensor, kernel_regularizer=None):
    block = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(input_tensor)
    block = BatchNormalization()(block)
    block = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(block)
    block = BatchNormalization()(block)
    return block

# U-Net 
def unet(input_size=(128, 128, 3), num_classes=1, l2_lambda=1e-4):
    inputs = Input(shape=input_size)

    # Encoder 
    conv1 = conv_block_unet(128, inputs, kernel_regularizer=l2(l2_lambda))  # 128 filtros
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block_unet(256, pool1, kernel_regularizer=l2(l2_lambda))  # 256 filtros
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block_unet(512, pool2, kernel_regularizer=l2(l2_lambda))  # 512 filtros
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block_unet(1024, pool3, kernel_regularizer=l2(l2_lambda))  # 1024 filtros
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck 
    conv5 = conv_block_unet(2048, pool4, kernel_regularizer=l2(l2_lambda))  # 2048 filtros
    drop5 = Dropout(0.5)(conv5)

    # Decoder
    up6 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(drop5)  # 1024 filtros
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = conv_block_unet(1024, merge6, kernel_regularizer=l2(l2_lambda))

    up7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv6)  # 512 filtros
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = conv_block_unet(512, merge7, kernel_regularizer=l2(l2_lambda))

    up8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7)  # 256 filtros
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = conv_block_unet(256, merge8, kernel_regularizer=l2(l2_lambda))

    up9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8)  # 128 filtros
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = conv_block_unet(128, merge9, kernel_regularizer=l2(l2_lambda))

    # Output layer
    conv10 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

# U-Net ++
def conv_block_uplus(x, filters, kernel_size=(3, 3), activation="relu", kernel_regularizer=None):
    x = Conv2D(filters, kernel_size, padding="same", kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, padding="same", kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def unet_plus_plus(input_size=(128, 128, 3), num_classes=1, l2_lambda=1e-4):
    inputs = Input(input_size)

    # Encoder
    x1_0 = conv_block_uplus(inputs, 64, kernel_regularizer=l2(l2_lambda))
    p1 = MaxPooling2D(pool_size=(2, 2))(x1_0)

    x2_0 = conv_block_uplus(p1, 128, kernel_regularizer=l2(l2_lambda))
    p2 = MaxPooling2D(pool_size=(2, 2))(x2_0)

    x3_0 = conv_block_uplus(p2, 256, kernel_regularizer=l2(l2_lambda))
    p3 = MaxPooling2D(pool_size=(2, 2))(x3_0)

    x4_0 = conv_block_uplus(p3, 512, kernel_regularizer=l2(l2_lambda))
    p4 = MaxPooling2D(pool_size=(2, 2))(x4_0)

    x5_0 = conv_block_uplus(p4, 1024, kernel_regularizer=l2(l2_lambda))

    # Decoder (dense connections)
    x4_1 = conv_block_uplus(concatenate([x4_0, UpSampling2D(size=(2, 2))(x5_0)]), 512, kernel_regularizer=l2(l2_lambda))
    x3_2 = conv_block_uplus(concatenate([x3_0, UpSampling2D(size=(2, 2))(x4_1)]), 256, kernel_regularizer=l2(l2_lambda))
    x2_3 = conv_block_uplus(concatenate([x2_0, UpSampling2D(size=(2, 2))(x3_2)]), 128, kernel_regularizer=l2(l2_lambda))
    x1_4 = conv_block_uplus(concatenate([x1_0, UpSampling2D(size=(2, 2))(x2_3)]), 64, kernel_regularizer=l2(l2_lambda))

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation="sigmoid")(x1_4)

    return Model(inputs, outputs)

# Feature Pyramid Network
def fpn(input_size=(128, 128, 3), num_classes=1):
    # Backbone (ResNet50)
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_size)
    c1 = base_model.get_layer("conv1_relu").output  # Salida 1/2
    c2 = base_model.get_layer("conv2_block3_out").output  # Salida 1/4
    c3 = base_model.get_layer("conv3_block4_out").output  # Salida 1/8
    c4 = base_model.get_layer("conv4_block6_out").output  # Salida 1/16

    # Top-down pathway (FPN)
    p4 = Conv2D(256, (1, 1), padding="same", activation="relu")(c4)
    p3 = Add()([
        UpSampling2D((2, 2))(p4),
        Conv2D(256, (1, 1), padding="same", activation="relu")(c3)
    ])
    p2 = Add()([
        UpSampling2D((2, 2))(p3),
        Conv2D(256, (1, 1), padding="same", activation="relu")(c2)
    ])
    p1 = Add()([
        UpSampling2D((2, 2))(p2),
        Conv2D(256, (1, 1), padding="same", activation="relu")(c1)
    ])

    # Final Upsampling to match the input resolution
    outputs = UpSampling2D((2, 2))(p1)  # Escala final a 128x128
    outputs = Conv2D(num_classes, (1, 1), activation="sigmoid")(outputs)

    return Model(inputs=base_model.input, outputs=outputs)

# U-Net con EfficientNet
def unet_efficientnet(input_size=(128, 128, 3), num_classes=1):
    # Backbone (EfficientNetB0)
    backbone = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_size)

    # Encoder
    enc1 = backbone.get_layer("block1a_activation").output  # 64x64
    enc2 = backbone.get_layer("block2a_activation").output  # 32x32
    enc3 = backbone.get_layer("block3a_activation").output  # 16x16
    enc4 = backbone.get_layer("block4a_activation").output  # 8x8

    # Bottleneck
    bottleneck = backbone.get_layer("block6a_activation").output  # 4x4

    # Decoder
    up4 = UpSampling2D((2, 2))(bottleneck)
    dec4 = Conv2D(256, (3, 3), padding="same", activation="relu")(Concatenate()([up4, enc4]))

    up3 = UpSampling2D((2, 2))(dec4)
    dec3 = Conv2D(128, (3, 3), padding="same", activation="relu")(Concatenate()([up3, enc3]))

    up2 = UpSampling2D((2, 2))(dec3)
    dec2 = Conv2D(64, (3, 3), padding="same", activation="relu")(Concatenate()([up2, enc2]))

    up1 = UpSampling2D((2, 2))(dec2)
    dec1 = Conv2D(32, (3, 3), padding="same", activation="relu")(Concatenate()([up1, enc1]))

    # Final Upsampling to (128, 128)
    dec1 = UpSampling2D((2, 2))(dec1)
    outputs = Conv2D(num_classes, (1, 1), activation="sigmoid")(dec1)

    return Model(inputs=backbone.input, outputs=outputs)

# Segformer
def segformer(input_size=(128, 128, 3), num_classes=1):
    # Capa base
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_size)

    # Extracción de características
    layer_outputs = [
        base_model.get_layer('conv2_block3_out').output,  # 32x32
        base_model.get_layer('conv3_block4_out').output,  # 16x16
        base_model.get_layer('conv4_block6_out').output   # 8x8
    ]

    # Decoder
    x = layer_outputs[-1]  # 8x8
    for filters, skip_connection in zip([256, 128, 64], reversed(layer_outputs[:-1])):
        x = UpSampling2D((2, 2))(x)  # Duplicar dimensiones
        x = Concatenate()([x, skip_connection])  # Concatenar skip connections
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)

    # Ajustar salida a 128x128
    x = UpSampling2D((2, 2))(x)  # De 32x32 a 64x64
    x = UpSampling2D((2, 2))(x)  # De 64x64 a 128x128

    # Capa de salida
    outputs = Conv2D(num_classes, (1, 1), activation="sigmoid")(x)

    return Model(inputs=base_model.input, outputs=outputs)


# FUNCIONES DE PERDIDA

# Pérdida de binary crossentropy ponderada
def weighted_binary_crossentropy(y_true, y_pred, weight=3):
    # Asigna más peso a la clase minoritaria (cuerpo de agua)
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true, y_pred) * (1 + weight * y_true)

# Pérdida de Dice
def dice_loss(y_true, y_pred, smooth=1): # sin aumentacion -> val_accuracy: 0.2142
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))

# Pérdida combinada (Binary Crossentropy + Dice)
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# Pérdida de IoU
def iou_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred) - intersection
    return 1 - (intersection + 1e-7) / (union + 1e-7)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    tversky_index = (tp + 1e-7) / (tp + alpha * fn + beta * fp + 1e-7)
    return tf.pow(1 - tversky_index, gamma)


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Pérdida combinada de IoU y Dice
def combined_iou_dice_loss(y_true, y_pred):
    print("Shape of y_true:", y_true.shape)
    print("Shape of y_pred:", y_pred.shape)
    return iou_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


# CALLBACKS
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


def post_process(image, threshold=0.4):
    """
    Procesa una imagen predicha aplicando un umbral.

    Args:
        image: Imagen predicha (matriz de valores continuos).
        threshold: Umbral para la segmentación binaria.

    Returns:
        Imagen procesada con valores binarios.
    """
    return image > threshold

def show_image(image, title="", cmap=None):
    """
    Muestra una imagen con un título.

    Args:
        image: Imagen a mostrar.
        title: Título de la imagen.
        cmap: Mapa de colores para la visualización.

    Returns:
        None.
    """
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')

class ShowProgress(Callback):
    def __init__(self, images, masks, save=False):
        """
        Callback para visualizar el progreso mostrando predicciones durante el entrenamiento.

        Args:
            images: Array de imágenes de entrada para visualización.
            masks: Array de máscaras reales correspondientes a las imágenes.
            save: Si es True, guarda las visualizaciones como archivos.
        """
        super().__init__()
        self.images = images
        self.masks = masks
        self.save = save

    def on_epoch_end(self, epoch, logs=None):
        """
        Método ejecutado al final de cada época para generar visualizaciones.

        Args:
            epoch: Número de la época actual.
            logs: Diccionario con métricas y datos de entrenamiento.

        Returns:
            None.
        """
        id = np.random.randint(len(self.images))
        real_img = self.images[id][np.newaxis, ...]
        pred_mask = self.model.predict(real_img).reshape(128, 128)
        proc_mask1 = post_process(pred_mask)
        proc_mask2 = post_process(pred_mask, threshold=0.5)
        proc_mask3 = post_process(pred_mask, threshold=0.9)
        mask = self.masks[id].reshape(128, 128)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 6, 1)
        show_image(real_img[0], title="Imagen Original")

        plt.subplot(1, 6, 2)
        show_image(pred_mask, title="Máscara Predicha", cmap='gray')

        plt.subplot(1, 6, 3)
        show_image(mask, title="Máscara Real", cmap='gray')

        plt.subplot(1, 6, 4)
        show_image(proc_mask1, title="Procesada@0.4", cmap='gray')

        plt.subplot(1, 6, 5)
        show_image(proc_mask2, title="Procesada@0.5", cmap='gray')

        plt.subplot(1, 6, 6)
        show_image(proc_mask3, title="Procesada@0.9", cmap='gray')

        plt.tight_layout()
        if self.save:
            plt.savefig(f"Progress-{epoch+1}.png")
        plt.show()

class DynamicThresholdCallback(Callback):
    def __init__(self, X_val, y_val, thresholds=np.arange(0.1, 0.9, 0.1)):
        """
        Callback para determinar dinámicamente el mejor umbral para las predicciones.

        Args:
            X_val: Imágenes de validación.
            y_val: Máscaras de validación.
            thresholds: Lista de umbrales a evaluar.
        """
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.thresholds = thresholds

    def on_epoch_end(self, epoch, logs=None):
        """
        Método ejecutado al final de cada época para calcular el mejor umbral.

        Args:
            epoch: Número de la época actual.
            logs: Diccionario con métricas y datos de entrenamiento.

        Returns:
            None.
        """
        best_iou, best_threshold = 0, 0.5
        for t in self.thresholds:
            y_pred_binary = (self.model.predict(self.X_val) > t).astype(np.int32)
            iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)
            iou_metric.update_state(self.y_val.flatten(), y_pred_binary.flatten())
            iou_score = iou_metric.result().numpy()
            if iou_score > best_iou:
                best_iou = iou_score
                best_threshold = t
        print(f"Época {epoch+1}: Mejor Umbral: {best_threshold}, Mejor IoU: {best_iou:.4f}")

class BestIoUCallback(Callback):
    def __init__(self, validation_data, thresholds=np.arange(0.3, 0.8, 0.1)):
        """
        Callback para encontrar el mejor IoU y umbral.

        Args:
            validation_data: Tupla (X_val, y_val) con imágenes y máscaras de validación.
            thresholds: Lista de umbrales a evaluar.
        """
        super().__init__()
        self.validation_data = validation_data
        self.thresholds = thresholds
        self.best_threshold = 0.5

    def on_epoch_end(self, epoch, logs=None):
        """
        Método ejecutado al final de cada época para calcular el IoU y el mejor umbral.

        Args:
            epoch: Número de la época actual.

        Returns:
            None.
        """
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val)
        best_iou, best_threshold = 0, 0.5
        for t in self.thresholds:
            y_pred_binary = (y_pred > t).astype(np.int32)
            iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)
            iou_metric.update_state(y_val.flatten(), y_pred_binary.flatten())
            iou_score = iou_metric.result().numpy()
            if iou_score > best_iou:
                best_iou = iou_score
                best_threshold = t
        self.best_threshold = best_threshold
        print(f"Época {epoch+1}: Mejor Umbral: {best_threshold}, Mejor IoU: {best_iou:.4f}")

# AUMENTACIONES
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A

def get_image_data_generator():
    """
    Configura un generador de datos con un rango reducido de aumentación utilizando ImageDataGenerator.

    Returns:
        datagen: Instancia configurada de ImageDataGenerator.
    """
    datagen = ImageDataGenerator(
        rotation_range=15,  # Rango reducido
        width_shift_range=0.2,  # Rango reducido
        height_shift_range=0.2,  # Rango reducido
        shear_range=0.2,  # Rango reducido
        zoom_range=0.2,  # Rango reducido
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen

def create_albumentations_augmentations():
    """
    Configura un conjunto de transformaciones con Albumentations con un rango reducido.

    Returns:
        augmentation: Composición de transformaciones Albumentations.
    """
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalización
    ], additional_targets={'mask': 'mask'})
    return augmentation

def albumentations_data_generator(images, masks, batch_size, augmentation):
    """
    Generador de datos con aumentación utilizando Albumentations.

    Args:
        images: Array de imágenes de entrenamiento.
        masks: Array de máscaras correspondientes.
        batch_size: Tamaño del batch.
        augmentation: Transformaciones de Albumentations.

    Yields:
        Tuplas (batch_images, batch_masks) aumentadas.
    """
    while True:
        idx = np.random.choice(len(images), batch_size)
        batch_images, batch_masks = [], []
        for i in idx:
            augmented = augmentation(image=images[i], mask=masks[i])
            batch_images.append(augmented['image'])
            batch_masks.append(augmented['mask'])
        yield np.array(batch_images), np.array(batch_masks)

def val_generator(images, masks, batch_size):
    """
    Generador de datos para validación sin aumentación.

    Args:
        images: Array de imágenes de validación.
        masks: Array de máscaras correspondientes.
        batch_size: Tamaño del batch.

    Yields:
        Tuplas (batch_images, batch_masks) sin modificaciones.
    """
    while True:
        idx = np.random.choice(len(images), batch_size)
        yield images[idx], masks[idx]

def visualize_augmented_batch(generator, num_samples=5):
    """
    Visualiza un lote de imágenes y máscaras generadas para verificar las transformaciones.

    Args:
        generator: Generador que produce imágenes y máscaras.
        num_samples: Número de ejemplos a visualizar.

    Returns:
        None
    """
    batch_images, batch_masks = next(generator)
    for i in range(min(num_samples, len(batch_images))):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(batch_images[i])
        plt.title("Imagen aumentada")
        plt.subplot(1, 2, 2)
        plt.imshow(batch_masks[i], cmap="gray")
        plt.title("Máscara aumentada")
        plt.show()

