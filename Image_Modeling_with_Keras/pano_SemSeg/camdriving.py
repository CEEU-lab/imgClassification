import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import Sequence
import albumentations as A
from keras.callbacks import Callback

#os.environ['SM_FRAMEWORK'] = 'tf.keras'
#import segmentation_models as sm

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

### 1. Data prep ###
# función auxiliar para visualización de datos
def visualize(**images):
    """Muestra imágenes en una fila."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        
        # Handle multi-channel masks
        if len(image.shape) == 3 and image.shape[-1] > 3:
            image = np.argmax(image, axis=-1)  # Select the most probable class
        
        # Normalize only if needed
        if image.dtype == np.float32:
            if image.max() <= 1.0:
                # Image is normalized to [0, 1], scale it back to [0, 255]
                print(f"Scaling normalized image '{name}' back to 0-255.")
                image = (image * 255).astype(np.uint8)
            else:
                # Image is in [0, 255] but float32, just cast to uint8
                print(f"Casting image '{name}' to uint8.")
                image = image.astype(np.uint8)
        
        # Handle masks and grayscale images with a colormap
        cmap = 'viridis' if len(image.shape) == 2 else None
        plt.imshow(image, cmap=cmap)
    plt.show()



# función auxiliar para visualización de datos
def denormalize(x):
    """Escala la imagen al rango 0..1 para una visualización correcta"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# clases para carga y preprocesamiento de datos
class Dataset:
    """Dataset CamVid. Lee imágenes, aplica aumentación y transformaciones de preprocesamiento.

    Args:
        images_dir (str): ruta a la carpeta de imágenes
        masks_dir (str): ruta a la carpeta de máscaras de segmentación
        class_values (list): valores de las clases a extraer de la máscara de segmentación
        augmentation (albumentations.Compose): pipeline de transformación de datos
            (ej. volteo, escala, etc.)
        preprocessing (albumentations.Compose): preprocesamiento de datos
            (ej. normalización, manipulación de forma, etc.)

    """

    CLASSES = ['cielo', 'edificio', 'poste', 'carretera', 'acera',
               'árbol', 'señal', 'valla', 'coche',
               'peatón', 'ciclista', 'sin_etiquetar']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convierte nombres de str a valores de clase en las máscaras
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # lee los datos
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extrae ciertas clases de la máscara (ej. coches)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')

        # añade fondo si la máscara no es binaria
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # aplica aumentaciones
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # aplica preprocesamiento
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image.astype('float32'), mask

    def __len__(self):
        return len(self.ids)


class Dataloader(Sequence):
    """Carga datos del dataset y forma lotes."""

    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
        Args:
            dataset (Dataset): Instancia del dataset.
            batch_size (int): Número de muestras por lote.
            shuffle (bool): Barajar los datos al final de cada época.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = min((i + 1) * self.batch_size, len(self.dataset))  # no exceder la longitud del dataset
        data = [self.dataset[j] for j in range(start, stop)]

        # Si el último lote tiene menos elementos que batch_size, rellenar opcionalmente
        if len(data) < self.batch_size:
            print("Repitiendo la ultima muestra para completar el lote")
            for _ in range(self.batch_size - len(data)):
                data.append(data[-1])  

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        # Convierte las listas de numpy arrays a tensores
        inputs = tf.convert_to_tensor(batch[0], dtype=tf.float32)
        targets = tf.convert_to_tensor(batch[1], dtype=tf.float32)
        
        #return batch
        return inputs, targets

    def __len__(self):
        """Devuelve el número de lotes por época."""
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def on_epoch_end(self):
        """Callback para barajar los índices al final de cada época."""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

#### 1.2. Augmentations ####

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define aumentaciones intensas
def get_intense_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),

        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        A.Perspective(scale=(0.05, 0.1), p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),  # Combina brillo y contraste
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            ],
            p=0.9,
        ),

        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)

def get_simple_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(height=320, width=320, always_apply=True),
    ])


def get_validation_augmentation():
    """Añade relleno para hacer que la forma de la imagen sea divisible por 32"""
    test_transform = [
        A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construye la transformación de preprocesamiento

    Args:
        preprocessing_fn (callable): función de normalización de datos
            (puede ser específica para cada red neuronal preentrenada)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# Pérdidas
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

# callbacks
class ClassMetricsCallback(Callback):
    def __init__(self, validation_data, class_names):
        """
        Callback para calcular métricas por clase (IOU) al final de cada epoch.

        Args:
            validation_data (tuple): Datos de validación en forma (val_data, val_masks).
            class_names (list): Nombres de las clases para las métricas.
        """
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        val_data, val_masks = self.validation_data
        predictions = self.model.predict(val_data)

        print(f"\n--- Métricas por clase: Epoch {epoch + 1} ---")
        for i, class_name in enumerate(self.class_names):
            # Convertir tensores a NumPy y aplanarlos
            y_true = tf.reshape(val_masks[..., i], [-1]).numpy()
            y_pred = tf.reshape(predictions[..., i], [-1]).numpy()

            # Calcular IOU
            intersection = np.sum(y_true * y_pred)
            union = np.sum(y_true) + np.sum(y_pred) - intersection
            iou = intersection / (union + 1e-7)

            print(f"Clase: {class_name} - IOU: {iou:.4f}")
