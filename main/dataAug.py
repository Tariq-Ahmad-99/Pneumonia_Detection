# Data Augmentation & Resizing

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore


train_datagen = ImageDataGenerator(rescale = 1. / 255,
                   horizontal_flip = 0.4,
                   vertical_flip = 0.4,
                   rotation_range = 40,
                   shear_range = 0.2,
                   width_shift_range = 0.4,
                   height_shift_range = 0.4,
                   fill_mode = "nearest")

valid_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory("data/train",
                                  batch_size = 32,
                                  target_size = (128,128),
                                  class_mode = "categorical",
                                  shuffle = True,
                                  seed = 42,
                                  color_mode = "rgb")

validation_generator = valid_datagen.flow_from_directory("data/val",
                                  batch_size = 32,
                                  target_size = (128,128),
                                  class_mode = "categorical",
                                  shuffle = True,
                                  seed = 42,
                                  color_mode = "rgb")

""" class_labels = train_generator.class_indices
class_name = {value:key for (key, value) in class_labels.items()}
print(class_name) """