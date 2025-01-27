#VGG19 CNN Architecture
from tensorflow.keras.applications import VGG19 # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Dropout # type: ignore
from tensorflow.keras.optimizers import SGD, RMSprop, Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore

base_model = VGG19(input_shape = (128,128, 3),
                  include_top = False,
                  weights = 'imagenet'  # Using ImageNet weights for transfer learning
                )
for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
flat = Flatten()(x)

class_1 = Dense(4608, activation = "relu")(flat)
dropout = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation = "relu")(dropout)
output = Dense(2, activation = "softmax")(class_2)

model_01 = Model(base_model.inputs, output)

""" model_01.summary() """