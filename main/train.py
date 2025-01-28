# VGG19 CNN Architecture
from tensorflow.keras.applications import VGG19 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from dataAug import train_generator, validation_generator

# Load the VGG19 model
base_model = VGG19(input_shape=(128, 128, 3),
                   include_top=False,
                   weights='imagenet')  # Using ImageNet weights for transfer learning

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = base_model.output
flat = Flatten()(x)

class_1 = Dense(4608, activation="relu")(flat)
dropout = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation="relu")(dropout)
output = Dense(2, activation="softmax")(class_2)

# Create the final model
model_01 = Model(base_model.inputs, output)

# Define the file path for saving the model
filePath = "model/model.h5"

# Define the callbacks
es = EarlyStopping(monitor="val_loss", verbose=1, mode="min", patience=4)
cp = ModelCheckpoint(filePath, monitor="val_loss", save_best_only=False, mode="auto", save_freq="epoch")
lrr = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.0001)

# Define the SGD optimizer with momentum
sgd = SGD(learning_rate=0.0001, momentum=0.5, nesterov=True)

# Compile the model
model_01.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

# Train the model
history_01 = model_01.fit(train_generator,
                          steps_per_epoch=50,
                          epochs=20,
                          callbacks=[es, cp, lrr],
                          validation_data=validation_generator)


# Save the trained model
model_01.save(filepath = filePath, overwrite = True)
