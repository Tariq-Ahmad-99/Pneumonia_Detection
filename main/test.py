import tensorflow as tf
from dataAug import test_datagen, validation_generator

test_generator = test_datagen.flow_from_directory("data/test",
                                  batch_size = 32,
                                  target_size = (128,128),
                                  class_mode = "categorical",
                                  shuffle = True,
                                  seed = 42,
                                  color_mode = "rgb")

""" # Load the trained model

model_01 = tf.keras.models.load_model("model\model.h5")

# Evaluate the model
vgg_val_eval_01 = model_01.evaluate(validation_generator)
vgg_test_eval_01 = model_01.evaluate(test_generator)

print(f"Validation Loss: {vgg_val_eval_01[0]}")
print(f"Validation Accuracy: {vgg_val_eval_01[1]}")

print(f"Test Loss: {vgg_test_eval_01[0]}")
print(f"Test Accuracy: {vgg_test_eval_01[1]}") """


# Load the trained model

model_02 = tf.keras.models.load_model("model\model_2.h5")

# Evaluate the model
vgg_val_eval_02 = model_02.evaluate(validation_generator)
vgg_test_eval_02 = model_02.evaluate(test_generator)

print(f"Validation Loss: {vgg_val_eval_02[0]}")
print(f"Validation Accuracy: {vgg_val_eval_02[1]}")

print(f"Test Loss: {vgg_test_eval_02[0]}")
print(f"Test Accuracy: {vgg_test_eval_02[1]}")