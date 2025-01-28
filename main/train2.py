from tensorflow.keras.applications import VGG19 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore