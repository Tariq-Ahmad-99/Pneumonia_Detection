import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.applications.vgg19 import VGG19 # type: ignore


base_model = VGG19(include_top=False, input_shape=(128,128,3))
x = base_model.output
flat=Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights('model/unfrozen.h5')
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "Normal"
	elif classNo==1:
		return "Pneumonia"


# Function to preprocess image and get model prediction
def getResult(img):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # CHANGED: Convert BGR to RGB

    image = Image.fromarray(image)
    image = image.resize((128, 128))  # Resize to match training size

    image = np.array(image) / 255.0  # CHANGED: Normalize pixel values (0 to 1)

    input_img = np.expand_dims(image, axis=0)  # Add batch dimension

    result = model_03.predict(input_img)
    
    print("Raw model output:", result)  # CHANGED: Debugging print statement
    result01 = np.argmax(result, axis=1)[0]  # CHANGED: Extract single prediction
    print("Predicted class:", result01)  # CHANGED: Debugging print statement

    return result01


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        value = getResult(file_path)
        result = get_className(value)

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)