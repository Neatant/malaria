import io
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

model_path = "model_vgg19.h5"
model = load_model(model_path)

def predict(image_file):
    # Load image
    img = Image.open(io.BytesIO(image_file))
    # Preprocess image
    img = img.resize((224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # Make prediction
    prediction = model.predict(img)
    # Decode prediction
    if prediction[0][0] < prediction[0][1]:
        return {"result": "Uninfected"}
    else:
        return {"result": "Parasitized"}