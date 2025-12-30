import tensorflow as tf
import numpy as np
import sys
from PIL import Image

IMG_SIZE = 224

model = tf.keras.models.load_model("saved_models/mobilenetv2.h5")

class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
 'Potato___healthy', 'Potato___Late_blight', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
 'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two-spotted_spider_mite']

img_path = sys.argv[1]

img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
idx = np.argmax(pred)

print("Prediction:", class_names[idx])
