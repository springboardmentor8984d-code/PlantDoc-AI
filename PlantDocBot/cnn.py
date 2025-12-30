# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# --- Dataset Path ---
dataset_dir = r"/mnt/c/Users/ASUS/OneDrive/Desktop/PlantDocBotDataset/PlantVillage"

# --- Image Configuration ---
img_size = (224, 224)
batch_size = 32

print(f"📂 Using dataset: {dataset_dir}")

# --- Define test directory path ---
test_dir = r"/mnt/c/Users/ASUS/OneDrive/Desktop/PlantDocBotDataset/PlantVillage/test"

# --- Data Augmentation for Training ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# --- Validation and Test — Only Rescaling ---
val_test_datagen = ImageDataGenerator(rescale=1./255)

# --- Load train, val, test sets ---
train_gen = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_gen = val_test_datagen.flow_from_directory(
    os.path.join(dataset_dir, "val"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# --- Check Classes ---
print("✅ Classes found:", train_gen.class_indices)
print("Total Classes:", len(train_gen.class_indices))

# Load pre-trained MobileNetV2 (without top classification layers)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers for your PlantDoc dataset
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

# Create the complete model
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers (for transfer learning)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()

# --- Train CNN Model ---
epochs = 15 # Start small; can increase to 10–20 later if stable

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    verbose=1
)

# --- Plot Accuracy and Loss Curves ---
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

test_loss, test_acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"📉 Test Loss: {test_loss:.4f}")

model.save("plant_disease_model.h5")
print("✅ Model saved as 'plant_disease_model.h5'")

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# --- Load model ---
model = load_model("plant_disease_model.h5")
print("✅ Model loaded successfully.")

# --- Unfreeze last N layers (excluding BatchNorm if needed) ---
for layer in model.layers[-40:]:
    layer.trainable = True

# --- Recompile with lower learning rate ---
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Continue fine-tuning ---
fine_tune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    verbose=1
)

# --- Validation Predictions (for your trained model) ---
import numpy as np
import matplotlib.pyplot as plt

# ✅ Take one batch of validation images
image_batch, label_batch = next(iter(val_gen))

# ✅ Make predictions
predictions = model.predict(image_batch)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(label_batch, axis=1)

# ✅ Get class names from your train_gen
class_names = list(train_gen.class_indices.keys())

# ✅ Display a few sample predictions
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    pred_class = class_names[predicted_labels[i]]
    true_class = class_names[true_labels[i]]
    color = "green" if pred_class == true_class else "red"
    plt.title(f"Pred: {pred_class}\nTrue: {true_class}", color=color)
    plt.axis("off")

plt.tight_layout()
plt.show()

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# --- Path to your test image ---
# 🖼️ Change this path to any image you want to test
img_path = r"/mnt/c/Users/ASUS/OneDrive/Desktop/PlantDocBotDataset/PlantDoc/test/Tomato_leaf/test_Tomato leaf_2.jpg"

# --- Image preprocessing ---
IMG_SIZE = (224, 224)
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# --- Make prediction ---
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

# --- Get class labels from your dataset ---
class_names = list(train_gen.class_indices.keys())
predicted_class = class_names[predicted_index]

# --- Display result ---
plt.imshow(img)
plt.axis("off")
plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
plt.show()

# --- Print text summary ---
print("🌿 --- PLANT DISEASE DIAGNOSIS REPORT --- 🌿")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

# --- Optional: Knowledge summary (customize with your dataset) ---
disease_info = {
    "Tomato___Leaf_Mold": "Caused by Cladosporium fulvum; a common tomato disease under humid conditions. It affects leaves and reduces yield.",
    "Tomato___Target_Spot": "Caused by Corynespora cassiicola fungus; creates brown circular lesions on leaves and fruits.",
    "Potato___Late_blight": "Caused by Phytophthora infestans; leads to dark spots and decay of leaves and tubers.",
    "Tomato___Healthy": "Leaf appears green and free from any spots or mold — indicates a healthy plant."
}

if predicted_class in disease_info:
    print("\n📘 Summary:")
    print(disease_info[predicted_class])
else:
    print("\nSummary: No additional info available for this class.")

model.save("plant_disease_model.h5")
print("✅ Model saved as 'plant_disease_model.h5'")

from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("plant_disease_model.h5")

# Evaluate performance
train_loss, train_acc = model.evaluate(train_gen, verbose=1)
val_loss, val_acc = model.evaluate(val_gen, verbose=1)
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print("\n📊 --- MODEL PERFORMANCE ---")
print(f"✅ Train Accuracy: {train_acc*100:.2f}% | Loss: {train_loss:.4f}")
print(f"✅ Validation Accuracy: {val_acc*100:.2f}% | Loss: {val_loss:.4f}")
print(f"✅ Test Accuracy: {test_acc*100:.2f}% | Loss: {test_loss:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# Pick one test image
img_path = r"/mnt/c/Users/ASUS/OneDrive/Desktop/PlantDocBotDataset/PlantVillage/test/Apple___Cedar_apple_rust/3ef85bf3-728f-46d7-95cc-a5bddf34db98___FREC_C.Rust 3885.JPG"

# Load and preprocess
IMG_SIZE = (224, 224)
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
pred = model.predict(img_array)
predicted_idx = np.argmax(pred)
confidence = np.max(pred) * 100

# Get class label
class_names = list(train_gen.class_indices.keys())
predicted_label = class_names[predicted_idx]

plt.imshow(image.load_img(img_path))
plt.axis("off")
plt.title(f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}%")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# List of image paths (can be from train, val, or test)
image_paths = [
    r"/mnt/c/Users/ASUS/OneDrive/Desktop/PlantDocBotDataset/PlantVillage/test/Blueberry___healthy/4bd2859a-1568-4765-bb20-17dd0de0e0ef___RS_HL 2584.JPG",
    r"/mnt/c/Users/ASUS/OneDrive/Desktop/PlantDocBotDataset/PlantVillage/val/Orange___Haunglongbing_(Citrus_greening)/1ba3557b-102d-48fe-99e7-a86f15b91994___UF.Citrus_HLB_Lab 0182.JPG",
    r"/mnt/c/Users/ASUS/OneDrive/Desktop/PlantDocBotDataset/PlantVillage/train/Pepper,_bell___Bacterial_spot/0a9cfb27-280e-475a-bbb4-8eeaeff38b8c___NREC_B.Spot 9177.JPG"
]

IMG_SIZE = (224, 224)
class_names = list(train_gen.class_indices.keys())

plt.figure(figsize=(12, 8))

for i, img_path in enumerate(image_paths):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)
    predicted_idx = np.argmax(pred)
    confidence = np.max(pred) * 100
    predicted_label = class_names[predicted_idx]

    plt.subplot(1, len(image_paths), i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{predicted_label}\n({confidence:.1f}%)")

plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report

# Predict on test set
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

# Get class labels
class_labels = list(test_gen.class_indices.keys())

print(classification_report(y_true, y_pred_classes, target_names=class_labels))

from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("plant_disease_model_finetuned.h5")

# Evaluate performance
train_loss, train_acc = model.evaluate(train_gen, verbose=1)
val_loss, val_acc = model.evaluate(val_gen, verbose=1)
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print("\n📊 --- MODEL PERFORMANCE ---")
print(f"✅ Train Accuracy: {train_acc*100:.2f}% | Loss: {train_loss:.4f}")
print(f"✅ Validation Accuracy: {val_acc*100:.2f}% | Loss: {val_loss:.4f}")
print(f"✅ Test Accuracy: {test_acc*100:.2f}% | Loss: {test_loss:.4f}")

model.save("plant_disease_model_finetuned.h5")
print("✅ Model saved as 'plant_disease_model_finetuned.h5'")