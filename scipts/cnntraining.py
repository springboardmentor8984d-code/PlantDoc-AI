import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# ======================================================
#              CONFIGURATION
# ======================================================
DATASET_PATH = "datasets/PlantVillage"   # path to dataset
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================================================
#       DATA AUGMENTATION  (GOOD + SAFE)
# ======================================================
train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ======================================================
#                MODEL CREATION
# ======================================================
base_model = MobileNetV2(
    include_top=False,
    input_shape=(224, 224, 3),
    weights="imagenet"
)

base_model.trainable = False   # freeze for first stage (FAST)

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================================================
#       PHASE 1 — TRAIN TOP LAYERS (5 EPOCHS)
# ======================================================
print("\n🔵 Training Phase 1 (Frozen Base) — 5 Epochs...\n")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# ======================================================
#       PHASE 2 — FINE-TUNE FULL MODEL (5 EPOCHS)
# ======================================================
print("\n🟢 Training Phase 2 (Fine-tuning) — 5 Epochs...\n")

base_model.trainable = True    # unfreeze everything

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# ======================================================
#               SAVE MODEL
# ======================================================
os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/mobilenetv2.h5")

print("\n🎉 Training Complete! Model saved at saved_models/mobilenetv2.h5\n")
