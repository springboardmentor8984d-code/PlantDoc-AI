# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import joblib

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize NLTK tools
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load your dataset
DATA_PATH = "/mnt/c/Users/ASUS/OneDrive/Desktop/PlantDocBotDataset/RV1/plant_disease_dataset_clean_lemmatized03.csv"
df = pd.read_csv(DATA_PATH)
print("Loaded rows:", len(df))
print("Columns:", df.columns.tolist())

# Clean text function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in STOPWORDS and w]
    return " ".join(words)

# Apply text cleaning
df["clean_text"] = df["Symptoms"].apply(clean_text)
print("Sample cleaned text:")
print(df[["Symptoms","clean_text"]].head(3))

# Disease mapping
mapping = {
    # Healthy
    "healthy": "healthy",

    # -------------------- BACTERIAL --------------------
    "bacterial spot": "bacterial",
    "citrus canker": "bacterial",
    "bacterial wilt": "bacterial",
    "common scab": "bacterial",

    # -------------------- BLIGHT --------------------
    "early blight": "blight",
    "late blight": "blight",
    "northern leaf blight": "blight",

    # -------------------- MILDEW / MOLD --------------------
    "powdery mildew": "mildew",
    "leaf mold": "mildew",

    # -------------------- ROT / MOLD --------------------
    "black rot": "rot_mold",
    "brown rot": "rot_mold",
    "gray mold": "rot_mold",
    "cane blight": "rot_mold",
    "mummy berry": "rot_mold",
    "anthracnose": "rot_mold",

    # -------------------- VIRAL --------------------
    "tomato mosaic virus": "viral",
    "tomato yellow leaf curl virus": "viral",
    "haunglongbing": "viral",

    # -------------------- LEAF SPOT --------------------
    "leaf spot": "leaf_spot",
    "cercospora leaf spot": "leaf_spot",
    "septoria leaf spot": "leaf_spot",
    "frogeye leaf spot": "leaf_spot",
    "leaf scorch": "leaf_spot",
    "target spot": "leaf_spot",

    # -------------------- RUST --------------------
    "common rust": "rust",
    "soybean rust": "rust",
    "cedar apple rust": "rust",

    # -------------------- SCAB --------------------
    "apple scab": "scab",

    # -------------------- CURL --------------------
    "peach leaf curl": "curl"
}

# Apply mapping
df["merged_label"] = df["Disease"].map(mapping)

# Show rows with no mapping
unmapped = df[df["merged_label"].isna()]["Disease"].unique()
print("UNMAPPED DISEASES (if any):", len(unmapped))
print(unmapped)

# Handle unmapped diseases
if len(unmapped) > 0:
    df.loc[df["merged_label"].isna(), "merged_label"] = "other"
    print("Unmapped diseases auto-assigned to 'other'")

# Show class counts
print("Merged class counts:")
print(df["merged_label"].value_counts())

# Encode labels
le = LabelEncoder()
df["encoded_label"] = le.fit_transform(df["merged_label"])

# Save label encoder and mapping
joblib.dump(le, "label_encoder_merged.joblib")
df[["Disease","merged_label"]].drop_duplicates().to_csv("disease_to_merged.csv", index=False)
print("Saved label_encoder_merged.joblib and disease_to_merged.csv")
print("Label classes:", list(le.classes_))

# Tokenize text with BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 320

enc = tokenizer(
    df["clean_text"].tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LEN,
    return_tensors="tf"
)

input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]
print("Tokenized:", input_ids.shape)

# Split data
labels = df["encoded_label"].values
train_idx, val_idx = train_test_split(
    np.arange(len(labels)),
    test_size=0.20,
    random_state=42,
    stratify=labels
)

train_input_ids = tf.gather(input_ids, train_idx)
train_attention_mask = tf.gather(attention_mask, train_idx)
train_labels = tf.gather(tf.convert_to_tensor(labels, dtype=tf.int32), train_idx)

val_input_ids = tf.gather(input_ids, val_idx)
val_attention_mask = tf.gather(attention_mask, val_idx)
val_labels = tf.gather(tf.convert_to_tensor(labels, dtype=tf.int32), val_idx)

print("Train:", train_labels.shape, "Val:", val_labels.shape)

# Create TensorFlow datasets
BATCH = 8

train_ds = tf.data.Dataset.from_tensor_slices((
    {"input_ids": train_input_ids, "attention_mask": train_attention_mask},
    train_labels
)).shuffle(1024).batch(BATCH).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((
    {"input_ids": val_input_ids, "attention_mask": val_attention_mask},
    val_labels
)).batch(BATCH).prefetch(tf.data.AUTOTUNE)

print("Datasets ready.")

# Create and train BERT model
num_labels = len(le.classes_)
model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

print("\n🔥 Training started...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    verbose=1
)

print("\n🎉 Training complete!")

# Evaluate model
val_preds = model.predict(val_ds)
logits = val_preds.logits
pred_classes = np.argmax(logits, axis=1)

print("\n=== Classification Report ===\n")
print(classification_report(val_labels, pred_classes, target_names=le.classes_))

print("\n=== Confusion Matrix ===\n")
print(confusion_matrix(val_labels, pred_classes))

# Save model and tokenizer
model.save_pretrained("Final_Text_Disease_Model")
tokenizer.save_pretrained("Final_Text_Disease_Model")
joblib.dump(le, "label_encoder_merged.joblib")
print("Model saved to 'Final_Text_Disease_Model'")