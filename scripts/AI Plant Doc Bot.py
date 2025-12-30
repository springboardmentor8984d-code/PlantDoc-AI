#!/usr/bin/env python
# coding: utf-8

# In[7]:


# --- Import necessary libraries ---
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# --- Step 1: Specify your dataset folder path ---
dataset_path = r"C:\Users\paul\Downloads\archive (2)\MergedDataset"  # <-- change this path to your actual extracted folder

# --- Step 2: Verify dataset folder exists ---
if not os.path.exists(dataset_path):
    print("❌ Error: Dataset folder not found. Please check the path.")
else:
    print(f"✅ Dataset folder found at: {dataset_path}")

    # --- Step 3: Count all image files in the dataset ---
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    count = sum(len([f for f in files if f.lower().endswith(valid_ext)]) 
                for _, _, files in os.walk(dataset_path))

    print(f" Total image files detected: {count:,}")

    # --- Step 4: List all subfolders (class names) ---
    folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f" Found {len(folders)} plant categories:")
    for f in folders[:10]:
        print("  -", f)
    if len(folders) > 10:
        print("  ...")

    # --- Step 5: Show a random grid of sample images ---
    sample_images = []
    for folder in random.sample(folders, min(9, len(folders))):  # pick random folders
        folder_path = os.path.join(dataset_path, folder)
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                  if f.lower().endswith(valid_ext)]
        if images:
            sample_images.append(random.choice(images))

    # Plotting 9 random images
    plt.figure(figsize=(10, 10))
    for i, img_path in enumerate(sample_images[:9]):
        img = Image.open(img_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(os.path.basename(os.path.dirname(img_path)))
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# In[ ]:




