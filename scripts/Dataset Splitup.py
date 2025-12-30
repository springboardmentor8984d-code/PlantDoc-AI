#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import shutil
from pathlib import Path

# 🔹 Path to resized dataset (input)
src_root = Path(r"C:\PlantDataset_Resized")     # change if your resized dataset is elsewhere

# 🔹 Path where train/val/test folders will be saved (output)
dst_root = Path(r"C:\PlantDataset_Split")

# 🔹 Define split names and ratios
splits = ["train", "val", "test"]
ratios = [0.8, 0.1, 0.1]   # 80% train, 10% validation, 10% test

print(f"Source Folder: {src_root}")
print(f"Destination Folder: {dst_root}")
print(f"Ratios: {ratios}")


# In[2]:


def copy_images(img_paths, dest_dir):
    """
    Copy images from source list to destination folder.
    Skips already existing images.
    """
    for img_path in img_paths:
        dst = dest_dir / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)


# In[3]:


# Loop through each plant folder
for plant_dir in src_root.iterdir():
    if not plant_dir.is_dir():
        continue

    # Loop through each disease folder inside plant folder
    for disease_dir in plant_dir.iterdir():
        if not disease_dir.is_dir():
            continue

        # List all images
        images = list(disease_dir.glob("*.*"))
        if len(images) == 0:
            continue  # skip empty folders

        # Shuffle the images for randomness
        random.shuffle(images)

        # Split based on ratios
        n = len(images)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        subsets = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        # Create corresponding destination folders and copy
        for split in splits:
            split_dir = dst_root / split / plant_dir.name / disease_dir.name
            split_dir.mkdir(parents=True, exist_ok=True)
            copy_images(subsets[split], split_dir)


# In[4]:


print("✅ Dataset successfully split into train/val/test (80/10/10)")


# In[5]:


def count_images(folder):
    return sum(1 for _ in Path(folder).rglob("*.*"))

train_count = count_images(dst_root / "train")
val_count = count_images(dst_root / "val")
test_count = count_images(dst_root / "test")

print(f"📊 Train: {train_count} images")
print(f"📊 Validation: {val_count} images")
print(f"📊 Test: {test_count} images")
print(f"📈 Total: {train_count + val_count + test_count} images")


# In[ ]:




