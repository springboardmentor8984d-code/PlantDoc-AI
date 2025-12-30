#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import cv2
import hashlib
import shutil
from pathlib import Path
from tqdm import tqdm

# 🔹 Path to your original dataset
root = Path(r"C:\Users\paul\Downloads\archive (2)\MergedDataset")   # <-- change this to your dataset path

# 🔹 Path where invalid images will be moved
invalid_dir = Path(r"C:\PlantDataset_Invalid")
invalid_dir.mkdir(exist_ok=True)

print(f"Source dataset: {root}")
print(f"Invalid images will be moved to: {invalid_dir}")


# In[4]:


def is_blurry(image_path, threshold=20.0):
    """Detect blurry images using Laplacian variance."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return True  # unreadable file
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold
    except Exception:
        return True

def file_hash(image_path):
    """Generate MD5 hash for image to find duplicates."""
    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


# In[5]:


# Dictionaries to store hashes and issues
seen_hashes = {}
duplicate_images = []
blurry_images = []
corrupt_images = []

image_files = list(root.rglob("*.*"))
print(f"🔍 Scanning {len(image_files)} images...")

for img_path in tqdm(image_files):
    # Skip non-image files
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    # Check if file can be opened
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            corrupt_images.append(img_path)
            continue
    except Exception:
        corrupt_images.append(img_path)
        continue

    # Check for duplicates
    h = file_hash(img_path)
    if h:
        if h in seen_hashes:
            duplicate_images.append(img_path)
        else:
            seen_hashes[h] = img_path

    # Check for blurriness
    if is_blurry(img_path, threshold=20.0):
        blurry_images.append(img_path)


# In[6]:


def move_images(image_list, reason):
    if not image_list:
        print(f"✅ No {reason} images found.")
        return
    subfolder = invalid_dir / reason
    subfolder.mkdir(exist_ok=True)
    for img_path in tqdm(image_list, desc=f"Moving {reason} images"):
        try:
            dst = subfolder / img_path.name
            shutil.move(str(img_path), str(dst))
        except Exception as e:
            print(f"⚠️ Error moving {img_path}: {e}")

move_images(corrupt_images, "corrupt")
move_images(duplicate_images, "duplicates")
move_images(blurry_images, "blurry")


# In[7]:


print("\n🌿 Cleaning Summary")
print("--------------------------------------")
print(f"Total images scanned : {len(image_files)}")
print(f"Corrupt images moved : {len(corrupt_images)}")
print(f"Duplicate images moved: {len(duplicate_images)}")
print(f"Blurry images moved  : {len(blurry_images)}")
print("--------------------------------------")
print(f"🗂️ Invalid images safely stored in: {invalid_dir}")



# In[ ]:




