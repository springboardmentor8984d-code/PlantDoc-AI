#!/usr/bin/env python
# coding: utf-8

# In[17]:


# ============================================================
# 🌱 Plant Disease Dataset Splitter (80/10/10)
# ============================================================
# ✅ This notebook splits your dataset (e.g., plant → disease → images)
#    into train, val, and test sets using symbolic links.
# ⚡ Super fast, minimal disk usage, and keeps folder structure.
# ============================================================

import os
import random
from pathlib import Path

print("✅ Imports loaded successfully.")


# In[21]:


# ------------------------------------------------------------
# 🗂️ Define your dataset directories
# ------------------------------------------------------------
# Change these paths to match your local setup

src_root = Path(r"C:\Users\paul\Downloads\archive (2)\MergedDataset")     # 📁 Original dataset
dst_root = Path(r"C:\Users\paul\Downloads\archive (2)\MergedDataset\split_dataset")    # 📁 Output split dataset

# Define split ratios (Train:Val:Test)
splits = ["train", "val", "test"]
ratios = [0.8, 0.1, 0.1]  # 80%, 10%, 10%

print(f"Source: {src_root}")
print(f"Destination: {dst_root}")
print(f"Splits: {splits} | Ratios: {ratios}")


# In[18]:


# ------------------------------------------------------------
# 🔗 Create a symbolic link instead of copying files
# ------------------------------------------------------------
def make_symlink(src, dst):
    """
    Create a symbolic link pointing to the original image.
    Works like a shortcut — saves space and time.
    """
    try:
        dst.symlink_to(src)
    except Exception as e:
        print(f"⚠️ Could not create symlink for {src.name}: {e}")

print("✅ Symlink function ready.")


# In[19]:


# ------------------------------------------------------------
# 🔀 Split the dataset into train/val/test folders
# ------------------------------------------------------------
def split_dataset():
    total_count = {"train": 0, "val": 0, "test": 0}

    # Loop through each plant category
    for plant_dir in src_root.iterdir():
        if not plant_dir.is_dir():
            continue

        # Loop through each disease/subfolder under that plant
        for subfolder in plant_dir.iterdir():
            if not subfolder.is_dir():
                continue

            # List all images
            images = list(subfolder.glob("*.*"))
            random.shuffle(images)

            n = len(images)
            n_train = int(n * ratios[0])
            n_val = int(n * ratios[1])

            subsets = {
                "train": images[:n_train],
                "val": images[n_train:n_train + n_val],
                "test": images[n_train + n_val:]
            }

            # Create split directories and symbolic links
            for split in splits:
                split_dir = dst_root / split / plant_dir.name / subfolder.name
                split_dir.mkdir(parents=True, exist_ok=True)

                for img_path in subsets[split]:
                    link_path = split_dir / img_path.name
                    if not link_path.exists():
                        make_symlink(img_path, link_path)

                total_count[split] += len(subsets[split])

            # Print summary for this subfolder
            print(f"📁 {plant_dir.name}/{subfolder.name}: "
                  f"train={len(subsets['train'])}, "
                  f"val={len(subsets['val'])}, "
                  f"test={len(subsets['test'])}")

    # Print overall summary
    print("\n✅ Dataset split completed using symbolic links (no duplication).")
    print("------------------------------------------------------------")
    for split, count in total_count.items():
        print(f"{split.upper():5}: {count} images")
    print("------------------------------------------------------------")

print("✅ Split logic function ready.")


# In[22]:


# ------------------------------------------------------------
# 🚀 Execute the dataset split
# ------------------------------------------------------------
split_dataset()


# In[ ]:




