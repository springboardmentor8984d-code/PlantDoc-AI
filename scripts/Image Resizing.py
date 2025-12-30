#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path
from PIL import Image
import os

# Paths
src_root = Path(r"C:\Users\paul\Downloads\archive (2)\PlantDataset")         # cleaned dataset
dst_root = Path(r"C:\PlantDataset_Resized") # output folder
target_size = (224, 224)

# Function to resize an image
def resize_image(src_path, dst_path, size=(224, 224)):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize(size, Image.LANCZOS)
            img.save(dst_path, quality=95)
    except Exception as e:
        print(f"⚠️ Skipping {src_path}: {e}")

count = 0
for plant_dir in src_root.iterdir():
    if not plant_dir.is_dir():
        continue
    for disease_dir in plant_dir.iterdir():
        if not disease_dir.is_dir():
            continue

        out_dir = dst_root / plant_dir.name / disease_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in disease_dir.glob("*.*"):
            dst_path = out_dir / img_path.name
            if not dst_path.exists():
                resize_image(img_path, dst_path, target_size)
                count += 1

print(f"\n✅ Resizing complete. Total images processed: {count}")


# In[3]:


from pathlib import Path
count = sum(1 for _ in Path(r"C:\PlantDataset_Resized").rglob("*.jpg"))
print(f"✅ Already resized images found: {count}")


# In[ ]:




