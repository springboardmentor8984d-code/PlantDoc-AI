"""
Dataset Path Finder - Automatically locate your PlantDoc dataset
Run this first to find your dataset paths
"""

import os
from pathlib import Path

def find_image_folders(root_dir='.', max_depth=3):
    """Find folders containing images"""
    image_folders = []
    
    def search_folder(path, depth=0):
        if depth > max_depth:
            return
        
        try:
            items = os.listdir(path)
            
            # Check if this folder has images
            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in items)
            
            # Check if subfolders have images (dataset structure)
            subfolders_with_images = []
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    sub_items = os.listdir(item_path)
                    if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in sub_items):
                        subfolders_with_images.append(item)
            
            # If this looks like a dataset root (has multiple class folders with images)
            if len(subfolders_with_images) > 5:
                image_count = sum(
                    len([f for f in os.listdir(os.path.join(path, sf)) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    for sf in subfolders_with_images
                )
                
                image_folders.append({
                    'path': path,
                    'classes': len(subfolders_with_images),
                    'images': image_count,
                    'class_names': subfolders_with_images[:5]  # Show first 5
                })
            
            # Continue searching subdirectories
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    search_folder(item_path, depth + 1)
        
        except PermissionError:
            pass
    
    search_folder(root_dir)
    return image_folders

def main():
    print("="*70)
    print("  🔍 PLANTDOCBOT - Dataset Finder")
    print("  Searching for your dataset...")
    print("="*70)
    
    # Search current directory
    print("\n📂 Searching current directory and subdirectories...")
    folders = find_image_folders()
    
    if not folders:
        print("\n❌ No dataset folders found!")
        print("\n💡 Common dataset structures:")
        print("   - Train/")
        print("       ├── Tomato_Early_blight_leaf/")
        print("       ├── Potato_leaf/")
        print("       └── ...")
        print("   - test/")
        print("       ├── Tomato_Early_blight_leaf/")
        print("       └── ...")
        print("\n📌 Make sure your dataset is in the current directory!")
        return
    
    print(f"\n✅ Found {len(folders)} potential dataset folder(s):\n")
    
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder['path']}")
        print(f"   Classes: {folder['classes']}")
        print(f"   Images: {folder['images']}")
        print(f"   Sample classes: {', '.join(folder['class_names'])}...")
        print()
    
    # Try to identify train/test pairs
    print("\n🎯 Recommended paths for training:\n")
    
    # Look for common train/test patterns
    train_candidates = [f for f in folders if 'train' in f['path'].lower()]
    test_candidates = [f for f in folders if 'test' in f['path'].lower() or 'val' in f['path'].lower()]
    
    if train_candidates and test_candidates:
        print("✅ Found train/test split!")
        print(f"   TRAIN_DIR: {train_candidates[0]['path']}")
        print(f"   VAL_DIR: {test_candidates[0]['path']}")
        
        # Create a config file
        with open('dataset_config.txt', 'w') as f:
            f.write(f"TRAIN_DIR={train_candidates[0]['path']}\n")
            f.write(f"VAL_DIR={test_candidates[0]['path']}\n")
        
        print("\n✅ Configuration saved to 'dataset_config.txt'")
        print("   You can now run: python improved_training.py")
    
    elif len(folders) == 1:
        print("⚠️  Only one folder found. You may need to split it into train/val")
        print(f"   Found: {folders[0]['path']}")
        print("\n💡 You can:")
        print("   1. Manually split the dataset into train/ and test/ folders")
        print("   2. Or use this folder for training (will need to create validation split)")
    
    else:
        print("📋 Multiple folders found. Choose the appropriate ones:")
        for folder in folders:
            print(f"   - {folder['path']} ({folder['classes']} classes, {folder['images']} images)")
    
    print("\n" + "="*70)
    print("  ✅ Search complete!")
    print("="*70)

if __name__ == '__main__':
    main()