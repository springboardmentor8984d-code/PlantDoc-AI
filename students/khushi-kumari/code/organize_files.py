"""
PlantDocBot - Automatic Cleanup Script
This will organize your files for GitHub upload
"""

import os
import shutil
from pathlib import Path

def organize_files():
    print("="*80)
    print("  🧹 PLANTDOCBOT - AUTOMATIC FILE ORGANIZER")
    print("="*80)
    print()
    
    # Create folders if they don't exist
    folders_to_create = ['archive', 'screenshots']
    
    for folder in folders_to_create:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ Created folder: {folder}/")
    
    print()
    print("="*80)
    print("  📦 MOVING FILES TO ARCHIVE")
    print("="*80)
    
    # Files to move to archive (old training/test scripts)
    files_to_archive = [
        'check_milestone1_completion.py',
        'check_progress.ps1',
        'final_cnn_train.py',
        'generate_class_names.py',
        'MILESTONE1_PROGRESS.txt',
        'preprocess.py',
        'resize_images_224.py',
        'resize_images_pil.py',
        'run_milestone1_simple.py',
        'simple_bert_train.py',
        'simple_cnn_train.py',
        'train_bert.py',
        'train_cnn_improved.py',
        'train_cnn_pytorch.py',
        'train_distilbert.py',
        'chat_test_results.json',
        'image_test_results.json',
        'test_chat_system.py',
        'test_cnn_model.py',
        'test_image_system.py',
        'test_scripts.py',
        'test_setup.py',
        'disease_treatments.json',
    ]
    
    moved_count = 0
    for filename in files_to_archive:
        if os.path.exists(filename):
            try:
                shutil.move(filename, f'archive/{filename}')
                print(f"  📦 Moved: {filename} → archive/")
                moved_count += 1
            except Exception as e:
                print(f"  ⚠️ Could not move {filename}: {e}")
    
    print(f"\n✅ Moved {moved_count} files to archive/")
    
    # Folders to move to archive
    folders_to_archive = ['data', 'models', 'notebooks', 'src', 'uploads', 'distilbert_plant_disease']
    
    print()
    print("="*80)
    print("  📦 MOVING FOLDERS TO ARCHIVE")
    print("="*80)
    
    folder_moved_count = 0
    for folder in folders_to_archive:
        if os.path.exists(folder) and os.path.isdir(folder):
            try:
                shutil.move(folder, f'archive/{folder}')
                print(f"  📦 Moved: {folder}/ → archive/")
                folder_moved_count += 1
            except Exception as e:
                print(f"  ⚠️ Could not move {folder}: {e}")
    
    print(f"\n✅ Moved {folder_moved_count} folders to archive/")
    
    print()
    print("="*80)
    print("  ✅ ORGANIZATION COMPLETE!")
    print("="*80)
    print()
    print("📁 Your project structure is now:")
    print()
    print("PlantDocBot/")
    print("├── app.py                    ✅ Main application")
    print("├── improved_training.py      ✅ Training script")
    print("├── dataset_finder.py        ✅ Dataset finder")
    print("├── file_checker.py          ✅ File checker")
    print("├── best_model.pth           ✅ Trained model")
    print("├── class_names.json         ✅ Disease classes")
    print("├── requirements.txt         ✅ Dependencies")
    print("├── README.md                ✅ Documentation")
    print("├── .gitignore               ✅ Git ignore")
    print("│")
    print("├── templates/               ✅ HTML files")
    print("├── static/                  ✅ CSS/JS/Images")
    print("├── screenshots/             📸 Project screenshots")
    print("└── archive/                 📦 Old files (not uploaded)")
    print()
    print("="*80)
    print("  ⚠️ IMPORTANT: DELETE venv/ FOLDER MANUALLY")
    print("="*80)
    print()
    print("Run this command in PowerShell:")
    print("  Remove-Item -Recurse -Force venv")
    print()
    print("Or in File Explorer:")
    print("  Right-click venv folder → Delete")
    print()
    print("="*80)
    print("  📝 NEXT STEPS")
    print("="*80)
    print()
    print("1. ✅ Files organized - DONE")
    print("2. ❌ Delete venv/ folder manually")
    print("3. ❌ Create README.md (copy from my earlier message)")
    print("4. ❌ Create .gitignore (copy from my earlier message)")
    print("5. ❌ Take screenshots")
    print("6. ✅ Ready for GitHub upload!")
    print()

if __name__ == '__main__':
    response = input("This will organize your files. Continue? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        organize_files()
        print("✅ Done! Now delete venv/ and create README.md and .gitignore")
    else:
        print("❌ Cancelled")