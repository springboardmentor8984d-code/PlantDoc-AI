"""
PlantDocBot - File Checker for GitHub Upload
This script checks which files should be uploaded and which should be deleted
"""

import os
from pathlib import Path

def get_file_size(filepath):
    """Get file size in MB"""
    try:
        size = os.path.getsize(filepath)
        return size / (1024 * 1024)  # Convert to MB
    except:
        return 0

def check_project_files():
    """Check all files in the project"""
    
    # Files/folders that MUST be kept
    KEEP_FILES = {
        'app.py': '✅ MAIN APPLICATION',
        'improved_training.py': '✅ TRAINING SCRIPT',
        'dataset_finder.py': '✅ DATASET FINDER',
        'README.md': '✅ PROJECT DOCUMENTATION',
        'requirements.txt': '✅ DEPENDENCIES LIST',
        '.gitignore': '✅ GIT IGNORE FILE',
        'class_names.json': '✅ DISEASE CLASSES',
    }
    
    # Files that SHOULD be deleted
    DELETE_PATTERNS = {
        '__pycache__': '❌ DELETE (Python cache)',
        '.pyc': '❌ DELETE (Compiled Python)',
        'venv': '❌ DELETE (Virtual environment - too large)',
        '.vscode': '❌ DELETE (VS Code settings)',
        '.idea': '❌ DELETE (IDE settings)',
        '.DS_Store': '❌ DELETE (Mac system file)',
        'Thumbs.db': '❌ DELETE (Windows system file)',
        '*.log': '❌ DELETE (Log files)',
        '*.tmp': '❌ DELETE (Temporary files)',
        '*.bak': '❌ DELETE (Backup files)',
    }
    
    # Files that MAY BE TOO LARGE
    LARGE_FILE_WARNING = {
        '.pth': '⚠️ MODEL FILE (check size)',
        '.pt': '⚠️ MODEL FILE (check size)',
        'PlantDoc-Dataset': '⚠️ DATASET (too large for GitHub)',
        'dataset': '⚠️ DATASET (too large for GitHub)',
        'Train': '⚠️ DATASET (too large for GitHub)',
        'test': '⚠️ DATASET (too large for GitHub)',
    }
    
    # Optional files (good to have)
    OPTIONAL_FILES = {
        'templates': '📁 OPTIONAL (HTML templates folder)',
        'static': '📁 OPTIONAL (CSS/JS/Images folder)',
        'screenshots': '📁 OPTIONAL (Project screenshots)',
        'best_model.pth': '⚠️ OPTIONAL (Model - check size)',
    }
    
    print("="*80)
    print("  🌿 PLANTDOCBOT - FILE CHECKER FOR GITHUB")
    print("="*80)
    print()
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"📂 Checking directory: {current_dir}")
    print()
    
    # Lists to store results
    must_keep = []
    must_delete = []
    check_size = []
    optional = []
    unknown = []
    
    # Check all files and folders
    for item in os.listdir('.'):
        item_path = Path(item)
        is_dir = item_path.is_dir()
        
        # Check if it's in KEEP list
        if item in KEEP_FILES:
            size = get_file_size(item)
            must_keep.append((item, KEEP_FILES[item], size))
            continue
        
        # Check if it should be deleted
        should_delete = False
        for pattern, reason in DELETE_PATTERNS.items():
            if pattern in item or item.endswith(pattern.replace('*', '')):
                must_delete.append((item, reason))
                should_delete = True
                break
        
        if should_delete:
            continue
        
        # Check if it needs size verification
        needs_check = False
        for pattern, reason in LARGE_FILE_WARNING.items():
            if pattern in item or item.endswith(pattern.replace('*', '')):
                size = get_file_size(item) if not is_dir else 0
                check_size.append((item, reason, size))
                needs_check = True
                break
        
        if needs_check:
            continue
        
        # Check if it's optional
        is_optional = False
        for pattern, reason in OPTIONAL_FILES.items():
            if pattern in item:
                size = get_file_size(item) if not is_dir else 0
                optional.append((item, reason, size))
                is_optional = True
                break
        
        if not is_optional:
            # Unknown file
            size = get_file_size(item) if not is_dir else 0
            unknown.append((item, size))
    
    # Display results
    print("\n" + "="*80)
    print("  ✅ FILES YOU MUST KEEP (Upload to GitHub)")
    print("="*80)
    if must_keep:
        for filename, reason, size in must_keep:
            if os.path.exists(filename):
                print(f"  ✅ {filename:<30} {reason:<30} ({size:.2f} MB)")
            else:
                print(f"  ⚠️ {filename:<30} {reason:<30} [MISSING - CREATE THIS!]")
    else:
        print("  ⚠️ No required files found!")
    
    print("\n" + "="*80)
    print("  ❌ FILES YOU MUST DELETE (Do NOT upload)")
    print("="*80)
    if must_delete:
        for filename, reason in must_delete:
            print(f"  ❌ {filename:<30} {reason}")
    else:
        print("  ✅ No files to delete - Good!")
    
    print("\n" + "="*80)
    print("  ⚠️ FILES TO CHECK SIZE (May be too large for GitHub)")
    print("="*80)
    if check_size:
        for filename, reason, size in check_size:
            if size > 100:
                print(f"  ❌ {filename:<30} {reason:<30} ({size:.2f} MB) - TOO LARGE!")
            elif size > 50:
                print(f"  ⚠️ {filename:<30} {reason:<30} ({size:.2f} MB) - Check if needed")
            else:
                print(f"  ✅ {filename:<30} {reason:<30} ({size:.2f} MB) - OK to upload")
    else:
        print("  ✅ No large files found")
    
    print("\n" + "="*80)
    print("  📁 OPTIONAL FILES (Good to have, but not required)")
    print("="*80)
    if optional:
        for filename, reason, size in optional:
            if os.path.exists(filename):
                print(f"  📁 {filename:<30} {reason:<30} ({size:.2f} MB)")
            else:
                print(f"  ⚠️ {filename:<30} {reason:<30} [NOT FOUND]")
    else:
        print("  ℹ️ No optional files found")
    
    print("\n" + "="*80)
    print("  ❓ UNKNOWN FILES (Review these)")
    print("="*80)
    if unknown:
        for filename, size in unknown:
            if filename.endswith('.py'):
                print(f"  📝 {filename:<30} Python file ({size:.2f} MB) - Decide if needed")
            elif filename.endswith(('.json', '.txt', '.md')):
                print(f"  📄 {filename:<30} Text file ({size:.2f} MB) - Probably keep")
            elif filename.endswith(('.jpg', '.png', '.jpeg')):
                print(f"  🖼️ {filename:<30} Image file ({size:.2f} MB) - Put in screenshots/")
            else:
                print(f"  ❓ {filename:<30} ({size:.2f} MB) - Review manually")
    else:
        print("  ✅ All files categorized!")
    
    print("\n" + "="*80)
    print("  📊 SUMMARY")
    print("="*80)
    print(f"  ✅ Files to keep:     {len(must_keep)}")
    print(f"  ❌ Files to delete:   {len(must_delete)}")
    print(f"  ⚠️ Files to check:    {len(check_size)}")
    print(f"  📁 Optional files:    {len(optional)}")
    print(f"  ❓ Unknown files:     {len(unknown)}")
    
    # Check if all required files exist
    print("\n" + "="*80)
    print("  🎯 MISSING REQUIRED FILES CHECK")
    print("="*80)
    
    missing_files = []
    for filename in KEEP_FILES.keys():
        if not os.path.exists(filename):
            missing_files.append(filename)
            print(f"  ❌ MISSING: {filename} - You need to create this!")
    
    if not missing_files:
        print("  ✅ All required files are present!")
    
    # Final recommendations
    print("\n" + "="*80)
    print("  💡 RECOMMENDATIONS")
    print("="*80)
    
    print("\n  1️⃣ CREATE MISSING FILES:")
    if missing_files:
        for f in missing_files:
            print(f"     - Create {f}")
    else:
        print("     ✅ All files exist")
    
    print("\n  2️⃣ DELETE THESE:")
    if must_delete:
        print("     Run these commands:")
        for filename, _ in must_delete:
            if os.path.isdir(filename):
                print(f"     rmdir /s /q {filename}  (Windows)")
                print(f"     rm -rf {filename}        (Linux/Mac)")
            else:
                print(f"     del {filename}            (Windows)")
                print(f"     rm {filename}             (Linux/Mac)")
    else:
        print("     ✅ Nothing to delete")
    
    print("\n  3️⃣ CHECK THESE LARGE FILES:")
    for filename, reason, size in check_size:
        if size > 100:
            print(f"     ❌ {filename} is {size:.2f} MB - DON'T upload to GitHub!")
            print(f"        Either use Git LFS or mention in README where to download it")
    
    print("\n  4️⃣ ORGANIZE:")
    print("     - Put screenshots in screenshots/ folder")
    print("     - Put HTML files in templates/ folder")
    print("     - Put CSS/JS in static/ folder")
    
    print("\n" + "="*80)
    print("  ✅ FILE CHECK COMPLETE!")
    print("="*80)
    print()

if __name__ == '__main__':
    check_project_files()
    
    print("\n💡 TIP: Run this script anytime to check your files before uploading!")
    print("Command: python file_checker.py")
    print()