"""
Enhanced Training Script for PlantDocBot
Includes: Advanced augmentation, learning rate scheduling, early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ADVANCED DATA AUGMENTATION
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Optional: Random erasing (helps with robustness)
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_names=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        if class_names is None:
            self.class_names = sorted([d for d in os.listdir(root_dir) 
                                      if os.path.isdir(os.path.join(root_dir, d))])
        else:
            self.class_names = class_names
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Load all samples
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"✅ Loaded {len(self.samples)} images from {len(self.class_names)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image if error
            return torch.zeros((3, 224, 224)), label

def create_model(num_classes, pretrained=True):
    """Create MobileNetV2 with improved architecture"""
    if pretrained:
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    else:
        model = models.mobilenet_v2(weights=None)
    
    # Enhanced classifier with batch normalization and dropout
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")

def train_model(train_dir, val_dir, num_epochs=50, batch_size=32, learning_rate=0.001):
    """Main training function with all improvements"""
    
    print("="*70)
    print("  🌿 PLANTDOCBOT - ENHANCED TRAINING")
    print("  Developed by: Khushi")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 Using device: {device}")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset = PlantDiseaseDataset(train_dir, transform=train_transform)
    val_dataset = PlantDiseaseDataset(val_dir, transform=val_transform, 
                                     class_names=train_dataset.class_names)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Save class names
    class_names = train_dataset.class_names
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"✅ Saved {len(class_names)} class names to class_names.json")
    
    # Create model
    print("\n🏗️ Building model...")
    model = create_model(len(class_names), pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("="*70)
    
    for epoch in range(num_epochs):
        print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
        print("-"*70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n📈 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names
            }, 'best_model.pth')
            print(f"   ✅ New best model saved! (Acc: {val_acc:.2f}%)")
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            break
    
    print("\n" + "="*70)
    print(f"  ✅ TRAINING COMPLETE!")
    print(f"  🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*70)
    
    # Final evaluation with confusion matrix
    print("\n📊 Generating final evaluation metrics...")
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    _, _, final_preds, final_labels = validate(model, val_loader, criterion, device)
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(final_labels, final_preds, 
                               target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    plot_confusion_matrix(cm, class_names)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Training history saved to training_history.png")
    
    return model, history

if __name__ == '__main__':
    # CONFIGURATION - Auto-detect common dataset structures
    possible_paths = [
        ('dataset/train', 'dataset/val'),
        ('dataset/Train', 'dataset/Val'),
        ('train', 'val'),
        ('Train', 'Val'),
        ('PlantDoc-Dataset/train', 'PlantDoc-Dataset/val'),
        ('data/train', 'data/val'),
    ]
    
    TRAIN_DIR = None
    VAL_DIR = None
    
    # Auto-detect dataset location
    for train_path, val_path in possible_paths:
        if os.path.exists(train_path) and os.path.exists(val_path):
            TRAIN_DIR = train_path
            VAL_DIR = val_path
            print(f"✅ Found dataset at: {TRAIN_DIR} and {VAL_DIR}")
            break
    
    # If not found, ask user
    if TRAIN_DIR is None:
        print("\n📂 Dataset not found automatically. Please provide paths:")
        TRAIN_DIR = input("Enter training directory path: ").strip()
        VAL_DIR = input("Enter validation directory path: ").strip()
        
        if not os.path.exists(TRAIN_DIR):
            print(f"❌ Error: Training directory not found: {TRAIN_DIR}")
            exit(1)
        if not os.path.exists(VAL_DIR):
            print(f"❌ Error: Validation directory not found: {VAL_DIR}")
            exit(1)
    
    # TRAINING PARAMETERS
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    print(f"\n⚙️ Training Configuration:")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Train Dir: {TRAIN_DIR}")
    print(f"   Val Dir: {VAL_DIR}")
    
    # Start training
    model, history = train_model(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )