import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Handle different import styles for different PyTorch versions
try:
    from torchvision.models import resnet50, ResNet50_Weights
except ImportError:
    # Fallback for older PyTorch versions
    from torchvision.models import resnet50
    ResNet50_Weights = None

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Optional imports with fallbacks
try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    print("Warning: ONNX dependencies not found. Export to ONNX will not be available.")
    ONNX_AVAILABLE = False

import json

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32

# Custom dataset class 
class PlantDiseaseDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Check if directory exists
        if not os.path.exists(image_dir):
            print(f"Warning: Directory {image_dir} does not exist")
            self.classes = ['placeholder']
            self.class_to_idx = {'placeholder': 0}
            self.images = []
            self.labels = []
            return
            
        # Get class folders
        self.classes = [d for d in sorted(os.listdir(image_dir)) 
                        if os.path.isdir(os.path.join(image_dir, d))]
        
        # If no class folders found, create a placeholder
        if not self.classes:
            print(f"Warning: No class folders found in {image_dir}")
            self.classes = ['placeholder']
            
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(image_dir, class_name)
            if os.path.isdir(class_dir):
                class_idx = self.class_to_idx[class_name]
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(img_path)
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.images)} images")
            
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a blank image as a fallback
            blank_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, label, img_path

# Create data loaders for training, validation, and testing
def get_data_loaders(data_dir):
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    # Check if directories exist, if not create temporary ones with a warning
    if not os.path.exists(train_dir):
        print(f"Warning: Train directory not found at {train_dir}")
        os.makedirs(train_dir, exist_ok=True)
    
    if not os.path.exists(val_dir):
        print(f"Warning: Validation directory not found at {val_dir}")
        os.makedirs(val_dir, exist_ok=True)
    
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory not found at {test_dir}")
        os.makedirs(test_dir, exist_ok=True)
        
    # Check class directories in train_dir
    if not os.listdir(train_dir):
        print(f"Warning: No class folders found in {train_dir}")
        # Create a placeholder class directory
        placeholder_dir = os.path.join(train_dir, 'placeholder')
        os.makedirs(placeholder_dir, exist_ok=True)
    
    train_dataset = PlantDiseaseDataset(train_dir, transform=train_transform)
    val_dataset = PlantDiseaseDataset(val_dir, transform=val_test_transform)
    test_dataset = PlantDiseaseDataset(test_dir, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Save class names
    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    
    # Save class info to JSON file for later use
    class_info = {
        'class_names': class_names,
        'class_to_idx': class_to_idx
    }
    
    with open('class_info.json', 'w') as f:
        json.dump(class_info, f)
    
    return train_loader, val_loader, test_loader, class_names

# Create ResNet model with custom classifier
def create_model(num_classes):
    try:
        # Try with weights parameter (PyTorch 1.13+)
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    except TypeError:
        # Fallback for older PyTorch versions
        try:
            model = resnet50(pretrained=True)
        except:
            print("Warning: Unable to load pretrained ResNet50 weights. Using randomly initialized model.")
            model = resnet50(pretrained=False)
    
    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    
    return model

# Function to save training history
def save_training_history(history):
    with open('training_history.json', 'w') as f:
        json.dump(history, f)

# Function to load training history
def load_training_history():
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    return history

# Function to save model checkpoint
def save_checkpoint(model, optimizer, epoch, accuracy, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    torch.save(checkpoint, filename)

# Function to load model checkpoint
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    return model, optimizer, epoch, accuracy

# Convert PyTorch model to ONNX format
def convert_to_onnx(model, input_size, output_file):
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Export model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Verify the model
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    
    return True

# Plot training and validation curves
def plot_training_curves(history):
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    return plt.gcf()

# Plot confusion matrix
def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    return plt.gcf()

# Calculate and return classification report
def get_classification_report(true_labels, pred_labels, class_names):
    report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
    return report

# Plot class-wise accuracy
def plot_class_accuracy(report, class_names):
    plt.figure(figsize=(12, 6))
    accuracies = [report[class_name]['precision'] for class_name in class_names]
    
    plt.bar(class_names, accuracies)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('class_accuracy.png')
    return plt.gcf()
