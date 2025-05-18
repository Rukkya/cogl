import os
import shutil
import random
from pathlib import Path
import streamlit as st
from tqdm import tqdm

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        input_dir (str): Path to the input dataset directory containing class folders
        output_dir (str): Path to the output directory where split dataset will be saved
        train_ratio (float): Ratio of images to use for training (default: 0.7)
        val_ratio (float): Ratio of images to use for validation (default: 0.2)
        test_ratio (float): Ratio of images to use for testing (default: 0.1)
        random_seed (int): Random seed for reproducibility (default: 42)
        
    Returns:
        dict: Dictionary containing paths to train, validation, and test directories
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create output directory structure
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'valid')
    test_dir = os.path.join(output_dir, 'test')
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get class folders
    class_folders = [folder for folder in os.listdir(input_dir) 
                     if os.path.isdir(os.path.join(input_dir, folder))]
    
    # If no class folders found, check if all images are directly in the input directory
    if not class_folders:
        class_folders = ['']  # Empty string to handle the case where images are directly in the input directory
    
    # Initialize counters
    total_images = 0
    train_count = 0
    val_count = 0
    test_count = 0
    
    # Process each class
    for class_folder in tqdm(class_folders, desc="Processing classes"):
        if class_folder:
            class_path = os.path.join(input_dir, class_folder)
            target_train_dir = os.path.join(train_dir, class_folder)
            target_val_dir = os.path.join(val_dir, class_folder)
            target_test_dir = os.path.join(test_dir, class_folder)
        else:
            # If images are directly in the input directory
            class_path = input_dir
            # Create a single class directory "unclassified" for these images
            class_folder = "unclassified"
            target_train_dir = os.path.join(train_dir, class_folder)
            target_val_dir = os.path.join(val_dir, class_folder)
            target_test_dir = os.path.join(test_dir, class_folder)
        
        # Create class directories in train, valid, and test splits
        os.makedirs(target_train_dir, exist_ok=True)
        os.makedirs(target_val_dir, exist_ok=True)
        os.makedirs(target_test_dir, exist_ok=True)
        
        # Get all images in the class folder
        images = [img for img in os.listdir(class_path) 
                 if img.lower().endswith(('.png', '.jpg', '.jpeg')) and 
                 os.path.isfile(os.path.join(class_path, img))]
        
        # Skip if no images found
        if not images:
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        train_idx = int(n_images * train_ratio)
        val_idx = int(n_images * (train_ratio + val_ratio))
        
        # Split images
        train_images = images[:train_idx]
        val_images = images[train_idx:val_idx]
        test_images = images[val_idx:]
        
        # Copy images to respective directories
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_train_dir, img)
            shutil.copy2(src, dst)
            train_count += 1
        
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_val_dir, img)
            shutil.copy2(src, dst)
            val_count += 1
        
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_test_dir, img)
            shutil.copy2(src, dst)
            test_count += 1
        
        total_images += n_images
    
    # Print summary
    print(f"Dataset split complete:")
    print(f"Total images: {total_images}")
    print(f"Training images: {train_count} ({train_count/total_images:.2%})")
    print(f"Validation images: {val_count} ({val_count/total_images:.2%})")
    print(f"Testing images: {test_count} ({test_count/total_images:.2%})")
    
    # Return paths
    return {
        'train_dir': train_dir,
        'val_dir': val_dir,
        'test_dir': test_dir,
        'stats': {
            'total_images': total_images,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count
        }
    }

def analyze_dataset_structure(dataset_path):
    """
    Analyze the structure of the dataset to determine if it needs splitting.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        dict: Dictionary with dataset analysis
    """
    # Check if the dataset has train, valid, test folders
    has_train = os.path.isdir(os.path.join(dataset_path, 'train'))
    has_valid = os.path.isdir(os.path.join(dataset_path, 'valid'))
    has_test = os.path.isdir(os.path.join(dataset_path, 'test'))
    
    # If all three exist, the dataset is already split
    already_split = has_train and has_valid and has_test
    
    # Check for class folders at the root level
    root_dirs = [d for d in os.listdir(dataset_path) 
                if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Remove train, valid, test from the list if they exist
    if has_train and 'train' in root_dirs:
        root_dirs.remove('train')
    if has_valid and 'valid' in root_dirs:
        root_dirs.remove('valid')
    if has_test and 'test' in root_dirs:
        root_dirs.remove('test')
    
    # Check if there are other directories that could be class folders
    has_class_folders = len(root_dirs) > 0
    
    # Check if there are images directly in the root folder
    root_images = [f for f in os.listdir(dataset_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) and
                  os.path.isfile(os.path.join(dataset_path, f))]
    has_root_images = len(root_images) > 0
    
    # Determine total images and classes
    total_images = 0
    class_names = []
    class_counts = {}
    
    if already_split:
        # Count images in the split folders
        train_classes = [d for d in os.listdir(os.path.join(dataset_path, 'train'))
                         if os.path.isdir(os.path.join(dataset_path, 'train', d))]
        
        for cls in train_classes:
            class_names.append(cls)
            
            # Count images in train folder
            train_images = [f for f in os.listdir(os.path.join(dataset_path, 'train', cls))
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Count images in valid folder if it exists
            valid_images = []
            if os.path.isdir(os.path.join(dataset_path, 'valid', cls)):
                valid_images = [f for f in os.listdir(os.path.join(dataset_path, 'valid', cls))
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Count images in test folder if it exists
            test_images = []
            if os.path.isdir(os.path.join(dataset_path, 'test', cls)):
                test_images = [f for f in os.listdir(os.path.join(dataset_path, 'test', cls))
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            total_class_images = len(train_images) + len(valid_images) + len(test_images)
            total_images += total_class_images
            class_counts[cls] = {
                'train': len(train_images),
                'valid': len(valid_images),
                'test': len(test_images),
                'total': total_class_images
            }
    
    elif has_class_folders:
        # Count images in each class folder
        for cls in root_dirs:
            class_path = os.path.join(dataset_path, cls)
            if os.path.isdir(class_path):
                class_names.append(cls)
                
                images = [f for f in os.listdir(class_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                total_images += len(images)
                class_counts[cls] = len(images)
    
    elif has_root_images:
        # Count images in the root folder
        class_names = ['unclassified']
        total_images = len(root_images)
        class_counts['unclassified'] = total_images
    
    # Return analysis results
    return {
        'already_split': already_split,
        'has_class_folders': has_class_folders,
        'has_root_images': has_root_images,
        'total_images': total_images,
        'class_names': class_names,
        'class_counts': class_counts,
        'needs_split': not already_split and (has_class_folders or has_root_images)
    }

def get_class_distribution(dataset_dir):
    """
    Get the distribution of classes in the dataset.
    
    Args:
        dataset_dir (str): Path to the dataset directory
    
    Returns:
        dict: Dictionary with class distribution information
    """
    # Analyze the dataset structure
    analysis = analyze_dataset_structure(dataset_dir)
    
    if analysis['already_split']:
        # Dataset is already split, return the class counts from analysis
        return {
            'class_names': analysis['class_names'],
            'class_counts': analysis['class_counts'],
            'total_images': analysis['total_images']
        }
    
    # If not already split, get counts from class folders or root images
    class_counts = {}
    total_images = 0
    
    if analysis['has_class_folders']:
        # Count images in class folders
        for class_name in analysis['class_names']:
            class_path = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                class_counts[class_name] = len(images)
                total_images += len(images)
    
    elif analysis['has_root_images']:
        # Count images in the root folder
        root_images = [f for f in os.listdir(dataset_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts['unclassified'] = len(root_images)
        total_images = len(root_images)
    
    return {
        'class_names': analysis['class_names'],
        'class_counts': class_counts,
        'total_images': total_images
    }

def is_kaggle_dataset(dataset_dir):
    """
    Check if the directory is likely to be the Kaggle Plant Disease dataset.
    
    Args:
        dataset_dir (str): Path to the dataset directory
    
    Returns:
        bool: True if it's likely the Kaggle Plant Disease dataset
    """
    # Common class names in the plant disease dataset
    common_classes = [
        'Tomato_healthy', 'Tomato_Early_blight', 'Tomato_Late_blight',
        'Apple_healthy', 'Apple_Black_rot', 'Apple_Cedar_apple_rust',
        'Grape_healthy', 'Grape_Black_rot', 'Grape_Leaf_blight'
    ]
    
    # Check if at least a few of these classes exist
    found_classes = 0
    
    # Check for these class names in the root directory
    for cls in common_classes:
        if os.path.isdir(os.path.join(dataset_dir, cls)):
            found_classes += 1
    
    # Check if there's a train directory and check there too
    train_dir = os.path.join(dataset_dir, 'train')
    if os.path.isdir(train_dir):
        for cls in common_classes:
            if os.path.isdir(os.path.join(train_dir, cls)):
                found_classes += 1
    
    # If we found at least 3 of the common classes, it's likely the Kaggle dataset
    return found_classes >= 3

if __name__ == "__main__":
    # Example usage
    input_dir = "path/to/dataset"
    output_dir = "path/to/output"
    
    # Analyze dataset
    analysis = analyze_dataset_structure(input_dir)
    
    if analysis['needs_split']:
        print(f"Dataset needs splitting. Found {analysis['total_images']} images in {len(analysis['class_names'])} classes.")
        
        # Split dataset
        split_results = split_dataset(input_dir, output_dir)
        
        print("Dataset split complete!")
        print(f"Train images: {split_results['stats']['train_count']}")
        print(f"Validation images: {split_results['stats']['val_count']}")
        print(f"Test images: {split_results['stats']['test_count']}")
    else:
        print("Dataset is already split or has an unexpected structure.")
