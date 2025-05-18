import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from utils import plot_training_curves, load_training_history
import cv2

def visualize_training_results(history_path):
    """Visualize training and validation curves from history file."""
    # Load training history
    if isinstance(history_path, str):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = history_path
    
    # Plot training curves
    return plot_training_curves(history)

def visualize_learning_rate(history_path):
    """Visualize learning rate schedule."""
    # Load training history
    if isinstance(history_path, str):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = history_path
    
    # Plot learning rate curve
    plt.figure(figsize=(10, 4))
    epochs = list(range(1, len(history['learning_rates']) + 1))
    plt.plot(epochs, history['learning_rates'], 'b-')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_rate_schedule.png')
    return plt.gcf()

def visualize_class_distribution(data_dir):
    """Visualize class distribution in dataset."""
    # Get class folders
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    # Get class names and counts
    class_names = sorted(os.listdir(train_dir))
    train_counts = []
    val_counts = []
    test_counts = []
    
    for class_name in class_names:
        # Count images in each class
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        if os.path.isdir(train_class_dir):
            train_counts.append(len([f for f in os.listdir(train_class_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]))
        else:
            train_counts.append(0)
        
        if os.path.isdir(val_class_dir):
            val_counts.append(len([f for f in os.listdir(val_class_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]))
        else:
            val_counts.append(0)
        
        if os.path.isdir(test_class_dir):
            test_counts.append(len([f for f in os.listdir(test_class_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]))
        else:
            test_counts.append(0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Class': class_names,
        'Train': train_counts,
        'Validation': val_counts,
        'Test': test_counts
    })
    
    # Plot class distribution
    plt.figure(figsize=(15, 8))
    df_melted = df.melt(id_vars=['Class'], value_vars=['Train', 'Validation', 'Test'],
                        var_name='Split', value_name='Count')
    
    sns.barplot(x='Class', y='Count', hue='Split', data=df_melted)
    plt.title('Class Distribution Across Splits')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    return plt.gcf()

def visualize_sample_images(data_dir, num_classes=10, num_samples=5):
    """Visualize sample images from each class."""
    # Get train directory and class names
    train_dir = os.path.join(data_dir, 'train')
    class_names = sorted(os.listdir(train_dir))
    
    # Limit number of classes to display
    if num_classes > len(class_names):
        num_classes = len(class_names)
    
    # Select classes to display
    if len(class_names) > num_classes:
        # Equally space the classes
        indices = np.linspace(0, len(class_names) - 1, num_classes, dtype=int)
        selected_classes = [class_names[i] for i in indices]
    else:
        selected_classes = class_names[:num_classes]
    
    # Setup figure
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(15, 2 * num_classes))
    
    # If only one class, make sure axes is 2D
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    # For each selected class
    for i, class_name in enumerate(selected_classes):
        class_dir = os.path.join(train_dir, class_name)
        
        if os.path.isdir(class_dir):
            # Get all image files
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Select random samples
            if len(image_files) > num_samples:
                selected_images = np.random.choice(image_files, num_samples, replace=False)
            else:
                selected_images = image_files[:num_samples]
            
            # Display each sample
            for j, img_file in enumerate(selected_images):
                if j < num_samples:  # Extra check to avoid index errors
                    img_path = os.path.join(class_dir, img_file)
                    img = Image.open(img_path).convert('RGB')
                    
                    # Display image
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    
                    # Only add class name to first image in row
                    if j == 0:
                        axes[i, j].set_title(class_name, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    return plt.gcf()

def visualize_misclassified_examples(test_results_path, data_dir, num_examples=10):
    """Visualize misclassified examples from test results."""
    # Load test results
    with open(test_results_path, 'r') as f:
        test_results = json.load(f)
    
    # Filter misclassified examples
    misclassified = [result for result in test_results if not result['correct']]
    
    # Limit number of examples
    if len(misclassified) > num_examples:
        misclassified = np.random.choice(misclassified, num_examples, replace=False)
    
    # Setup figure
    fig, axes = plt.subplots(len(misclassified), 1, figsize=(10, 4 * len(misclassified)))
    
    # If only one example, make sure axes is an array
    if len(misclassified) == 1:
        axes = [axes]
    
    # Display each misclassified example
    for i, result in enumerate(misclassified):
        # Load image
        img_path = result['image_path']
        img = Image.open(img_path).convert('RGB')
        
        # Display image and prediction info
        axes[i].imshow(img)
        axes[i].set_title(f"True: {result['true_class']}, Predicted: {result['pred_class']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    return plt.gcf()

def visualize_prediction_confidence(test_results_path):
    """Visualize prediction confidence for correct and incorrect predictions."""
    # Load test results
    with open(test_results_path, 'r') as f:
        test_results = json.load(f)
    
    # Extract confidence scores
    correct_conf = []
    incorrect_conf = []
    
    for result in test_results:
        # Get confidence score for predicted class
        pred_idx = result['pred_label']
        conf = result['probabilities'][pred_idx]
        
        if result['correct']:
            correct_conf.append(conf)
        else:
            incorrect_conf.append(conf)
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 20)
    
    plt.hist(correct_conf, bins=bins, alpha=0.5, label='Correct Predictions')
    plt.hist(incorrect_conf, bins=bins, alpha=0.5, label='Incorrect Predictions')
    
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Predictions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_confidence.png')
    return plt.gcf()

def visualize_grad_cam(model_path, image_path, class_names_path, output_path, device_name="cuda"):
    """Visualize Grad-CAM for a given image."""
    # Setup device
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    
    # Load class names
    with open(class_names_path, 'r') as f:
        class_info = json.load(f)
    class_names = class_info['class_names']
    
    # Load model
    num_classes = len(class_names)
    model = create_model(num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()
    
    pred_class = class_names[pred_idx]
    confidence = prob[0][pred_idx].item()
    
    # Get feature maps from the last convolutional layer
    feature_maps = None
    def save_feature_maps(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach()
    
    # Attach hook to get feature maps
    model.layer4.register_forward_hook(save_feature_maps)
    
    # Forward pass again to get feature maps
    output = model(input_tensor)
    
    # Get gradients using backward pass
    model.zero_grad()
    loss = output[0, pred_idx]
    loss.backward()
    
    # Get gradients from the last convolutional layer
    gradients = None
    def save_gradients(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()
    
    # Attach hook to get gradients
    model.layer4.register_backward_hook(save_gradients)
    
    # Forward and backward pass again to get gradients
    model.zero_grad()
    output = model(input_tensor)
    loss = output[0, pred_idx]
    loss.backward()
    
    # Compute Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(feature_maps.shape[1]):
        feature_maps[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(feature_maps, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Resize heatmap to match original image
    img_np = np.array(img)
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # Apply colormap to heatmap
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    superimposed = cv2.addWeighted(img_np[:, :, ::-1], 0.6, heatmap, 0.4, 0)
    
    # Create figure with original image and Grad-CAM visualization
    plt.figure(figsize=(12, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original Image")
    plt.axis('off')
    
    # Grad-CAM visualization
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed[:, :, ::-1])
    plt.title(f"Grad-CAM: {pred_class} ({confidence:.2f})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    return plt.gcf()

if __name__ == "__main__":
    # Example usage
    history_path = "output/training_history.json"
    data_dir = "path/to/dataset"
    test_results_path = "test_results/test_results.json"
    
    # Visualize training results
    visualize_training_results(history_path)
    
    # Visualize learning rate
    visualize_learning_rate(history_path)
    
    # Visualize class distribution
    visualize_class_distribution(data_dir)
    
    # Visualize sample images
    visualize_sample_images(data_dir)
    
    # Visualize misclassified examples
    visualize_misclassified_examples(test_results_path, data_dir)
    
    # Visualize prediction confidence
    visualize_prediction_confidence(test_results_path)
    
    # Visualize Grad-CAM
    model_path = "output/best_model.pth"
    image_path = "path/to/image.jpg"
    class_names_path = "class_info.json"
    output_path = "grad_cam.png"
    
    visualize_grad_cam(model_path, image_path, class_names_path, output_path)
