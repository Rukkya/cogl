import os
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from utils import (
    get_data_loaders, create_model, load_checkpoint,
    plot_confusion_matrix, get_classification_report, plot_class_accuracy
)

def test_model(data_dir, model_path, output_dir, device_name="cuda"):
    # Setup device
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data loaders
    _, _, test_loader, class_names = get_data_loaders(data_dir)
    num_classes = len(class_names)
    
    # Create and load model
    model = create_model(num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Testing phase
    true_labels = []
    pred_labels = []
    image_paths = []
    probabilities = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, paths in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Track statistics
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # Store results for later
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            image_paths.extend(paths)
            probabilities.extend(probs.cpu().numpy())
    
    # Calculate accuracy
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate confusion matrix
    cm_fig = plot_confusion_matrix(true_labels, pred_labels, class_names)
    cm_path = os.path.join(output_dir, 'test_confusion_matrix.png')
    cm_fig.savefig(cm_path)
    
    # Generate classification report
    report = get_classification_report(true_labels, pred_labels, class_names)
    report_path = os.path.join(output_dir, 'test_classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f)
    
    # Generate class accuracy plot
    ca_fig = plot_class_accuracy(report, class_names)
    ca_path = os.path.join(output_dir, 'test_class_accuracy.png')
    ca_fig.savefig(ca_path)
    
    # Save test results
    test_results = []
    for i in range(len(image_paths)):
        test_results.append({
            'image_path': image_paths[i],
            'true_label': int(true_labels[i]),
            'true_class': class_names[int(true_labels[i])],
            'pred_label': int(pred_labels[i]),
            'pred_class': class_names[int(pred_labels[i])],
            'probabilities': probabilities[i].tolist(),
            'correct': int(true_labels[i]) == int(pred_labels[i])
        })
    
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f)
    
    # Return test results
    results = {
        'accuracy': accuracy,
        'confusion_matrix_path': cm_path,
        'classification_report_path': report_path,
        'class_accuracy_path': ca_path,
        'results_path': results_path,
        'test_results': test_results
    }
    
    return results

def predict_image(image_path, model_path, class_names, device_name="cuda"):
    # Setup device
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    
    # Load class names if provided as path
    if isinstance(class_names, str) and os.path.exists(class_names):
        with open(class_names, 'r') as f:
            class_info = json.load(f)
            class_names = class_info['class_names']
    
    num_classes = len(class_names)
    
    # Create and load model
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
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Get prediction and probability
    pred_idx = preds.item()
    pred_class = class_names[pred_idx]
    probabilities = probs[0].cpu().numpy()
    
    result = {
        'image_path': image_path,
        'predicted_class': pred_class,
        'predicted_index': pred_idx,
        'probabilities': probabilities.tolist()
    }
    
    return result, image

if __name__ == "__main__":
    # Example usage for testing dataset
    data_dir = "path/to/dataset"
    model_path = "output/best_model.pth"
    output_dir = "test_results"
    
    results = test_model(
        data_dir=data_dir,
        model_path=model_path,
        output_dir=output_dir
    )
    
    # Example usage for single image prediction
    image_path = "path/to/image.jpg"
    class_info_path = "class_info.json"
    
    result, image = predict_image(
        image_path=image_path,
        model_path=model_path,
        class_names=class_info_path
    )
    
    print(f"Predicted class: {result['predicted_class']}")
