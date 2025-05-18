import os
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from utils import (
    get_data_loaders, create_model, load_checkpoint,
    plot_confusion_matrix, get_classification_report, plot_class_accuracy
)

def validate_model(data_dir, model_path, output_dir, device_name="cuda"):
    # Setup device
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data loaders
    _, val_loader, _, class_names = get_data_loaders(data_dir)
    num_classes = len(class_names)
    
    # Create and load model
    model = create_model(num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Validation phase
    true_labels = []
    pred_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, _ in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Track statistics
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            # Store labels for confusion matrix
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
    
    # Calculate accuracy
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Generate confusion matrix
    cm_fig = plot_confusion_matrix(true_labels, pred_labels, class_names)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    cm_fig.savefig(cm_path)
    
    # Generate classification report
    report = get_classification_report(true_labels, pred_labels, class_names)
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f)
    
    # Generate class accuracy plot
    ca_fig = plot_class_accuracy(report, class_names)
    ca_path = os.path.join(output_dir, 'class_accuracy.png')
    ca_fig.savefig(ca_path)
    
    # Return validation results
    results = {
        'accuracy': accuracy,
        'confusion_matrix_path': cm_path,
        'classification_report_path': report_path,
        'class_accuracy_path': ca_path
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/dataset"
    model_path = "output/best_model.pth"
    output_dir = "validation_results"
    
    results = validate_model(
        data_dir=data_dir,
        model_path=model_path,
        output_dir=output_dir
    )
