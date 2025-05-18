import os
import torch
import json
import onnx
import onnxruntime
import numpy as np
from utils import create_model, convert_to_onnx

def export_to_onnx(model_path, output_path, input_size=224, device_name="cuda"):
    # Setup device
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    
    # Load class info
    with open('class_info.json', 'r') as f:
        class_info = json.load(f)
    
    num_classes = len(class_info['class_names'])
    
    # Create model
    model = create_model(num_classes)
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Convert to ONNX
    success = convert_to_onnx(model, input_size, output_path)
    
    if success:
        print(f"Model successfully exported to ONNX format at: {output_path}")
        
        # Verify the ONNX model
        print("Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
        
        # Test ONNX model with a sample input
        print("Testing ONNX model with a sample input...")
        
        # Create a random sample input
        sample_input = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
        
        # Get PyTorch model prediction
        torch_input = torch.from_numpy(sample_input).to(device)
        with torch.no_grad():
            torch_output = model(torch_input).cpu().numpy()
        
        # Get ONNX model prediction
        ort_session = onnxruntime.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: sample_input}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        np.testing.assert_allclose(torch_output, ort_output, rtol=1e-03, atol=1e-05)
        print("PyTorch and ONNX model outputs match!")
        
        # Save model metadata
        metadata = {
            'input_size': input_size,
            'num_classes': num_classes,
            'class_names': class_info['class_names'],
            'class_to_idx': class_info['class_to_idx'],
            'pytorch_model_path': model_path,
            'onnx_model_path': output_path
        }
        
        metadata_path = os.path.join(os.path.dirname(output_path), 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"Model metadata saved to: {metadata_path}")
        
        return True, metadata
    else:
        print("Failed to export model to ONNX format.")
        return False, None

if __name__ == "__main__":
    # Example usage
    model_path = "output/best_model.pth"
    output_path = "output/plant_disease_model.onnx"
    
    success, metadata = export_to_onnx(
        model_path=model_path,
        output_path=output_path
    )
