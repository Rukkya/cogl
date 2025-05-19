import streamlit as st
import os
import time
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import shutil
import sys
import onnx
import onnxruntime as ort

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from validate import validate_model
from ttest import test_model, predict_image
from visualize import (
    visualize_class_distribution,
    visualize_sample_images,
    visualize_misclassified_examples,
    visualize_prediction_confidence,
    visualize_grad_cam
)
from dataset_splitter import (
    analyze_dataset_structure,
    split_dataset,
    get_class_distribution,
    is_kaggle_dataset
)

# Set page config
st.set_page_config(
    page_title="Plant Disease Classification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set styles
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
    }
</style>
""", unsafe_allow_html=True)

# Create directory if it doesn't exist
def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

# Initialize session state
if 'model_info' not in st.session_state:
    st.session_state.model_info = {
        'model_loaded': False,
        'model_path': None,
        'onnx_model_path': None,
        'pth_model_path': None,
        'class_names': None,
        'data_dir': None,
        'output_dir': None,
        'test_results': None
    }

# Helper functions
def display_progress_bar(iteration, total, description=""):
    """Custom progress bar using Streamlit."""
    progress_bar = st.progress(0)
    progress_text = st.empty()
    for i in range(iteration + 1):
        progress = int(100 * i / total)
        progress_bar.progress(progress)
        progress_text.text(f"{description} {progress}%")
        time.sleep(0.01)
    progress_text.text(f"{description} Complete!")

# Function to get ONNX model information
def get_onnx_model_info(model_path):
    """Extract information from an ONNX model."""
    try:
        # Load the ONNX model
        model = onnx.load(model_path)

        # Get model metadata
        metadata = {}
        if model.metadata_props:
            for prop in model.metadata_props:
                metadata[prop.key] = prop.value

        # Get input and output shapes
        input_shape = None
        output_shape = None

        if model.graph.input:
            input_info = model.graph.input[0]
            if hasattr(input_info.type.tensor_type, 'shape'):
                input_shape = []
                for dim in input_info.type.tensor_type.shape.dim:
                    if hasattr(dim, 'dim_value'):
                        input_shape.append(dim.dim_value)
                    else:
                        input_shape.append(None)  # For dynamic dimensions

        if model.graph.output:
            output_info = model.graph.output[0]
            if hasattr(output_info.type.tensor_type, 'shape'):
                output_shape = []
                for dim in output_info.type.tensor_type.shape.dim:
                    if hasattr(dim, 'dim_value'):
                        output_shape.append(dim.dim_value)
                    else:
                        output_shape.append(None)  # For dynamic dimensions

        # Create a session to get providers
        sess = ort.InferenceSession(model_path)
        providers = sess.get_providers()

        return {
            "metadata": metadata,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "providers": providers,
            "version": model.ir_version,
            "graph_name": model.graph.name if model.graph.name else "Unknown"
        }
    except Exception as e:
        return {"error": str(e)}

def convert_pth_to_onnx(pth_model_path, model_name, output_dir):
    """
    Convert a PyTorch model (.pth) to ONNX format.

    Args:
        pth_model_path (str): Path to the PyTorch model file (.pth).
        model_name (str): Name of the model.
        output_dir (str): Directory to save the ONNX model.

    Returns:
        str: Path to the converted ONNX model.
    """
    import torch
    import torchvision.models as models

    # Load the PyTorch model
    model = torch.load(pth_model_path)

    # If the model is a state_dict, we need to create a model instance and load the state_dict
    if isinstance(model, dict):
        # Create a model instance (you might need to adjust this based on your model architecture)
        # This is a placeholder - you should replace it with your actual model architecture
        model_instance = models.resnet18(pretrained=False)
        num_classes = len(st.session_state.model_info['class_names'])
        model_instance.fc = torch.nn.Linear(model_instance.fc.in_features, num_classes)

        # Load the state_dict
        model_instance.load_state_dict(model)
        model = model_instance

    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the input size as needed

    # Define the ONNX model path
    onnx_model_path = os.path.join(output_dir, f"{model_name}.onnx")

    # Export the model to ONNX
    torch.onnx.export(
        model,                      # Model being exported
        dummy_input,                # Model input (or a tuple for multiple inputs)
        onnx_model_path,            # Where to save the model
        export_params=True,         # Store the trained parameter weights inside the model file
        opset_version=11,           # The ONNX version to export the model to
        do_constant_folding=True,   # Whether to execute constant folding for optimization
        input_names=['input'],      # The model's input names
        output_names=['output'],    # The model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},  # Variable length axes
            'output': {0: 'batch_size'}
        }
    )

    return onnx_model_path

# Function to make predictions using ONNX
def predict_with_onnx(image_path, onnx_model_path, class_names):
    """Make predictions using ONNX model."""
    try:
        # Load the ONNX model
        session = ort.InferenceSession(onnx_model_path)

        # Get input details
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Preprocess the image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))  # Adjust to match model input size

        # Convert to numpy and normalize
        img_np = np.array(img, dtype=np.float32)
        img_np = img_np / 255.0  # Normalize to [0, 1]

        # Standard normalization (commonly used for ImageNet models)
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        img_np = (img_np - mean) / std

        # Transpose to NCHW format (needed for most models)
        img_np = img_np.transpose((2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension

        # Make prediction
        outputs = session.run([output_name], {input_name: img_np})
        probabilities = outputs[0][0]

        # Get top prediction
        predicted_idx = np.argmax(probabilities)

        if isinstance(class_names, list):
            predicted_class = class_names[predicted_idx]
        else:  # If class_names is a path to a JSON file
            with open(class_names, 'r') as f:
                class_dict = json.load(f)
                class_names_list = [class_dict[str(i)] for i in range(len(class_dict))]
                predicted_class = class_names_list[predicted_idx]

        result = {
            'predicted_class': predicted_class,
            'predicted_idx': int(predicted_idx),
            'probabilities': probabilities.tolist()
        }

        return result, probabilities
    except Exception as e:
        return {"error": str(e)}, None

# Sidebar
st.sidebar.title("🌿 Plant Disease Classification")
st.sidebar.markdown("---")

# Main tabs
tabs = st.tabs([
    "📊 Dataset Configuration",
    "🔄 Model Upload",
    "📈 Visualization",
    "🔍 Validation",
    "🧪 Testing",
    "📦 Export Model"
])

# Tab 1: Dataset Configuration
with tabs[0]:
    st.header("Dataset Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Path")
        data_dir = st.text_input(
            "Enter the path to your dataset directory",
            value="",
            help="Path to the dataset directory. Can be a directory with class folders or a directory with train/valid/test subdirectories."
        )

        if st.button("Analyze Dataset"):
            if not data_dir or not os.path.exists(data_dir):
                st.error("Invalid dataset path. Please enter a valid directory path.")
            else:
                with st.spinner("Analyzing dataset structure..."):
                    # Analyze dataset structure
                    analysis = analyze_dataset_structure(data_dir)

                    if analysis['already_split']:
                        st.success("Dataset is already split into train, validation, and test sets.")
                        st.session_state.model_info['data_dir'] = data_dir
                        st.session_state.model_info['class_names'] = analysis['class_names']

                    elif analysis['needs_split']:
                        st.warning(f"Dataset needs to be split. Found {analysis['total_images']} images in {len(analysis['class_names'])} classes.")
                        st.info("Use the 'Split Dataset' section to create train, validation, and test sets.")

                        # Store dataset analysis in session state
                        if 'dataset_analysis' not in st.session_state:
                            st.session_state.dataset_analysis = {}

                        st.session_state.dataset_analysis = analysis
                    else:
                        st.error("Unknown dataset structure. Please make sure the dataset contains image files.")

    with col2:
        st.subheader("Output Directory")
        output_dir = st.text_input(
            "Enter the path to save model outputs",
            value="output",
            help="All model outputs will be saved to this directory."
        )

        if st.button("Set Output Directory"):
            try:
                ensure_dir(output_dir)
                st.success(f"Output directory set to {output_dir}")
                st.session_state.model_info['output_dir'] = output_dir
            except Exception as e:
                st.error(f"Error creating output directory: {str(e)}")

    # Dataset splitting section (show only if dataset needs splitting)
    if 'dataset_analysis' in st.session_state and st.session_state.dataset_analysis.get('needs_split', False):
        st.markdown("---")
        st.subheader("Split Dataset")

        col1, col2 = st.columns(2)

        with col1:
            # Split ratios
            train_ratio = st.slider("Training Set Ratio", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
            val_ratio = st.slider("Validation Set Ratio", min_value=0.05, max_value=0.3, value=0.15, step=0.05)
            test_ratio = st.slider("Test Set Ratio", min_value=0.05, max_value=0.3, value=0.15, step=0.05)

            # Ensure ratios sum to 1
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 1e-5:
                st.warning(f"Ratios sum to {total_ratio:.2f}, not 1.0. Please adjust.")

        with col2:
            # Split destination
            split_output_dir = st.text_input(
                "Split Output Directory",
                value=os.path.join(output_dir, "split_dataset"),
                help="Directory where the split dataset will be saved."
            )

            # Random seed
            random_seed = st.number_input("Random Seed", min_value=1, max_value=1000, value=42)

            # Split button
            split_button = st.button("Split Dataset")

        if split_button:
            if abs(total_ratio - 1.0) > 1e-5:
                st.error(f"Ratios must sum to 1.0, but they sum to {total_ratio:.2f}. Please adjust.")
            else:
                with st.spinner("Splitting dataset..."):
                    try:
                        # Split dataset
                        split_results = split_dataset(
                            input_dir=data_dir,
                            output_dir=split_output_dir,
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                            test_ratio=test_ratio,
                            random_seed=random_seed
                        )

                        # Update session state
                        st.session_state.model_info['data_dir'] = split_output_dir

                        # Get class names from the new train directory
                        train_dir = os.path.join(split_output_dir, 'train')
                        class_names = sorted([d for d in os.listdir(train_dir)
                                             if os.path.isdir(os.path.join(train_dir, d))])

                        st.session_state.model_info['class_names'] = class_names

                        # Success message with stats
                        st.success(f"Dataset successfully split into train, validation, and test sets!")
                        st.info(f"Training set: {split_results['stats']['train_count']} images ({train_ratio:.0%})\n"
                               f"Validation set: {split_results['stats']['val_count']} images ({val_ratio:.0%})\n"
                               f"Test set: {split_results['stats']['test_count']} images ({test_ratio:.0%})")

                        # Remove dataset analysis from session state
                        if 'dataset_analysis' in st.session_state:
                            del st.session_state.dataset_analysis

                    except Exception as e:
                        st.error(f"Error splitting dataset: {str(e)}")
                        st.exception(e)

    # Display dataset statistics if loaded
    if st.session_state.model_info['data_dir']:
        st.markdown("---")
        st.subheader("Dataset Statistics")

        # Check if the dataset directory exists and has the expected structure
        data_dir = st.session_state.model_info['data_dir']
        train_dir = os.path.join(data_dir, 'train')
        valid_dir = os.path.join(data_dir, 'valid')
        test_dir = os.path.join(data_dir, 'test')

        if not os.path.exists(train_dir) or not os.path.exists(valid_dir) or not os.path.exists(test_dir):
            st.warning("Dataset does not have the expected structure (train/valid/test directories).")
        else:
            # Get statistics
            class_names = st.session_state.model_info['class_names']
            train_counts = []
            val_counts = []
            test_counts = []

            for class_name in class_names:
                # Count images in each class
                train_class_dir = os.path.join(train_dir, class_name)
                val_class_dir = os.path.join(valid_dir, class_name)
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
                'Test': test_counts,
                'Total': [t + v + ts for t, v, ts in zip(train_counts, val_counts, test_counts)]
            })

            col1, col2 = st.columns([2, 1])

            with col1:
                # Plot class distribution
                df_melted = df.melt(id_vars=['Class'], value_vars=['Train', 'Validation', 'Test'],
                                  var_name='Split', value_name='Count')

                fig = px.bar(df_melted, x='Class', y='Count', color='Split', barmode='group',
                           title='Class Distribution Across Splits')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)

            with col2:
                # Display dataset summary
                st.dataframe(df)

                total_images = sum(train_counts) + sum(val_counts) + sum(test_counts)
                st.write(f"Total classes: {len(class_names)}")
                st.write(f"Total images: {total_images}")
                st.write(f"Training images: {sum(train_counts)}")
                st.write(f"Validation images: {sum(val_counts)}")
                st.write(f"Testing images: {sum(test_counts)}")

                # Check if dataset is imbalanced
                if min(df['Total']) < max(df['Total']) * 0.3:
                    st.warning("Dataset is imbalanced. Some classes have much fewer images than others.")
                    st.info("Consider this when evaluating model performance.")

            # Sample images
            st.subheader("Sample Images")
            with st.expander("Show sample images from each class"):
                # Display sample images
                cols = 5  # Number of sample images per class
                for class_name in class_names[:5]:  # Limit to first 5 classes to avoid too many images
                    st.write(f"**Class: {class_name}**")

                    # Get image files
                    train_class_dir = os.path.join(train_dir, class_name)
                    if os.path.isdir(train_class_dir):
                        image_files = [f for f in os.listdir(train_class_dir)
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                        if image_files:
                            # Sample up to 5 images
                            sample_files = image_files[:cols] if len(image_files) >= cols else image_files

                            # Create columns for images
                            image_cols = st.columns(len(sample_files))

                            # Display each image
                            for i, img_file in enumerate(sample_files):
                                img_path = os.path.join(train_class_dir, img_file)
                                try:
                                    img = Image.open(img_path)
                                    image_cols[i].image(img, width=150)
                                except Exception as e:
                                    image_cols[i].error(f"Error loading image: {str(e)}")
                        else:
                            st.info(f"No images found for class {class_name}")
                    else:
                        st.warning(f"Directory for class {class_name} not found")

# Tab 2: Model Upload
with tabs[1]:
    st.header("Upload Model")

    # Upload model section
    st.subheader("Upload Your Model")

    model_type = st.radio(
        "Select model type:",
        options=["ONNX Model", "PyTorch Model (.pth)"],
        index=0
    )

    if model_type == "ONNX Model":
        model_upload = st.file_uploader("Choose an ONNX model file (.onnx)", type=["onnx"])
    else:
        model_upload = st.file_uploader("Choose a PyTorch model file (.pth)", type=["pth"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Names")
        class_names_method = st.radio(
            "Class Names Source",
            options=["From Dataset", "Manual Entry", "Upload JSON File"],
            index=0
        )

        if class_names_method == "From Dataset":
            if st.session_state.model_info['class_names']:
                st.success(f"Using class names from dataset: {', '.join(st.session_state.model_info['class_names'])}")
            else:
                st.warning("No dataset loaded. Please load a dataset first or choose another method.")

        elif class_names_method == "Manual Entry":
            class_names_text = st.text_area(
                "Enter class names (one per line)",
                height=150,
                help="Enter each class name on a new line. Order matters!"
            )

            if class_names_text:
                class_names = [name.strip() for name in class_names_text.split('\n') if name.strip()]
                st.session_state.model_info['class_names'] = class_names
                st.success(f"Defined {len(class_names)} classes.")

        elif class_names_method == "Upload JSON File":
            class_names_file = st.file_uploader("Upload class names JSON file", type=["json"])

            if class_names_file:
                try:
                    class_names_json = json.load(class_names_file)

                    # Handle different JSON formats
                    if isinstance(class_names_json, list):
                        class_names = class_names_json
                    elif isinstance(class_names_json, dict):
                        # If it's a dict, try to sort by key values
                        try:
                            # Convert string indices to integers if possible
                            indices = sorted([int(k) for k in class_names_json.keys()])
                            class_names = [class_names_json[str(i)] for i in indices]
                        except:
                            # If conversion fails, just take the values
                            class_names = list(class_names_json.values())

                    st.session_state.model_info['class_names'] = class_names
                    st.success(f"Loaded {len(class_names)} classes from JSON file.")

                    # Save the class names to a file
                    if st.session_state.model_info['output_dir']:
                        class_names_path = os.path.join(st.session_state.model_info['output_dir'], 'class_names.json')
                        with open(class_names_path, 'w') as f:
                            json.dump(class_names, f)
                except Exception as e:
                    st.error(f"Error loading class names JSON: {str(e)}")

    with col2:
        st.subheader("Model Settings")
        model_name = st.text_input("Model Name", value="plant_disease_model")

        # Get available providers
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        execution_provider = st.selectbox("Execution Provider", options=providers)

        # Load model button
        load_model_button = st.button("Load Model")

    # Process the uploaded model
    if model_upload is not None and load_model_button:
        # Create directory for model if it doesn't exist
        if st.session_state.model_info['output_dir']:
            model_dir = os.path.join(st.session_state.model_info['output_dir'], 'models')
            ensure_dir(model_dir)

            # Save the uploaded model
            if model_type == "ONNX Model":
                model_path = os.path.join(model_dir, f"{model_name}.onnx")
                with open(model_path, "wb") as f:
                    f.write(model_upload.getbuffer())

                with st.spinner("Loading ONNX model..."):
                    try:
                        # Get model info
                        model_info = get_onnx_model_info(model_path)

                        if "error" in model_info:
                            st.error(f"Error loading model: {model_info['error']}")
                        else:
                            # Update session state
                            st.session_state.model_info['model_loaded'] = True
                            st.session_state.model_info['onnx_model_path'] = model_path

                            # Display success message
                            st.success(f"Model successfully loaded from {model_path}")

                            # Display model info
                            st.subheader("Model Information")

                            # Display input/output shapes
                            cols = st.columns(2)
                            with cols[0]:
                                st.write("**Input Shape:**", model_info['input_shape'])
                            with cols[1]:
                                st.write("**Output Shape:**", model_info['output_shape'])

                            # Display providers
                            st.write("**Available Providers:**", ", ".join(model_info['providers']))

                            # Display metadata
                            if model_info['metadata']:
                                st.write("**Metadata:**")
                                for key, value in model_info['metadata'].items():
                                    st.write(f"- {key}: {value}")

                            # Check if class count matches
                            if st.session_state.model_info['class_names']:
                                num_classes = len(st.session_state.model_info['class_names'])
                                if model_info['output_shape'] and model_info['output_shape'][-1] != num_classes:
                                    st.warning(f"Model output size ({model_info['output_shape'][-1]}) "
                                              f"doesn't match number of classes ({num_classes})!")

                    except Exception as e:
                        st.error(f"Error processing model: {str(e)}")
                        st.exception(e)
            else:
                model_path = os.path.join(model_dir, f"{model_name}.pth")
                with open(model_path, "wb") as f:
                    f.write(model_upload.getbuffer())

                # Add option to convert to ONNX
                convert_to_onnx = st.checkbox("Convert to ONNX format", value=True)

                if convert_to_onnx:
                    with st.spinner("Converting PyTorch model to ONNX..."):
                        try:
                            # Convert PyTorch model to ONNX
                            onnx_model_path = convert_pth_to_onnx(model_path, model_name, model_dir)

                            # Get model info
                            model_info = get_onnx_model_info(onnx_model_path)

                            if "error" in model_info:
                                st.error(f"Error loading converted model: {model_info['error']}")
                            else:
                                # Update session state
                                st.session_state.model_info['model_loaded'] = True
                                st.session_state.model_info['onnx_model_path'] = onnx_model_path

                                # Display success message
                                st.success(f"Model successfully converted to ONNX and loaded from {onnx_model_path}")

                                # Display model info
                                st.subheader("Model Information")

                                # Display input/output shapes
                                cols = st.columns(2)
                                with cols[0]:
                                    st.write("**Input Shape:**", model_info['input_shape'])
                                with cols[1]:
                                    st.write("**Output Shape:**", model_info['output_shape'])

                                # Display providers
                                st.write("**Available Providers:**", ", ".join(model_info['providers']))

                                # Display metadata
                                if model_info['metadata']:
                                    st.write("**Metadata:**")
                                    for key, value in model_info['metadata'].items():
                                        st.write(f"- {key}: {value}")

                                # Check if class count matches
                                if st.session_state.model_info['class_names']:
                                    num_classes = len(st.session_state.model_info['class_names'])
                                    if model_info['output_shape'] and model_info['output_shape'][-1] != num_classes:
                                        st.warning(f"Model output size ({model_info['output_shape'][-1]}) "
                                                  f"doesn't match number of classes ({num_classes})!")

                        except Exception as e:
                            st.error(f"Error converting model: {str(e)}")
                            st.exception(e)
                else:
                    st.warning("Using PyTorch model directly is not fully supported in this interface. Please convert to ONNX for full functionality.")
                    st.session_state.model_info['model_loaded'] = True
                    st.session_state.model_info['pth_model_path'] = model_path
                    st.success(f"PyTorch model loaded from {model_path}")
        else:
            st.error("Please set an output directory first.")

    # If model loaded, display a test section
    if st.session_state.model_info['model_loaded']:
        st.markdown("---")
        st.subheader("Quick Test")

        test_image = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])

        if test_image:
            # Save the image
            temp_dir = "temp_uploads"
            ensure_dir(temp_dir)

            test_path = os.path.join(temp_dir, test_image.name)
            with open(test_path, "wb") as f:
                f.write(test_image.getbuffer())

            # Display the image
            col1, col2 = st.columns(2)

            with col1:
                st.image(test_image, caption="Test Image", use_column_width=True)

            with col2:
                if st.button("Run Prediction"):
                    with st.spinner("Making prediction..."):
                        try:
                            if st.session_state.model_info.get('onnx_model_path'):
                                result, probabilities = predict_with_onnx(
                                    test_path,
                                    st.session_state.model_info['onnx_model_path'],
                                    st.session_state.model_info['class_names']
                                )
                            else:
                                st.error("ONNX model not available for prediction.")
                                result = {"error": "ONNX model not available"}

                            if "error" in result:
                                st.error(f"Prediction error: {result['error']}")
                            else:
                                st.success(f"Predicted class: {result['predicted_class']}")

                                # Show top 5 predictions as a bar chart
                                class_names = st.session_state.model_info['class_names']
                                probs = result['probabilities']

                                # Get top 5 indices
                                top5_indices = np.argsort(probs)[-5:][::-1]
                                top5_names = [class_names[i] for i in top5_indices]
                                top5_probs = [probs[i] for i in top5_indices]

                                # Create bar chart
                                fig = px.bar(
                                    x=top5_probs,
                                    y=top5_names,
                                    orientation='h',
                                    labels={'x': 'Probability', 'y': 'Class'},
                                    title='Top 5 Predictions',
                                    width=400,
                                    height=300
                                )
                                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig)

                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            st.exception(e)

# Tab 3: Visualization
with tabs[2]:
    st.header("Dataset Visualizations")

    # Check if dataset is loaded
    if not st.session_state.model_info['data_dir']:
        st.warning("No dataset loaded. Please load a dataset first.")
    else:
        # Visualization options
        st.subheader("Select Visualizations")

        viz_cols = st.columns(3)

        with viz_cols[0]:
            show_class_distribution = st.checkbox("Class Distribution", value=True)

        with viz_cols[1]:
            show_sample_images = st.checkbox("Sample Images", value=True)

        with viz_cols[2]:
            show_button = st.button("Display Visualizations")

        # Show visualizations if button is clicked
        if show_button:
            with st.spinner("Generating visualizations..."):
                data_dir = st.session_state.model_info['data_dir']

                # Create visualizations
                if show_class_distribution:
                    st.subheader("Class Distribution")
                    fig = visualize_class_distribution(data_dir)
                    st.pyplot(fig)

                if show_sample_images:
                    st.subheader("Sample Images")
                    fig = visualize_sample_images(data_dir, num_classes=5, num_samples=5)
                    st.pyplot(fig)

# Tab 4: Validation
with tabs[3]:
    st.header("Model Validation")

    # Check if model is loaded
    if not st.session_state.model_info['model_loaded']:
        st.warning("No model loaded. Please upload a model first.")
    elif not st.session_state.model_info['data_dir']:
        st.warning("No dataset loaded. Please load a dataset first.")
    else:
        st.info("Currently, validation functionality is limited for ONNX models. Please use the Testing tab for model evaluation.")
        st.warning("Full validation requires PyTorch models. ONNX models can be used for inference but not for detailed validation.")

# Tab 5: Testing
with tabs[4]:
    st.header("Model Testing")

    # Check if model is loaded
    if not st.session_state.model_info['model_loaded']:
        st.warning("No model loaded. Please upload a model first.")
    else:
        testing_tabs = st.tabs(["Single Image Testing", "Batch Testing"])

        # Tab for single image testing
        with testing_tabs[0]:
            st.subheader("Test a Single Image")

            uploaded_file = st.file_uploader("Choose an image to test", type=["jpg", "jpeg", "png"], key="test_tab_image")

            if uploaded_file is not None:
                # Save the uploaded file
                temp_dir = "temp_uploads"
                ensure_dir(temp_dir)

                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Display the uploaded image
                col1, col2 = st.columns(2)

                with col1:
                    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

                with col2:
                    # Run prediction
                    if st.button("Predict"):
                        with st.spinner("Running prediction..."):
                            try:
                                # Run prediction
                                result, probabilities = predict_with_onnx(
                                    file_path,
                                    st.session_state.model_info['onnx_model_path'],
                                    st.session_state.model_info['class_names']
                                )

                                if "error" in result:
                                    st.error(f"Prediction error: {result['error']}")
                                else:
                                    # Display prediction results
                                    st.subheader("Prediction Results")
                                    st.write(f"Predicted class: {result['predicted_class']}")

                                    # Show top 5 predictions
                                    class_names = st.session_state.model_info['class_names']
                                    probs = result['probabilities']

                                    # Get top 5 indices
                                    top5_indices = np.argsort(probs)[-5:][::-1]
                                    top5_names = [class_names[i] for i in top5_indices]
                                    top5_probs = [probs[i] for i in top5_indices]

                                    # Create bar chart
                                    fig = px.bar(
                                        x=top5_probs,
                                        y=top5_names,
                                        orientation='h',
                                        labels={'x': 'Probability', 'y': 'Class'},
                                        title='Top 5 Predictions',
                                        width=400,
                                        height=300
                                    )
                                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                                    st.plotly_chart(fig)

                            except Exception as e:
                                st.error(f"Prediction failed: {str(e)}")
                                st.exception(e)

        # Tab for batch testing
        with testing_tabs[1]:
            st.subheader("Test Multiple Images")

            if not st.session_state.model_info['data_dir']:
                st.warning("No dataset loaded. Please load a dataset first to perform batch testing.")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Run Batch Testing")
                    # Select which dataset split to test on
                    test_split = st.radio(
                        "Select dataset split to test on:",
                        ["test", "valid", "train"],
                        index=0
                    )

                    # Number of images to test
                    max_images = st.slider(
                        "Maximum number of images to test per class",
                        min_value=1,
                        max_value=100,
                        value=10
                    )

                    run_batch_test = st.button("Start Batch Testing")

                with col2:
                    st.subheader("Testing Settings")
                    execution_provider = st.selectbox(
                        "Execution Provider",
                        options=["CPUExecutionProvider", "CUDAExecutionProvider"]
                        if "CUDAExecutionProvider" in ort.get_available_providers()
                        else ["CPUExecutionProvider"],
                        key="batch_exec_provider"
                    )

                # Run batch testing if button is clicked
                if run_batch_test:
                    with st.spinner("Running batch testing..."):
                        try:
                            # Get the path for the selected dataset split
                            data_dir = st.session_state.model_info['data_dir']
                            split_dir = os.path.join(data_dir, test_split)

                            if not os.path.exists(split_dir):
                                st.error(f"Split directory '{test_split}' does not exist.")
                            else:
                                # Create results directory
                                results_dir = os.path.join(st.session_state.model_info['output_dir'], 'batch_test_results')
                                ensure_dir(results_dir)

                                # Get class names
                                class_names = st.session_state.model_info['class_names']

                                # Initialize results dictionary
                                all_results = []
                                class_correct = {cls: 0 for cls in class_names}
                                class_total = {cls: 0 for cls in class_names}

                                # Process each class
                                progress_text = st.empty()
                                progress_bar = st.progress(0)

                                for i, class_name in enumerate(class_names):
                                    class_dir = os.path.join(split_dir, class_name)

                                    if os.path.exists(class_dir) and os.path.isdir(class_dir):
                                        # Get image files
                                        image_files = [
                                            f for f in os.listdir(class_dir)
                                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                                        ]

                                        # Limit number of images
                                        if max_images > 0 and len(image_files) > max_images:
                                            image_files = image_files[:max_images]

                                        # Update progress
                                        progress_text.text(f"Processing class: {class_name} ({i+1}/{len(class_names)})")

                                        # Process each image
                                        for j, img_file in enumerate(image_files):
                                            img_path = os.path.join(class_dir, img_file)

                                            # Update progress
                                            progress = int(100 * (i / len(class_names) + (j / len(image_files)) / len(class_names)))
                                            progress_bar.progress(progress)

                                            # Predict
                                            result, _ = predict_with_onnx(
                                                img_path,
                                                st.session_state.model_info['onnx_model_path'],
                                                class_names
                                            )

                                            if "error" not in result:
                                                # Update counters
                                                true_class = class_name
                                                pred_class = result['predicted_class']
                                                is_correct = (pred_class == true_class)

                                                class_total[true_class] += 1
                                                if is_correct:
                                                    class_correct[true_class] += 1

                                                # Save result
                                                all_results.append({
                                                    'image_path': img_path,
                                                    'true_class': true_class,
                                                    'pred_class': pred_class,
                                                    'correct': is_correct,
                                                    'probabilities': result['probabilities']
                                                })

                                # Compute overall accuracy
                                total_correct = sum(class_correct.values())
                                total_images = sum(class_total.values())
                                overall_accuracy = total_correct / total_images if total_images > 0 else 0

                                # Save results to file
                                results_path = os.path.join(results_dir, 'batch_test_results.json')
                                with open(results_path, 'w') as f:
                                    json.dump(all_results, f)

                                # Calculate class accuracies
                                class_accuracies = {
                                    cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
                                    for cls in class_names
                                }

                                # Create confusion matrix
                                confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
                                for result in all_results:
                                    true_idx = class_names.index(result['true_class'])
                                    pred_idx = class_names.index(result['pred_class'])
                                    confusion_matrix[true_idx, pred_idx] += 1

                                # Save confusion matrix visualization
                                plt.figure(figsize=(12, 10))
                                plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
                                plt.title('Confusion Matrix')
                                plt.colorbar()
                                tick_marks = np.arange(len(class_names))
                                plt.xticks(tick_marks, class_names, rotation=90)
                                plt.yticks(tick_marks, class_names)
                                plt.tight_layout()
                                plt.ylabel('True label')
                                plt.xlabel('Predicted label')

                                confusion_matrix_path = os.path.join(results_dir, 'confusion_matrix.png')
                                plt.savefig(confusion_matrix_path)
                                plt.close()

                                # Create class accuracy visualization
                                plt.figure(figsize=(12, 8))
                                bars = plt.bar(class_names, [class_accuracies[cls] for cls in class_names])
                                plt.title('Class-wise Accuracy')
                                plt.xlabel('Class')
                                plt.ylabel('Accuracy')
                                plt.xticks(rotation=90)
                                plt.tight_layout()

                                # Add accuracy values on top of bars
                                for bar in bars:
                                    height = bar.get_height()
                                    plt.text(
                                        bar.get_x() + bar.get_width()/2.,
                                        height + 0.01,
                                        f'{height:.2f}',
                                        ha='center',
                                        va='bottom',
                                        rotation=0
                                    )

                                class_accuracy_path = os.path.join(results_dir, 'class_accuracy.png')
                                plt.savefig(class_accuracy_path)
                                plt.close()

                                # Store results
                                test_results = {
                                    'accuracy': overall_accuracy,
                                    'class_accuracies': class_accuracies,
                                    'confusion_matrix_path': confusion_matrix_path,
                                    'class_accuracy_path': class_accuracy_path,
                                    'results_path': results_path
                                }

                                st.session_state.model_info['test_results'] = test_results

                                # Display results
                                st.success(f"Testing completed with accuracy: {overall_accuracy:.4f}")

                                # Show confusion matrix
                                st.subheader("Confusion Matrix")
                                cm_image = Image.open(confusion_matrix_path)
                                st.image(cm_image, caption="Confusion Matrix", use_column_width=True)

                                # Show class accuracy
                                st.subheader("Class-wise Accuracy")
                                ca_image = Image.open(class_accuracy_path)
                                st.image(ca_image, caption="Class-wise Accuracy", use_column_width=True)

                                # Show misclassified examples
                                st.subheader("Misclassified Examples")

                                # Filter misclassified examples
                                misclassified = [r for r in all_results if not r['correct']]

                                if misclassified:
                                    for i, result in enumerate(misclassified[:5]):  # Show first 5
                                        col1, col2 = st.columns([1, 2])

                                        with col1:
                                            # Display image
                                            img = Image.open(result['image_path'])
                                            st.image(img, caption=f"Example {i+1}", use_column_width=True)

                                        with col2:
                                            # Display prediction info
                                            st.write(f"True class: {result['true_class']}")
                                            st.write(f"Predicted class: {result['pred_class']}")

                                            # Create bar chart for top 5 predictions
                                            probs = result['probabilities']

                                            # Get top 5 indices
                                            top5_indices = np.argsort(probs)[-5:][::-1]
                                            top5_names = [class_names[i] for i in top5_indices]
                                            top5_probs = [probs[i] for i in top5_indices]

                                            # Create bar chart
                                            fig = px.bar(
                                                x=top5_probs,
                                                y=top5_names,
                                                orientation='h',
                                                labels={'x': 'Probability', 'y': 'Class'},
                                                title='Top 5 Predictions',
                                                width=400,
                                                height=300
                                            )
                                            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                                            st.plotly_chart(fig)
                                else:
                                    st.info("No misclassified examples found in the test set.")

                        except Exception as e:
                            st.error(f"Batch testing failed: {str(e)}")
                            st.exception(e)

# Tab 6: Export Model
with tabs[5]:
    st.header("Export Model")

    if not st.session_state.model_info['model_loaded']:
        st.warning("No model loaded. Please upload a model first.")
    else:
        st.subheader("Export Options")

        export_format = st.radio(
            "Select export format:",
            options=["ONNX", "TorchScript"],
            index=0
        )

        if export_format == "ONNX":
            if st.session_state.model_info.get('onnx_model_path'):
                st.success(f"ONNX model already available at: {st.session_state.model_info['onnx_model_path']}")
            else:
                st.warning("No ONNX model available. Please convert your model to ONNX first.")

                if st.button("Convert to ONNX"):
                    if st.session_state.model_info.get('pth_model_path'):
                        with st.spinner("Converting model to ONNX..."):
                            try:
                                model_dir = os.path.dirname(st.session_state.model_info['pth_model_path'])
                                model_name = os.path.splitext(os.path.basename(st.session_state.model_info['pth_model_path']))[0]

                                onnx_model_path = convert_pth_to_onnx(
                                    st.session_state.model_info['pth_model_path'],
                                    model_name,
                                    model_dir
                                )

                                st.session_state.model_info['onnx_model_path'] = onnx_model_path
                                st.success(f"Model successfully converted to ONNX: {onnx_model_path}")
                            except Exception as e:
                                st.error(f"Error converting model: {str(e)}")
                                st.exception(e)
                    else:
                        st.error("No PyTorch model available for conversion.")
        else:
            st.warning("TorchScript export is not currently implemented in this interface.")

        if st.session_state.model_info.get('onnx_model_path'):
            st.subheader("Download Model")

            with open(st.session_state.model_info['onnx_model_path'], "rb") as file:
                st.download_button(
                    label="Download ONNX Model",
                    data=file,
                    file_name=os.path.basename(st.session_state.model_info['onnx_model_path']),
                    mime="application/octet-stream"
                )

            # Option to download class names
            if st.session_state.model_info.get('class_names'):
                class_names_path = os.path.join(
                    os.path.dirname(st.session_state.model_info['onnx_model_path']),
                    'class_names.json'
                )

                with open(class_names_path, 'w') as f:
                    json.dump(st.session_state.model_info['class_names'], f)

                with open(class_names_path, "rb") as file:
                    st.download_button(
                        label="Download Class Names",
                        data=file,
                        file_name="class_names.json",
                        mime="application/json"
                    )
