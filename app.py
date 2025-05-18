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

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from train import train_model
from validate import validate_model
from ttest import test_model, predict_image
from export_onnx import export_to_onnx
from visualize import (
    visualize_training_results,
    visualize_learning_rate,
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
        'trained': False,
        'model_path': None,
        'class_names': None,
        'data_dir': None,
        'output_dir': None,
        'history_path': None,
        'onnx_path': None,
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

# Sidebar
st.sidebar.title("🌿 Plant Disease Classification")
st.sidebar.markdown("---")

# Main tabs
tabs = st.tabs([
    "📊 Dataset Configuration",
    "🧠 Training",
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
                    st.info("You may want to use data augmentation or class weights during training to address this.")

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

# Tab 2: Training
with tabs[1]:
    st.header("Model Training")

    # Training parameters
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Parameters")
        num_epochs = st.slider("Number of Epochs", min_value=1, max_value=100, value=10)
        batch_size = st.slider("Batch Size", min_value=8, max_value=128, value=32, step=8)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")

        device_options = ["cuda", "cpu"]
        device_name = st.selectbox("Device", options=device_options, index=0 if torch.cuda.is_available() else 1)

    with col2:
        st.subheader("Start Training")

        start_training = st.button("Start Training")

        # Display device info
        if torch.cuda.is_available():
            st.success(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("CUDA is not available. Training will use CPU, which may be slow.")

        # Training status
        training_status = st.empty()
        progress_bar = st.empty()

    # Start training if button is clicked
    if start_training:
        if not st.session_state.model_info['data_dir']:
            st.error("Please load a dataset first.")
        elif not st.session_state.model_info['output_dir']:
            st.error("Please set an output directory first.")
        else:
            training_status.info("Training started...")

            try:
                with st.spinner("Training in progress..."):
                    # Run training
                    model, history, class_names = train_model(
                        data_dir=st.session_state.model_info['data_dir'],
                        output_dir=st.session_state.model_info['output_dir'],
                        num_epochs=num_epochs,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        device_name=device_name
                    )

                    # Update session state
                    st.session_state.model_info['trained'] = True
                    st.session_state.model_info['model_path'] = os.path.join(
                        st.session_state.model_info['output_dir'], 'best_model.pth'
                    )
                    st.session_state.model_info['history_path'] = os.path.join(
                        st.session_state.model_info['output_dir'], 'training_history.json'
                    )

                    training_status.success("Training completed successfully!")

                    # Display final training stats
                    st.subheader("Training Results")
                    val_acc = history['val_acc'][-1]
                    train_acc = history['train_acc'][-1]

                    st.write(f"Final validation accuracy: {val_acc:.4f}")
                    st.write(f"Final training accuracy: {train_acc:.4f}")

                    # Show button to view visualizations
                    st.success("Training completed! Go to the Visualization tab to view results.")

            except Exception as e:
                training_status.error(f"Training failed: {str(e)}")
                st.exception(e)

# Tab 3: Visualization
with tabs[2]:
    st.header("Training Visualizations")

    # Check if model is trained
    if not st.session_state.model_info['trained']:
        st.warning("No trained model found. Please train a model first.")
    else:
        # Visualization options
        st.subheader("Select Visualizations")

        viz_cols = st.columns(3)

        with viz_cols[0]:
            show_training_curves = st.checkbox("Training Curves", value=True)
            show_class_distribution = st.checkbox("Class Distribution", value=True)

        with viz_cols[1]:
            show_learning_rate = st.checkbox("Learning Rate", value=True)
            show_sample_images = st.checkbox("Sample Images", value=True)

        with viz_cols[2]:
            show_button = st.button("Display Visualizations")

        # Show visualizations if button is clicked
        if show_button:
            with st.spinner("Generating visualizations..."):
                # Load history
                history_path = st.session_state.model_info['history_path']
                data_dir = st.session_state.model_info['data_dir']

                # Create visualizations
                if show_training_curves:
                    st.subheader("Training and Validation Curves")
                    fig = visualize_training_results(history_path)
                    st.pyplot(fig)

                if show_learning_rate:
                    st.subheader("Learning Rate Schedule")
                    fig = visualize_learning_rate(history_path)
                    st.pyplot(fig)

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

    # Check if model is trained
    if not st.session_state.model_info['trained']:
        st.warning("No trained model found. Please train a model first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Run Validation")
            run_validation = st.button("Start Validation")

        with col2:
            st.subheader("Validation Settings")
            validation_device = st.selectbox("Device for Validation", options=["cuda", "cpu"],
                                            index=0 if torch.cuda.is_available() else 1)

        # Run validation if button is clicked
        if run_validation:
            with st.spinner("Running validation..."):
                try:
                    # Run validation
                    validation_results = validate_model(
                        data_dir=st.session_state.model_info['data_dir'],
                        model_path=st.session_state.model_info['model_path'],
                        output_dir=os.path.join(st.session_state.model_info['output_dir'], 'validation'),
                        device_name=validation_device
                    )

                    # Display validation results
                    st.success(f"Validation completed with accuracy: {validation_results['accuracy']:.4f}")

                    # Show confusion matrix
                    st.subheader("Confusion Matrix")
                    confusion_matrix_path = validation_results['confusion_matrix_path']
                    if os.path.exists(confusion_matrix_path):
                        cm_image = Image.open(confusion_matrix_path)
                        st.image(cm_image, caption="Confusion Matrix", use_column_width=True)

                    # Show class accuracy
                    st.subheader("Class-wise Accuracy")
                    class_accuracy_path = validation_results['class_accuracy_path']
                    if os.path.exists(class_accuracy_path):
                        ca_image = Image.open(class_accuracy_path)
                        st.image(ca_image, caption="Class-wise Accuracy", use_column_width=True)

                    # Show classification report
                    st.subheader("Classification Report")
                    report_path = validation_results['classification_report_path']
                    if os.path.exists(report_path):
                        with open(report_path, 'r') as f:
                            report = json.load(f)

                        # Create a DataFrame for the report
                        report_data = []
                        for class_name in report:
                            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                report_data.append({
                                    'Class': class_name,
                                    'Precision': report[class_name]['precision'],
                                    'Recall': report[class_name]['recall'],
                                    'F1-Score': report[class_name]['f1-score'],
                                    'Support': report[class_name]['support']
                                })

                        report_df = pd.DataFrame(report_data)
                        st.dataframe(report_df)

                        # Add overall metrics
                        st.write(f"Overall Accuracy: {report['accuracy']:.4f}")
                        st.write(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
                        st.write(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")

                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")
                    st.exception(e)

# Tab 5: Testing
with tabs[4]:
    st.header("Model Testing")

    # Check if model is trained
    if not st.session_state.model_info['trained']:
        st.warning("No trained model found. Please train a model first.")
    else:
        testing_tabs = st.tabs(["Batch Testing", "Single Image Testing"])

        # Tab for batch testing
        with testing_tabs[0]:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Run Batch Testing")
                run_batch_test = st.button("Start Batch Testing")

            with col2:
                st.subheader("Testing Settings")
                test_device = st.selectbox("Device for Testing", options=["cuda", "cpu"],
                                          index=0 if torch.cuda.is_available() else 1)

            # Run batch testing if button is clicked
            if run_batch_test:
                with st.spinner("Running batch testing..."):
                    try:
                        # Run testing
                        test_results = test_model(
                            data_dir=st.session_state.model_info['data_dir'],
                            model_path=st.session_state.model_info['model_path'],
                            output_dir=os.path.join(st.session_state.model_info['output_dir'], 'test'),
                            device_name=test_device
                        )

                        # Update session state
                        st.session_state.model_info['test_results'] = test_results

                        # Display test results
                        st.success(f"Testing completed with accuracy: {test_results['accuracy']:.4f}")

                        # Show confusion matrix
                        st.subheader("Confusion Matrix")
                        confusion_matrix_path = test_results['confusion_matrix_path']
                        if os.path.exists(confusion_matrix_path):
                            cm_image = Image.open(confusion_matrix_path)
                            st.image(cm_image, caption="Confusion Matrix", use_column_width=True)

                        # Show class accuracy
                        st.subheader("Class-wise Accuracy")
                        class_accuracy_path = test_results['class_accuracy_path']
                        if os.path.exists(class_accuracy_path):
                            ca_image = Image.open(class_accuracy_path)
                            st.image(ca_image, caption="Class-wise Accuracy", use_column_width=True)

                        # Show misclassified examples
                        st.subheader("Misclassified Examples")

                        # Get misclassified examples
                        results_path = test_results['results_path']
                        if os.path.exists(results_path):
                            with open(results_path, 'r') as f:
                                all_results = json.load(f)

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
                            else:
                                st.info("No misclassified examples found in the test set.")

                    except Exception as e:
                        st.error(f"Batch testing failed: {str(e)}")
                        st.exception(e)

        # Tab for single image testing
        with testing_tabs[1]:
            st.subheader("Test a Single Image")

            uploaded_file = st.file_uploader("Choose an image to test", type=["jpg", "jpeg", "png"])

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
                                result, _ = predict_image(
                                    image_path=file_path,
                                    model_path=st.session_state.model_info['model_path'],
                                    class_names='class_info.json',
                                    device_name=test_device
                                )

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

                                # Show Grad-CAM visualization
                                st.subheader("Grad-CAM Visualization")

                                with st.spinner("Generating Grad-CAM..."):
                                    output_path = os.path.join(temp_dir, "grad_cam_result.png")

                                    try:
                                        grad_cam_fig = visualize_grad_cam(
                                            model_path=st.session_state.model_info['model_path'],
                                            image_path=file_path,
                                            class_names_path='class_info.json',
                                            output_path=output_path,
                                            device_name=test_device
                                        )

                                        st.pyplot(grad_cam_fig)
                                    except Exception as e:
                                        st.error(f"Grad-CAM generation failed: {str(e)}")

                            except Exception as e:
                                st.error(f"Prediction failed: {str(e)}")
                                st.exception(e)

# Tab 6: Export Model
with tabs[5]:
    st.header("Export Model to ONNX")

    # Check if model is trained
    if not st.session_state.model_info['trained']:
        st.warning("No trained model found. Please train a model first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ONNX Export Settings")

            onnx_filename = st.text_input("ONNX Filename", value="plant_disease_model.onnx")
            export_device = st.selectbox("Device for Export", options=["cuda", "cpu"],
                                        index=0 if torch.cuda.is_available() else 1)

        with col2:
            st.subheader("Run Export")
            export_button = st.button("Export to ONNX")

        # Run export if button is clicked
        if export_button:
            with st.spinner("Exporting model to ONNX..."):
                try:
                    # Create output path
                    onnx_dir = os.path.join(st.session_state.model_info['output_dir'], 'onnx')
                    ensure_dir(onnx_dir)
                    output_path = os.path.join(onnx_dir, onnx_filename)

                    # Run export
                    success, metadata = export_to_onnx(
                        model_path=st.session_state.model_info['model_path'],
                        output_path=output_path,
                        device_name=export_device
                    )

                    if success:
                        # Update session state
                        st.session_state.model_info['onnx_path'] = output_path

                        # Display success message
                        st.success(f"Model successfully exported to ONNX at: {output_path}")

                        # Display model metadata
                        st.subheader("Model Metadata")
                        metadata_df = pd.DataFrame([
                            {"Property": "Input Size", "Value": metadata['input_size']},
                            {"Property": "Number of Classes", "Value": metadata['num_classes']},
                            {"Property": "PyTorch Model Path", "Value": metadata['pytorch_model_path']},
                            {"Property": "ONNX Model Path", "Value": metadata['onnx_model_path']}
                        ])
                        st.table(metadata_df)

                        # Add download button
                        with open(output_path, 'rb') as f:
                            onnx_bytes = f.read()

                        st.download_button(
                            label="Download ONNX Model",
                            data=onnx_bytes,
                            file_name=onnx_filename,
                            mime="application/octet-stream"
                        )
                    else:
                        st.error("Failed to export model to ONNX format.")

                except Exception as e:
                    st.error(f"ONNX export failed: {str(e)}")
                    st.exception(e)

# Display app info in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Application Status")

# Display model info
model_status = "Trained ✅" if st.session_state.model_info['trained'] else "Not Trained ❌"
st.sidebar.write(f"Model Status: {model_status}")

if st.session_state.model_info['data_dir']:
    st.sidebar.write(f"Dataset: {os.path.basename(st.session_state.model_info['data_dir'])}")

if st.session_state.model_info['trained']:
    st.sidebar.write(f"Model Path: {st.session_state.model_info['model_path']}")

    if st.session_state.model_info['onnx_path']:
        st.sidebar.write(f"ONNX Path: {st.session_state.model_info['onnx_path']}")

# About section
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This application provides a complete workflow for plant disease classification "
    "using deep learning. It includes dataset configuration, model training, "
    "visualization, validation, testing, and model export to ONNX format."
)

