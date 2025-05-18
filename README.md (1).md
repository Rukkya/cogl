# Plant Disease Classification System

A comprehensive PyTorch-based solution for plant disease classification with a Streamlit GUI.

## Features

- **Complete Data Pipeline**: Handles training, validation, testing, and model export
- **Interactive Streamlit GUI**: Control all aspects through a user-friendly interface
- **Visualizations**: Training curves, class distribution, sample images, and more
- **ONNX Export**: Export trained models to ONNX format for deployment
- **Advanced Analysis**: Confusion matrix, classification reports, Grad-CAM visualizations

## Project Structure

```
├── app.py                 # Main Streamlit application
├── train.py               # Training script
├── validate.py            # Validation script
├── test.py                # Testing script
├── export_onnx.py         # Script for exporting to ONNX format
├── visualize.py           # Visualization utilities
├── utils.py               # Common utility functions
└── requirements.txt       # Dependencies
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/plant-disease-classification.git
cd plant-disease-classification
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The system expects the dataset to be organized in the following structure:

```
dataset_root/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
├── valid/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── class_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
```

The PlantVillage dataset from Kaggle (https://www.kaggle.com/datasets/emmarex/plantdisease) can be used with this system.

## Usage

### Running the Streamlit App

Start the Streamlit application:

```bash
streamlit run app.py
```

This will open a browser window with the application interface.

### Using the Application

1. **Dataset Configuration**:
   - Enter the path to your dataset
   - Set the output directory for model artifacts

2. **Training**:
   - Configure training parameters (epochs, batch size, learning rate)
   - Start training with the "Start Training" button
   - Monitor training progress

3. **Visualization**:
   - View training curves, class distribution, and sample images
   - Analyze model performance through various visualizations

4. **Validation**:
   - Run validation on the validation set
   - View confusion matrix and class-wise accuracy

5. **Testing**:
   - Perform batch testing on the test set
   - Test individual images and view predictions
   - Analyze misclassified examples

6. **Export Model**:
   - Export the trained model to ONNX format
   - Download the model for deployment

### Running Individual Scripts

Each component can also be run independently:

```bash
# Training
python train.py

# Validation
python validate.py

# Testing
python test.py

# ONNX Export
python export_onnx.py

# Visualizations
python visualize.py
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Torchvision
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- ONNX
- ONNX Runtime
- Scikit-learn
- OpenCV
- Pillow

## Customization

The system is designed to be modular and easily customizable:

- **Model Architecture**: Modify `create_model()` in `utils.py` to use different architectures
- **Training Parameters**: Adjust hyperparameters in the Streamlit interface
- **Visualizations**: Add custom visualizations in `visualize.py`
- **Data Augmentation**: Customize transforms in `get_data_loaders()` in `utils.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit team for the interactive app framework
- Kaggle for hosting the PlantVillage dataset
