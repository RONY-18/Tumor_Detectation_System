# Multi-Class Brain Tumor MRI Classification

A deep learning-based system for automated classification of brain tumors from MRI scans using PyTorch. This project implements a custom Convolutional Neural Network (CNN) to distinguish between four different types of brain conditions: Glioma, Meningioma, Pituitary tumors, and No tumor (healthy brain).

## 🧠 Overview

This project is designed to assist in medical diagnosis by automatically classifying brain MRI scans into four categories:
- **Glioma**: A type of tumor that occurs in the brain and spinal cord
- **Meningioma**: A tumor that arises from the meninges
- **Pituitary**: Tumors that form in the pituitary gland
- **No Tumor**: Healthy brain tissue

The model is built from scratch using PyTorch and does not rely on pre-trained models, allowing for complete customization and understanding of the architecture.

## 🌟 Features

- **Custom CNN Architecture**: Built from scratch with 5 convolutional blocks and batch normalization
- **Multi-class Classification**: Distinguishes between 4 different brain conditions
- **Data Augmentation**: Includes rotation, flipping, color jittering for better generalization
- **Interactive Menu System**: User-friendly command-line interface
- **Comprehensive Evaluation**: Detailed metrics including confusion matrix and per-class accuracy
- **Batch Prediction**: Process multiple images at once
- **Model Persistence**: Save and load trained models
- **Training Visualization**: Plot training history and metrics
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: Adaptive learning rate adjustment

## 🛠️ Technology Stack

- **Python 3.7+**
- **PyTorch**: Deep learning framework
- **torchvision**: Image preprocessing and transformations
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization and plotting
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Evaluation metrics
- **PIL (Pillow)**: Image processing
- **OpenCV**: Computer vision operations
- **tqdm**: Progress bars

## 📋 Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
Pillow>=8.3.0
opencv-python>=4.5.0
tqdm>=4.62.0
pandas>=1.3.0
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv brain_tumor_env
   source brain_tumor_env/bin/activate  # On Windows: brain_tumor_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Dataset Structure

Organize your dataset in the following structure:

```
dataset/
├── Training/
│   ├── glioma/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── meningioma/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── no_tumor/
│   │   ├── image1.jpg
│   │   └── ...
│   └── pituitary/
│       ├── image1.jpg
│       └── ...
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

## 🎯 Usage

### Running the Application

Execute the main script to start the interactive menu:

```bash
python main.py
```

### Menu Options

1. **Train new model**: Train a fresh model from scratch
2. **Continue training**: Resume training from a saved checkpoint
3. **Predict single image**: Classify a single MRI scan
4. **Predict batch images**: Process multiple images in a folder
5. **Evaluate model**: Test model performance on validation/test set
6. **Show training history**: Visualize training progress
7. **Save/Load model**: Manage model persistence

### Quick Start Example

```python
from main import MultiClassBrainTumorClassifier

# Initialize classifier
classifier = MultiClassBrainTumorClassifier(img_size=(224, 224), batch_size=16)

# Prepare data
classifier.prepare_data("path/to/your/dataset")

# Build and train model
classifier.build_model()
classifier.train_model(epochs=100, learning_rate=0.001)

# Evaluate model
classifier.evaluate_model(use_test_set=True)

# Make predictions
classifier.predict_single_image("path/to/test/image.jpg")
```

## 🏗️ Model Architecture

The CNN architecture consists of:

- **5 Convolutional Blocks**: Each with Conv2D → BatchNorm → ReLU → MaxPool → Dropout
- **Global Average Pooling**: Reduces spatial dimensions
- **4 Fully Connected Layers**: With dropout for regularization
- **Output Layer**: 4 neurons for multi-class classification

**Model Parameters**: ~15M trainable parameters

## 📊 Performance Metrics

The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Per-class Accuracy**: Individual class performance
- **Precision, Recall, F1-Score**: Detailed classification metrics
- **Confusion Matrix**: Visual representation of classification results
- **Training History**: Loss and accuracy curves

## 🔧 Configuration Options

### Training Parameters
- **Image Size**: Default (224, 224), adjustable
- **Batch Size**: Default 16, can be modified based on GPU memory
- **Learning Rate**: Default 0.001, with adaptive scheduling
- **Epochs**: Default 100, with early stopping
- **Validation Split**: Default 20% of training data

### Data Augmentation
- Random rotation (±20°)
- Random horizontal/vertical flips
- Color jittering
- Random affine transformations
- Normalization using ImageNet statistics

## 📈 Training Tips

1. **GPU Acceleration**: The model automatically detects and uses CUDA if available
2. **Memory Management**: Reduce batch size if encountering memory issues
3. **Data Quality**: Ensure consistent image quality and proper labeling
4. **Class Balance**: Monitor class distribution for balanced training
5. **Early Stopping**: Implemented with patience=15 to prevent overfitting

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Dataset Path Error**: Verify folder structure matches requirements
3. **Import Errors**: Ensure all dependencies are installed
4. **Low Accuracy**: Check data quality and increase training epochs

### Performance Optimization

- Use GPU acceleration when available
- Optimize batch size based on hardware
- Consider mixed precision training for larger models
- Use data loading with multiple workers

## 📝 Example Results

```
Test Accuracy: 0.9234
Classification Report:
                precision    recall  f1-score   support
     glioma         0.91      0.94      0.92       100
  meningioma        0.93      0.89      0.91        98
    no_tumor        0.95      0.96      0.95       102
   pituitary        0.92      0.91      0.91        95
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Performance improvements
- New features
- Documentation improvements
- Additional evaluation metrics

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset sourced from Kaggle brain tumor classification datasets
- Inspired by recent advances in medical image analysis
- Thanks to the PyTorch community for excellent documentation

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please reach out:

- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

## 🚀 Future Enhancements

- [ ] Integration with web interface
- [ ] Support for DICOM format
- [ ] Ensemble model implementation
- [ ] Real-time inference optimization
- [ ] Mobile app development
- [ ] Integration with hospital systems
- [ ] Multi-modal input support (T1, T2, FLAIR sequences)

---

**⚠️ Medical Disclaimer**: This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.
