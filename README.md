# Advanced Computer Vision and Video Analytics
A comprehensive implementation of advanced computer vision techniques and video analytics solutions using Python, OpenCV, and deep learning frameworks.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Modules & Implementations](#modules--implementations)
- [Notebook: Image Processing & Digits Classification](#notebook-image-processing--digits-classification)
- [Notebook: Shape and Image Transformations](#notebook-shape-and-image-transformations)
- [Notebook: Edge Detection and Image Segmentation](#notebook-edge-detection-and-image-segmentation)
- [Notebook: Histogram Analysis, Equalization & DFT](#notebook-histogram-analysis-equalization--dft)
- [Notebook: Image Compression & Deep Learning Classification ](#notebook-image-compression--deep-learning-classification)
- [Notebook: Segmentation, Detection & Classification](#notebook-segmentation-detection--classification)
- [Notebook: Blob Detection, Image Enhancement & Classification](#notebook-blob-detection-image-enhancement--classification)

## Overview
This repository contains implementations of advanced computer vision algorithms and video analytics techniques including:
- Image transformations and geometric operations
- Object detection and tracking
- Face recognition and analysis
- Pose estimation
- Video processing and frame analysis
- Motion detection and activity recognition
- Semantic and instance segmentation
- 3D reconstruction

## Features
✨ **Image Processing**
- Geometric transformations (rotation, scaling, translation, shearing, reflection)
- Image filtering and enhancement
- Morphological operations
- Edge detection and contour analysis

✨ **Video Analytics**
- Real-time video processing
- Frame-level analysis
- Motion tracking
- Temporal analysis

✨ **Deep Learning Integration**
- Pre-trained model support (YOLO, ResNet, etc.)
- Custom model implementations
- Transfer learning examples

✨ **Visualization Tools**
- Matplotlib-based visualization
- Real-time plotting
- Annotated frame display

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup
```bash
git clone https://github.com/anshika1279/Computer-Vision-Implementation.git
cd Computer-Vision-Implementation
pip install -r requirements.txt
```

## Project Structure
- README.md: Project overview and instructions.
- requirements.txt: Python dependencies for all notebooks.
- image_processing_and_digits_classification.ipynb: Image resizing/blur demo plus digits classification with multiple models.
- ShapeAndImageTransformations.ipynb: Shape and image transformation examples.
- edge_detection_and_image_segmentation.ipynb: Edge detection operators and image segmentation techniques.
- Histogram_Analysis_Equalization_DFT.ipynb: Histogram analysis, contrast enhancement, and frequency domain transformations.
- image_compression_techniques_DCT_Deep_learning_image_classification.ipynb: DCT compression and CNN-based digit/object classification.
- segmentation_detection_classification.ipynb: Advanced CV pipeline with edge/region segmentation, Hough transform, YOLO/R-CNN detection, and Fashion-MNIST/CIFAR-100 classification.
- blob_detection_image_enhancement_classification.ipynb: Blob detection algorithms (LoG, DoG, DoH), comprehensive image enhancement techniques, and transfer learning with AlexNet/VGG16 on CIFAR-100.

## Modules & Implementations
- Image resizing with multiple interpolation methods and blurring with box, Gaussian, and bilateral filters.
- Digits classification using sklearn digits dataset with Gaussian Naive Bayes, RBF SVM, and Random Forest, including cross-validation and ROC visualization.

## Notebook: Image Processing & Digits Classification
- File: image_processing_and_digits_classification.ipynb
- Part 1: Resize (linear, nearest, cubic) and blur (box, Gaussian, bilateral) a local image; expects image.png in the repo root and displays a comparison grid.
- Part 2: Train/evaluate classifiers (Gaussian Naive Bayes, RBF SVM, Random Forest) on sklearn digits with 5-fold CV; prints metrics and shows ROC curves.

## Notebook: Shape and Image Transformations
- File: ShapeAndImageTransformations.ipynb
- Part 1: 2D rectangle transformations (translate, scale, rotate, reflect, shear, composite) visualized with Matplotlib.
- Part 2: Image transformations on `input.jpg` using OpenCV (translate, reflect, rotate, scale, crop, shear on x/y) with side-by-side plots.
- Part 3: Additional 2D shape transformations with reusable helpers for translate/scale/rotate/reflect/shear and composite examples.

## Notebook: Edge Detection and Image Segmentation
- File: edge_detection_and_image_segmentation.ipynb
- **Edge Detection**: Implements multiple edge detection operators including:
  - Sobel operator (combined X and Y gradients)
  - Prewitt edge detection
  - Roberts cross operator
  - Canny edge detector
- **Image Segmentation**: Demonstrates various segmentation techniques:
  - Global thresholding (fixed threshold binarization)
  - Adaptive thresholding (local neighborhood-based)
  - Watershed segmentation with morphological operations
- **Preprocessing**: Includes color space conversions (BGR→RGB→Grayscale→Binary) and image metrics calculation
- **Visualization**: Displays all results in a comprehensive grid layout with labeled subplots
- **Outputs**: Saves processed images (edge maps, segmented regions) for further analysis

## Notebook: Histogram Analysis, Equalization & DFT
- File: Histogram_Analysis_Equalization_DFT.ipynb
- **Histogram Analysis**: Computes and plots histograms for both grayscale and color (RGB) images
  - Individual channel histograms for color images (B, G, R)
  - Histogram normalization to probability distributions
  - Visualization with matplotlib for histogram analysis
- **Contrast Enhancement**: Implements histogram equalization for improving image contrast
  - Before/after comparison of grayscale images
  - Visual quality assessment with text annotations
  - Side-by-side display of original and equalized images
- **Discrete Fourier Transform (DFT)**: Frequency domain analysis and transformations
  - DFT computation with magnitude spectrum visualization
  - Inverse DFT for image reconstruction
  - Rotation property verification (45° rotation test)
  - Demonstrates spatial vs. frequency domain correspondence
- **Compatible with Google Colab**: Uses `cv2_imshow` for Colab environments
## Notebook: Image Compression & Deep Learning Classification 
- File: image_compression_techniques_DCT_Deep_learning_image_classification .ipynb
- **DCT-Based Image Compression**: Implements both lossy and lossless compression techniques
  - Lossy compression with quantization using JPEG standard quantization matrix
  - Lossless compression preserving all DCT coefficients
  - Block-wise DCT/IDCT operations (8×8 blocks)
  - Compression ratio analysis and file size comparison
- **MNIST Digit Classification**: CNN implementations for handwritten digit recognition
  - Basic 3-layer CNN architecture
  - Enhanced CNN with BatchNormalization, Dropout, and L2 regularization
  - Data augmentation (rotation, shifts, zoom)
  - Learning rate scheduling and early stopping
- **CIFAR-10 Classification**: Color image classification with CNN
  - 10-class object recognition on 32×32 color images
  - Similar architecture adapted for RGB inputs
- **Model Evaluation**: Comprehensive performance metrics
  - Classification reports with precision, recall, F1-score
  - Confusion matrices with heatmap visualization
  - ROC curves and AUC scores

## Notebook: Segmentation, Detection & Classification
- File: segmentation_detection_classification.ipynb
- **Image Segmentation**: Multiple segmentation approaches
  - Edge-based segmentation using Canny edge detection
  - Region-based segmentation with thresholding techniques
  - Visualization with matplotlib for result comparison
- **Hough Transform**: Line detection and feature extraction
  - Probabilistic Hough Line Transform for straight line detection
  - Configurable parameters for line detection sensitivity
  - Visual overlay of detected lines on original images
- **Object Detection**: State-of-the-art detection models
  - **YOLOv8**: Real-time object detection with ultralytics framework
  - **Faster R-CNN**: Region-based detection with ResNet50-FPN backbone
  - Bounding box visualization with confidence scores
  - Pre-trained models on COCO dataset for 80+ object classes
- **Deep Learning Classification**: Multi-dataset CNN training
  - **Fashion-MNIST**: Clothing classification (10 classes, 28×28 grayscale)
  - **CIFAR-100**: Fine-grained object classification (100 classes, 32×32 RGB)
  - Custom CNN architectures with Conv2D, MaxPooling, and Dense layers
  - 5-epoch training with validation accuracy tracking
  - Classification reports with precision, recall, and F1-scores
- **Integrated Pipeline**: End-to-end processing workflow combining segmentation, detection, and classification
- **Dual Environment Support**: Compatible with both local (matplotlib) and Google Colab (cv2_imshow) environments

## Notebook: Blob Detection, Image Enhancement & Classification
- File: blob_detection_image_enhancement_classification.ipynb
- **Blob Detection Algorithms**: Implementation of three advanced blob detection methods
  - **LoG (Laplacian of Gaussian)**: Scale-space blob detection with adjustable sigma parameters
  - **DoG (Difference of Gaussian)**: Efficient approximation of LoG for faster computation
  - **DoH (Determinant of Hessian)**: Hessian matrix-based blob detection for feature localization
  - Purple region extraction using HSV color masking
  - Morphological preprocessing pipeline (erosion, dilation, opening, closing, area operations)
  - Handles RGBA images with alpha channel conversion
  - Red circle overlay visualization of detected blobs
- **Image Enhancement Pipeline**: Eight comprehensive image processing techniques
  - Brightness & Contrast adjustment with alpha/beta parameters
  - Image sharpening using custom convolution kernels
  - Denoising with Non-Local Means algorithm
  - Color enhancement using PIL ImageEnhance
  - Image resizing with interpolation
  - Inverse transform (bitwise NOT operation)
  - Histogram equalization (grayscale and color via YCrCb)
  - LAB color space-based color correction
  - Grid visualization (3×3 layout) of all enhancement results
- **Transfer Learning Classification**: CIFAR-100 fine-grained classification
  - **AlexNet**: Pre-trained on ImageNet, fine-tuned for 100 classes
  - **VGG16**: Deep architecture with 16 layers, adapted for CIFAR-100
  - Modified final classifier layers for 100-class output
  - SGD optimizer with momentum (lr=0.0001, momentum=0.9)
  - Cross-entropy loss function
  - Training loop with tqdm progress bars
  - Batch size optimized for memory efficiency (batch_size=16)
  - Automatic device selection (CUDA/MPS/CPU)
  - ImageNet normalization for transfer learning compatibility
- **Model Evaluation**: Comprehensive accuracy metrics on CIFAR-100 test set
- **Multi-Image Processing**: Batch processing across multiple test images (p1.jpg, p2.jpg, p3.png, p4.png, p5.jpg)
