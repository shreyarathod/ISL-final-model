# Indian Sign Language Interpretation CNN Model

This project presents SignSense, a convolutional neural network (CNN) model designed for interpreting Indian Sign Language (ISL). The model takes images of hand gestures as input, analyzes them, performs feature extraction, and outputs the corresponding letter the sign represents.

## Tech Stack

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Seaborn
- Streamlit

## Dataset

The dataset used for training the model is an original dataset created by us, consisting of 13,000 Indian Sign Language (ISL) images. Each sign has approximately 500 images, providing a diverse range of hand gestures representing different letters in Indian Sign Language. The dataset has been preprocessed with Gaussian blurring, grayscale conversion, and thresholding to enhance the images and reduce computational complexity.

## Preprocessing

Before feeding the images into the model, the following preprocessing steps were applied:

- **Grayscale Conversion**: The input images were converted to grayscale to simplify the image data and reduce computational complexity.
  
- **Gaussian Blurring**: Gaussian blurring was applied to reduce noise and smooth out the images, helping the model focus on important features.
  
- **Thresholding**: Thresholding was used to binarize the images, separating the background from the foreground and enhancing the contrast, which aids in feature extraction.

## Model

The model architecture consists of several key components:

- **Convolutional Layers**: Detect features in the input images using various filters.
  
- **Pooling Layers**: Reduce the dimensionality of the feature maps, retaining important information.
  
- **Dropout Layers**: Prevent overfitting by randomly deactivating a fraction of neurons during training.
  
- **Dense Layers**: Act as classifiers, processing the high-level features and making predictions.

This architecture enables the model to learn hierarchical representations of features from the input images and make accurate predictions about the hand gestures.


## Links

- Dataset: https://www.kaggle.com/datasets/dhananjayka/isl-dataset-spit


