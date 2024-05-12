# Rice Image Classification Project

This project focuses on classifying different varieties of rice using machine learning techniques. The dataset used in this project contains images of five different types of rice: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.

## Dataset
The dataset used in this project is available on Kaggle: [Rice Image Dataset](https://www.kaggle.com/muratkokludataset/rice-image-dataset). It consists of images of rice grains belonging to each variety.

### Data Preparation
- The dataset is organized into folders, with each folder representing a rice variety.
- Images are preprocessed and resized to 224x224 pixels to fit the input shape required for the models.

## Models Implemented
### 1. Custom Convolutional Neural Network (CNN)
- A custom CNN model is built using Keras with the following architecture:
    - Input layer: Convolutional layer with ReLU activation
    - MaxPooling layer
    - Flatten layer
    - Dense layer with ReLU activation
    - Dropout layer
    - Output layer with softmax activation
- The model is trained and evaluated on the dataset.

### 2. Basic Neural Network
- A basic neural network model is implemented using Keras with the following architecture:
    - Flatten layer
    - Dense layer with ReLU activation
    - Output layer with softmax activation
- The model is trained using SGD optimizer and evaluated on the dataset.

### 3. Transfer Learning with InceptionV3
- Transfer learning is employed using the InceptionV3 pre-trained model.
- The InceptionV3 model is loaded with pre-trained ImageNet weights and appended with additional layers.
- The model is fine-tuned on the rice image dataset.

### 4. Transfer Learning with ResNet50
- Transfer learning is also applied using the ResNet50 pre-trained model.
- The base ResNet50 model is loaded with pre-trained ImageNet weights and extended with custom layers.
- The model is trained on the rice dataset while keeping the base ResNet50 layers frozen.

## Dependencies
- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- Seaborn
- NumPy
- Pandas

## Usage
1. Download the dataset from Kaggle and place it in the project directory.
2. Run the provided Jupyter notebook or Python script to train and evaluate the models.

## Results
- The performance of each model is evaluated based on accuracy and loss metrics.
- Model training progress is visualized using matplotlib.

## Conclusion
This project demonstrates the application of various machine learning techniques for rice image classification. The models implemented achieve satisfactory accuracy in classifying different rice varieties.
