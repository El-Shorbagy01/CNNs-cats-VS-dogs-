# Cats vs Dogs Image Classification

## Overview
This project focuses on building an image classification model to distinguish between cats and dogs using Convolutional Neural Networks (CNNs) and transfer learning with the VGG16 model. The dataset used contains images of cats and dogs, and the goal is to achieve high accuracy in classifying these images.

## Dataset
The dataset consists of training and testing sets for both cats and dogs. Images are organized into separate folders for training and testing purposes.

- Training set paths:
    - Cats: "/content/cats_and_dogs_small/train/cats"
    - Dogs: "/content/cats_and_dogs_small/train/dogs"
    
- Testing set paths:
    - Cats: "/content/cats_and_dogs_small/test/cats"
    - Dogs: "/content/cats_and_dogs_small/test/dogs"

## Code Structure
1. **Load and Preprocess Data:**
    - Load images of cats and dogs from the provided dataset.
    - Organize the data into training and testing sets.
    - Normalize pixel values in the range [0, 1].

2. **Convolutional Neural Network (CNN) Model:**
    - Implement a CNN model for image classification.
    - Train the model with the original dataset.
    
3. **Data Augmentation with Transfer Learning:**
    - Apply data augmentation using `ImageDataGenerator` for enhanced model performance.
    - Utilize transfer learning with the VGG16 model.
    - Train the model with augmented data.

4. **VGG16 Transfer Learning:**
    - Load the VGG16 model pre-trained on ImageNet data.
    - Freeze the convolutional base of VGG16.
    - Add dense layers for classification on top of the VGG16 base.
    - Train the model using the VGG16 transfer learning approach.

5. **Results:**
    - Plot loss and accuracy curves to visualize the training and validation performance.
    - Analyze and interpret the results.

## Running the Code
1. Open the provided Colab notebook in your Google Colab environment.
2. Execute each code cell sequentially.

## Dependencies
Ensure the following Python libraries are installed in your Colab environment:

```bash
!pip install tensorflow
!pip install keras
!pip install pillow
