# Coral Reef Classification using ResNet50

This project applies deep learning to classify coral reef images into two categories: bleached and healthy.
The goal is to demonstrate how transfer learning with ResNet50 can be applied in environmental monitoring tasks, specifically for studying the health of coral reefs.

Coral reefs are highly sensitive ecosystems, and bleaching events caused by rising sea temperatures are a major threat.
By building an automated classification model, we can explore how computer vision techniques may assist researchers and conservationists in monitoring reef health at scale.

## Dataset

The dataset is organized into two main folders:
- healthy_corals/ (contains images of healthy coral reefs)
- bleached_corals/ (contains images of bleached/damaged coral reefs)

All images were preprocessed using data augmentation techniques (resizing, flipping, rotation, zooming, etc.) to improve generalization and handle dataset size limitations.

(Note: Due to size, the dataset is uploaded as a .zip file.)

## Methodology

The methodology followed to develop this computer vision model includes the following major steps:

1. Data Preprocessing
   - Image resizing to 224x224
   - Normalization of pixel values
   - Augmentation (random flips, rotations, zoom, brightness adjustments)

2. Model Development
   - Base model: ResNet50 (with pre-trained ImageNet weights)
   - Top layers replaced with custom fully connected layers
   - Dropout for regularization

3. Training
   - Loss function: binary_crossentropy
   - Optimizer: Adam
   - Metrics: accuracy
   - Training/validation split: 80/20
   - Epochs: 50

4. Evaluation
   - Accuracy and loss curves
   - Confusion matrix
   - Classification report (precision, recall, F1-score)

This model was developed in Google Colab as heavy training for computer vision projects is much easier with a GPU (as Google Colab provides) and then was exported as an .ipynb file.

## Results

The model achieved a validation accuracy of 73.37% with a validation loss of 1.79.

The model shows moderate performance in distinguishing between bleached and healthy corals. While the accuracy is not very high, this project demonstrates the practical application of transfer learning with ResNet50 on a real-world dataset, including image preprocessing, augmentation, and evaluation using metrics such as confusion matrix and classification report. It also highlights the potential of deep learning models for environmental monitoring tasks. Further improvements could be achieved with a larger dataset, additional fine-tuning, or experimenting with different architectures.

## How to Run?

1. Open the notebook `coral_reef_classification.ipynb` directly in Google Colab.
2. Enable GPU:
   - Go to `Runtime` > `Change runtime type` > Select `GPU`.
3. Upload the dataset (or modify the code to point to your dataset location).
4. Run all cells to train and evaluate the model.

## Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

## Author
Hajara Sabnam Kareem Navaz [ML & DS Enthusiast :)]
